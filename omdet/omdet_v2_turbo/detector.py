from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from typing import List, Tuple

import numpy as np
# import open_clip
from detectron2.structures import Instances, Boxes, ImageList
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import detector_postprocess
from detectron2.layers import batched_nms
from detectron2.modeling import build_backbone
from omdet.utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from omdet.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from omdet.omdet_v2_turbo.build_components import build_encoder_model, build_decoder_model, build_detr_head
from detectron2.config import configurable
from .qim import build as build_query_interaction_layer
from omdet.modeling.language_backbone import build_language_backbone
from detectron2.utils.logger import setup_logger
from .detr_torch import SetCriterion
from ..modeling.structures import pairwise_iou, matched_boxlist_iou
from ..modeling.backbone.swint import build_swintransformer_backbone
from ..modeling.language_backbone.clip.models import clip as clip
from .torch_utils import bbox_cxcywh_to_xyxy
__all__ = ['OmDetV2Turbo']

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from ..utils.cache import LRUCache

class ClipMatcher(nn.Module):
    def __init__(self, matcher,
                       weight_dict,
                       losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        frame_id = self._current_frame_idx - 1
        gt_instances = self.gt_instances[frame_id]
        outputs = {
            'pred_logits': track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss('labels',
                                     outputs=outputs,
                                     gt_instances=[gt_instances],
                                     indices=[(src_idx, tgt_idx)],
                                     num_boxes=1)
        self.losses_dict.update(
            {'frame_{}_track_{}'.format(frame_id, key): value for key, value in
             track_losses.items()})

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        #TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes[mask]),
            box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses
    
    def sigmoid_focal_loss(self, inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, mean_in_dim1=True):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
    
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        if mean_in_dim1:
            return loss.mean(1).sum() / num_boxes
        else:
            return loss.sum() / num_boxes

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J)
            # set labels of track-appear slots to 0.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o
        if self.focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = self.sigmoid_focal_loss(src_logits.flatten(1),
                                             gt_labels_target.flatten(1),
                                             alpha=0.25,
                                             gamma=2,
                                             num_boxes=num_boxes, mean_in_dim1=False)
            loss_ce = loss_ce.sum()
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def match_for_single_frame(self, outputs):
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances = outputs
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.

        obj_idxes = gt_instances_i.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0),
            'pred_boxes': pred_boxes_i.tensor.unsqueeze(0),
        }

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        for j in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[j].item()
            # set new target idx.
            if obj_id >= 0:
                if obj_id in obj_idx_to_gt_idx:
                    track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[obj_id]
                else:
                    num_disappear_track += 1
                    track_instances.matched_gt_idxes[j] = -1  # track-disappear case.
            else:
                track_instances.matched_gt_idxes[j] = -1

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(pred_logits_i.device)
        matched_track_idxes = (track_instances.obj_idxes >= 0)  # occu 
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(
            pred_logits_i.device)

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances_i)).to(pred_logits_i.device)
        tgt_state[tgt_indexes] = 1
        untracked_tgt_indexes = torch.arange(len(gt_instances_i)).to(pred_logits_i.device)[tgt_state == 0]
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]

        def match_for_single_decoder_layer(unmatched_outputs, matcher):
            new_track_indices = matcher(unmatched_outputs,
                                             [untracked_gt_instances])  # list[tuple(src_idx, tgt_idx)]

            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt.
            new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
                                              dim=1).to(pred_logits_i.device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].tensor.unsqueeze(0),
        }
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_cxcywh_to_xyxy(active_track_boxes.tensor)
            gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        # if 'aux_outputs' in outputs:
        #     for i, aux_outputs in enumerate(outputs['aux_outputs']):
        #         unmatched_outputs_layer = {
        #             'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
        #             'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
        #         }
        #         new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher)
        #         matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
        #         for loss in self.losses:
        #             if loss == 'masks':
        #                 # Intermediate masks losses are too costly to compute, we ignore them.
        #                 continue
        #             l_dict = self.get_loss(loss,
        #                                    aux_outputs,
        #                                    gt_instances=[gt_instances_i],
        #                                    indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
        #                                    num_boxes=1, )
        #             self.losses_dict.update(
        #                 {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
        #                  l_dict.items()})
        self._step()
        return track_instances

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses

class RuntimeTrackerBase(object):
    def __init__(self, miss_tolerance=5):
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, score_thresh, nms_thresh, track_instances: Instances):
        track_instances.disappear_time[track_instances.scores >= score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= score_thresh:
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < score_thresh * 0.9:
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # Set the obj_id to -1.
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1


# @META_ARCH_REGISTRY.register()
class OmDetV2Turbo(nn.Module):

    @configurable
    def __init__(self, cfg):
        super(OmDetV2Turbo, self).__init__()
        self.cfg = cfg
        self.logger = setup_logger(name=__name__)

        # self.backbone = build_backbone(cfg)
        self.backbone = build_swintransformer_backbone(cfg)
        self.decoder = build_decoder_model(cfg)
        self.neck = build_encoder_model(cfg)
        self.track_qim = build_query_interaction_layer(cfg)
        self.loss_head = build_detr_head(cfg)
        self.device = cfg.MODEL.DEVICE

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.normalizer = normalizer

        self.size_divisibility = self.backbone.size_divisibility
        self.nms_test_th = 0.0
        self.conf_test_th = 0.0
        self.loss_type = 'FOCAL'
        self.use_language_cache = True
        self.language_encoder_type = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        self.num_proposals = cfg.MODEL.ELADecoder.num_queries
        self.hidden_dim = cfg.MODEL.ELADecoder.hidden_dim

        # Build language Encoder
        self.language_backbone = build_language_backbone(cfg)
        self.language_cache_label = LRUCache(100)
        self.language_cache_prompt = LRUCache(100)

        # Tracker
        self.track_base = RuntimeTrackerBase()


    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        return {
            'cfg': cfg
        }

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)
        ann_types = [x["ann_type"] if "ann_type" in x else "box" for x in batched_inputs]
        return images, images_whwh, ann_types

    def gen_output(self, hs, box_cls, box_pred, batched_inputs, images, score_thresh, nms_thresh, do_postprocess,
                   max_num_det=None):

        results = self.inference(hs, box_cls, box_pred, images.image_sizes, score_thresh, nms_thresh, max_num_det)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            results = processed_results
        return results
    
    def gen_track_output(self, track_instances, pos_emb, hs, box_cls, box_pred, image_sizes, score_thresh, nms_thresh,
                   max_num_det=None):
        
        if (track_instances != None) and (len(track_instances) > 0):
            track_embs = hs[ : , max_num_det: ]
            track_boxes = box_pred[ : , max_num_det: ] if self.training else box_cxcywh_to_xyxy(box_pred[ : , max_num_det: ]) * torch.tensor(image_sizes[0]).repeat(2).to(self.device)
            track_scores = torch.sigmoid(box_cls[:, max_num_det:])
            track_instances.output_embs = track_embs[0]
            track_instances.pred_logits = box_cls[0][max_num_det: ]
            track_instances.pred_boxes = Boxes(track_boxes[0])
            track_instances.pred_boxes.clip(image_sizes[0])
        
        newborn_emb = hs[:, :max_num_det]
        newborn_cls = box_cls[:, :max_num_det]
        newborn_box = box_pred[:, :max_num_det]

        results = self.inference(pos_emb, newborn_emb, newborn_cls, newborn_box, image_sizes, track_instances, score_thresh, nms_thresh, max_num_det)

        if self.training:
            track_instances = self.criterion.match_for_single_frame(results[0])
            results[0] = track_instances
        
        results = self.track_qim(results)
        
        return results

    def inference(self, pos_emb, hs, box_cls, box_pred, image_sizes, track_instances=None, score_thresh=None, nms_thresh=None, max_num_det=None):
        assert len(box_cls) == len(image_sizes)
        if score_thresh is None:
            score_thresh = self.conf_test_th

        if nms_thresh is None:
            nms_thresh = self.nms_test_th

        num_classes = box_cls.shape[2]
        scores, labels = self.compute_score(box_cls)
        results = []
        if self.loss_type in {"FOCAL", "BCE"}:
            for i, (scores_img, box_per_img, image_size) in enumerate(zip(scores, box_pred, image_sizes
                                                                          )):
                results.append(self.inference_single_image(pos_emb, hs, box_cls, box_per_img, scores_img, labels, track_instances, image_size, num_classes,
                                                           score_thresh=score_thresh,
                                                           nms_thresh=nms_thresh,
                                                           max_num_det=max_num_det))
        else:
            for i, (scores_img, label_img, box_per_img, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                results.append(
                    self.inference_single_image(pos_emb, hs, box_cls, box_per_img, scores_img, label_img, track_instances, image_size, num_classes,
                                                score_thresh=score_thresh,
                                                nms_thresh=nms_thresh,
                                                max_num_det=max_num_det))

        return results

    def inference_single_image(self, pos_emb, hs, box_cls, boxes, scores, labels,
                               track_instances,
                               image_size: Tuple[int, int],
                               num_classes: int,
                               score_thresh: float,
                               nms_thresh: float,
                               max_num_det: int = None):
        """
        Call `fast_rcnn_inference_single_image` for all images.
        Args:
            boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
                boxes for each image. Element i has shape (Ri, K * 4) if doing
                class-specific regression, or (Ri, 4) if doing class-agnostic
                regression, where Ri is the number of predicted objects for image i.
                This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
            scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
            image_size (list[tuple]): A list of (width, height) tuples for each image in the batch.
            score_thresh (float): Only return detections with a confidence score exceeding this
                threshold.
            nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        Returns:
            instances: (list[Instances]): A list of N instances, one for each image in the batch,
                that stores the topk most confidence detections.
            kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
                the corresponding boxes/scores index in [0, Ri) from the input, for image i.
        """
        # scores_per_image: num_proposal
        # labels_per_image: num_proposal
        # box_per_images: num_proposal x 4'
        if self.loss_type in {"FOCAL", "BCE"}:
            proposal_num = len(boxes) if max_num_det is None else max_num_det
            predicted_classes = torch.argmax(scores, dim=1)
            scores_per_image = scores[torch.arange(scores.size(0)), predicted_classes]
            labels_per_image = predicted_classes
            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(proposal_num, sorted=False)
            # labels_per_image = labels[topk_indices]
            box_pred_per_image = boxes
            # box_pred_per_image = boxes.view(-1, 1, 4).repeat(1, num_classes, 1).view(-1, 4)
            # box_pred_per_image = box_pred_per_image[topk_indices]
        else:
            box_pred_per_image = boxes
            scores_per_image = scores
            labels_per_image = labels

        # Score filtering
        if self.training:
            box_pred_per_image = box_pred_per_image
        else:
            box_pred_per_image = bbox_cxcywh_to_xyxy(box_pred_per_image) * torch.tensor(image_size).repeat(2).to(self.device)
        filter_mask = scores_per_image > score_thresh  # R x K
        score_keep = filter_mask.nonzero(as_tuple=False).view(-1)
        box_pred_per_image = box_pred_per_image[score_keep]
        scores_per_image = scores_per_image[score_keep]
        labels_per_image = labels_per_image[score_keep]

        # NMS
        scores_per_image.to(self.device)
        keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, nms_thresh)
        embed = hs[:, keep]
        query_pos = pos_emb[:, keep]
        box_pred_per_image = box_pred_per_image[keep]
        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]

        # create an instance
        result = Instances(image_size)
        result.pred_boxes = Boxes(box_pred_per_image)
        result.pred_boxes.clip(image_size)
        result.obj_idxes = torch.full((len(result),), -1, dtype=torch.long, device=self.device)
        result.query_pos = query_pos[0]
        result.output_embs = embed[0]
        result.matched_gt_idxes = torch.full((len(result),), -1, dtype=torch.long, device=self.device)
        result.disappear_time = torch.zeros((len(result), ), dtype=torch.long, device=self.device)
        result.iou = torch.zeros((len(result),), dtype=torch.float, device=self.device)
        result.scores = scores_per_image
        result.pred_logits = box_cls[0][keep]
        result.pred_classes = labels_per_image

        if track_instances != None:
            result = Instances.cat([track_instances, result])

        if not self.training:
            self.track_base.update(score_thresh, nms_thresh, result)

        return result

    def compute_score(self, box_cls):
        """
        Args:
            box_cls: tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.

        Returns:
        """
        if self.loss_type in {"FOCAL", "BCE"}:
            num_classes = box_cls.shape[2]
            proposal_num = box_cls.shape[1]
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(num_classes, device=self.device). \
                unsqueeze(0).repeat(proposal_num, 1).flatten(0, 1)
        else:
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)
            # scores: batch_size x num_proposal

        return scores, labels

    def language_encode(self, batched_inputs, encode_type="task"):
        texts = batched_inputs

        if self.language_encoder_type == "clip":
            text_input = clip.tokenize(texts, truncate=True).to(self.device)

        return self.language_backbone(text_input, encode_type == "task")

    def get_cached_label_emb(self, labels):
        self.logger.info('processing labels embeddings for {}'.format(labels))
        not_cached_index = []
        not_cached_labels = []
        total_embs = []
        for idx, l in enumerate(labels):
            if self.language_cache_label.has(l):
                total_embs.append(self.language_cache_label.get(l))
            else:
                total_embs.append(None)
                not_cached_index.append(idx)
                not_cached_labels.append(l)

        self.logger.info('cached label emb num: {}, not cached num: {}'.format(len(total_embs) - len(not_cached_labels),
                                                                               len(not_cached_labels)))

        if not_cached_labels:
            embeddings = self.language_encode(not_cached_labels, encode_type="label")
            for idx, emb in enumerate(embeddings):
                idx_to_put = not_cached_index[idx]
                total_embs[idx_to_put] = emb
                self.language_cache_label.put(not_cached_labels[idx], emb)

        total_label_embs = torch.stack(total_embs).to(self.device)
        return total_label_embs

    def get_cached_prompt_emb(self, batched_tasks):
        self.logger.info('processing prompt embeddings for {}'.format(batched_tasks))
        not_cached_index = []
        not_cached_tasks = []
        total_task_features = []
        total_task_masks = []
        for idx, t in enumerate(batched_tasks):
            if self.language_cache_prompt.has(t):
                task_feature, task_mask = self.language_cache_prompt.get(t)
                total_task_features.append(task_feature)
                total_task_masks.append(task_mask)
            else:
                total_task_features.append(None)
                total_task_masks.append(None)
                not_cached_index.append(idx)
                not_cached_tasks.append(t)

        self.logger.info(
            'cached prompt emb num: {}, not cached num: {}'.format(len(total_task_features) - len(not_cached_tasks),
                                                                  len(not_cached_tasks)))

        if not_cached_tasks:
            embeddings, task_masks = self.language_encode(not_cached_tasks, encode_type="task")

            for idx in range(embeddings.shape[1]):
                emb = embeddings[:, [idx], :]
                idx_to_put = not_cached_index[idx]
                cur_mask = torch.unsqueeze(task_masks[idx], dim=0).to(self.device)
                total_task_features[idx_to_put] = emb
                total_task_masks[idx_to_put] = cur_mask
                self.language_cache_prompt.put(not_cached_tasks[idx], (emb, cur_mask))

        total_prompt_features = torch.cat(total_task_features, dim=1)
        total_prompt_masks = torch.cat(total_task_masks, dim=0).to(self.device)

        return total_prompt_features, total_prompt_masks

    def get_language_embedding(self, batched_inputs):
        batched_labels = [a["label_set"] for a in batched_inputs]
        batched_tasks = [a['tasks'] for a in batched_inputs]

        max_label_size = max([len(a) for a in batched_labels])
        label_features = []
        for i, s_labels in enumerate(batched_labels):
            pad_size = max_label_size - len(s_labels)

            label_emb = self.get_cached_label_emb(s_labels)
            label_features.append(F.pad(label_emb, (0, 0, 0, pad_size)).unsqueeze(1).to(self.device))

        label_features = torch.cat(label_features, dim=1)  # num_label x batch_size x dim_size

        # Task Features
        # prompt_features: max_task_len x batch_size x dim_size
        # prompt_mask: batch_size x max_task_len
        # batched_tasks = ['detect a person', 'detect dog and cat']
        prompt_features, prompt_mask = self.get_cached_prompt_emb(batched_tasks)

        return label_features, prompt_features, prompt_mask
    
    def forward(self, data: dict, do_postprocess=True, score_thresh=0.0, nms_thresh=1.0, debug=False):
        # Backbone
        frames = data['imgs']  # list of Tensor.
        track_instances = None
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }
        results = []
        language_input = [{
            "label_set": [data['infos'][0]['caption'][0]],
            "tasks": data['infos'][0]['caption'][0]
        }]

        if self.training:
           score_thresh=0.0
           self.criterion.num_classes = len(language_input[0]["label_set"])

        track_instances = None

        for frame_index, frame in enumerate(frames):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1
            body_feats = self.backbone(frame.unsqueeze(0).to(self.device))

            if type(body_feats) is dict:
                body_feats = [body_feats[i] for i in body_feats.keys()]

            encoder_feats = self.neck(body_feats)

            # create label and prompt embeddings
            label_feats, prompt_feats, prompt_mask = self.get_language_embedding(language_input)
            embed, hs, pos, decoder_feats = self.decoder(encoder_feats, label_feats, prompt_feats, prompt_mask, track_instances)
            box_pred, box_cls, _ = self.loss_head(decoder_feats)

            pos_emb = torch.cat((pos, embed), dim=-1)

            result = self.gen_track_output(track_instances, pos_emb, hs, box_cls, box_pred, [frame.shape[1:]],
                                      score_thresh, nms_thresh,
                                      max_num_det=self.num_proposals)
            track_instances = result
            results.append(result)
            outputs['pred_logits'].append(result.pred_logits)
            outputs['pred_boxes'].append(result.pred_boxes)

        outputs['losses_dict'] = self.criterion.losses_dict

        return outputs
    
    def forward_track(self, batched_inputs, track_instances=None, do_postprocess=True, score_thresh=0.0, nms_thresh=1.0, debug=False):
        images, images_whwh, ann_types = self.preprocess_image(batched_inputs)

        # Backbone
        body_feats = self.backbone(images.tensor)

        if type(body_feats) is dict:
            body_feats = [body_feats[i] for i in body_feats.keys()]

        encoder_feats = self.neck(body_feats)

        if not self.training:
            # create label and prompt embeddings
            label_feats, prompt_feats, prompt_mask = self.get_language_embedding(batched_inputs)
            embed, hs, pos, decoder_feats = self.decoder(encoder_feats, label_feats, prompt_feats, prompt_mask, track_instances)
            box_pred, box_cls, _ = self.loss_head(decoder_feats)

            pos_emb = torch.cat((pos, embed), dim=-1)

            result = self.gen_track_output(track_instances, pos_emb, hs, box_cls, box_pred, images.image_sizes,
                                      score_thresh, nms_thresh,
                                      max_num_det=self.num_proposals)
            track_instances = result

        return track_instances

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

def build(cfg):
    device = torch.device(cfg.MODEL.DEVICE)
    model = OmDetV2Turbo(cfg)

    img_matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            'frame_{}_loss_refer'.format(i): args.refer_loss_coef,
                            })

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    'frame_{}_aux{}_loss_refer'.format(i, j): args.refer_loss_coef,
                                    })
    if args.memory_bank_type is not None and len(args.memory_bank_type) > 0:
        memory_bank = build_memory_bank(args, d_model, hidden_dim, d_model * 2)
        for i in range(num_frames_per_batch):
            weight_dict.update({"frame_{}_track_loss_ce".format(i): args.cls_loss_coef})
    else:
        memory_bank = None
    losses = ['labels', 'boxes', 'refers']
    criterion = ClipMatcher(num_classes, matcher=img_matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(device)
    postprocessors = {}
    model = TransRMOT(
        backbone,
        transformer,
        track_embed=query_interaction_layer,
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        memory_bank=memory_bank,
        use_checkpoint=args.use_checkpoint,
    )
    return model, criterion, postprocessors