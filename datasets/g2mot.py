# @Author       : Ruopeng Gao
# @Date         : 2022/8/30
import os
import json
from math import floor
from random import randint

import torch
from PIL import Image
import datasets.transforms as T
# from typing import List
# from torch.utils.data import Dataset
from .mot import MOTDataset
from collections import defaultdict

import matplotlib.pyplot as plt
from detectron2.structures import Instances
from torchvision.transforms import ToPILImage


class G2MOT(MOTDataset):
    def __init__(self, cfg, split, transform):
        super(G2MOT, self).__init__(config=cfg, split=split, transform=transform)

        self.config = cfg
        self.transform = transform
        self.dataset_name = cfg.TRAINING.DATASET.NAME
        self.dataset_root = cfg.TRAINING.DATASET.DATA_ROOT 
        self.split_dir = os.path.join(cfg.TRAINING.DATASET.DATA_ROOT, self.dataset_name)
        assert os.path.exists(self.split_dir), f"Dir {self.split_dir} is not exist."

        # Sampling setting.
        self.sample_steps: list = cfg.TRAINING.SAMPLE_STEPS
        self.sample_intervals: list = cfg.TRAINING.SAMPLE_INTERVAL
        self.sample_modes: list = cfg.TRAINING.SAMPLE_MODES
        self.sample_lengths: list = cfg.TRAINING.SAMPLE_LENGTHS
        self.sample_stage = None
        self.sample_begin_frames = None
        self.sample_length = None
        self.sample_mode = None
        self.sample_interval = None
        self.sample_vid_tmax = None

        self.gts = defaultdict(lambda: defaultdict(list))
        self.vid_idx = dict()
        self.idx_vid = dict()

        with open(cfg.TRAINING.DATASET.json_path, 'r') as file:
            data = json.load(file)
            for item in data:
                if not item['is_eval']:
                    continue

                vid = item['track_path'][:-4]
                self.gts[vid]['gt_tracks'] = defaultdict(list)
                gt_path = os.path.join(cfg.TRAINING.DATASET.DATA_ROOT, item['track_path'])
                print(f'loading video {vid}')
                for line in open(gt_path):
                    t, i, *xywh, a, b, c = line.strip().split(",")[:9]
                    t, i, a, b, c = map(lambda x: int(float(x)), (t, i, a, b, c))
                    x, y, w, h = map(float, xywh)
                    self.gts[vid]['gt_tracks'][t].append([i, x, y, w, h])
                self.gts[vid]['gt_caption'].append(item['caption'])

        vids = list(self.gts.keys())

        for vid in vids:
            self.vid_idx[vid] = len(self.vid_idx)
            self.idx_vid[self.vid_idx[vid]] = vid

        self.set_epoch(0)

        return

    def __getitem__(self, item):
        vid, begin_frame = self.sample_begin_frames[item]
        frame_idxs = self.sample_frames_idx(vid=vid, begin_frame=begin_frame)
        imgs, infos = self.get_multi_frames(vid=vid, idxs=frame_idxs)
        if self.transform is not None:
            imgs, infos = self.transform(imgs, infos)
        gt_instances = []
        for img_i, targets_i in zip(imgs, infos):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
        return {
            "imgs": imgs,
            "infos": infos,
            'gt_instances': gt_instances
        }

    def __len__(self):
        assert self.sample_begin_frames is not None, "Please use set_epoch to init DanceTrack Dataset."
        return len(self.sample_begin_frames)
    
    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['areas']
        return gt_instances

    def sample_frames_idx(self, vid: int, begin_frame: int) -> list[int]:
        if self.sample_mode == "random_interval":
            assert self.sample_length > 1, "Sample length is less than 2."
            remain_frames = self.sample_vid_tmax[vid] - begin_frame
            max_interval = floor(remain_frames / (self.sample_length - 1))
            interval = min(randint(1, self.sample_interval), max_interval)
            frame_idxs = [begin_frame + interval * i for i in range(self.sample_length)]
            return frame_idxs
        else:
            raise ValueError(f"Sample mode {self.sample_mode} is not supported.")

    def set_epoch(self, epoch: int):
        self.sample_begin_frames = list()
        self.sample_vid_tmax = dict()
        self.sample_stage = 0
        for step in self.sample_steps:
            if epoch >= step:
                self.sample_stage += 1
        assert self.sample_stage < len(self.sample_steps) + 1
        self.sample_length = self.sample_lengths[min(len(self.sample_lengths) - 1, self.sample_stage)]
        self.sample_mode = self.sample_modes[min(len(self.sample_modes) - 1, self.sample_stage)]
        self.sample_interval = self.sample_intervals[min(len(self.sample_intervals) - 1, self.sample_stage)]
        for vid in self.vid_idx.keys():
            t_min = min(self.gts[vid]['gt_tracks'].keys())
            t_max = max(self.gts[vid]['gt_tracks'].keys())
            self.sample_vid_tmax[vid] = t_max
            for t in range(t_min, t_max - (self.sample_length - 1) + 1):
                self.sample_begin_frames.append((vid, t))

        return

    def get_single_frame(self, vid: str, idx: int):
        vid_path = '/'.join(vid.split('/')[:-1])
        if "DanceTrack" in vid_path:
            frame_name = f"{idx:08d}.jpg"
        elif "AnimalTrack" in vid_path:
            frame_name = f"{idx:04d}.jpg"
        else:
            frame_name = f"{idx:06d}.jpg"
        img_path = os.path.join(
            self.dataset_root,
            vid_path.replace('box_gt','frames'), "img1",
            frame_name)
        img = Image.open(img_path)
        info = {}
        ids_offset = self.vid_idx[vid] * 100000

        # 真值：
        info["boxes"] = list()
        info["obj_ids"] = list()
        info["labels"] = list()
        info["areas"] = list()
        info["img_path"] = img_path
        info["frame_idx"] = torch.as_tensor(idx)

        for i, *xywh in self.gts[vid]['gt_tracks'][idx]:
            info["boxes"].append(list(map(float, xywh)))
            info["areas"].append(xywh[2] * xywh[3])     # area = w * h
            info["obj_ids"].append(i + ids_offset)
            info["labels"].append(0)                    # DanceTrack, all people.
        info["caption"] = self.gts[vid]['gt_caption']
        info["boxes"] = torch.as_tensor(info["boxes"])
        info["areas"] = torch.as_tensor(info["areas"])
        info["obj_ids"] = torch.as_tensor(info["obj_ids"])
        info["labels"] = torch.as_tensor(info["labels"])
        # xywh to x1y1x2y2
        if len(info["boxes"]) > 0:
            info["boxes"][:, 2:] += info["boxes"][:, :2]
        else:
            info["boxes"] = torch.zeros((0, 4))
            info["ids"] = torch.zeros((0,), dtype=torch.long)
            info["labels"] = torch.zeros((0,), dtype=torch.long)

        return img, info

    def get_multi_frames(self, vid: str, idxs: list[int]):
        return zip(*[self.get_single_frame(vid=vid, idx=i) for i in idxs])


def transfroms_for_train(coco_size: bool = False, overflow_bbox: bool = False, reverse_clip: bool = False):
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]  # from MOTR
    return T.MultiCompose([
        T.MultiRandomHorizontalFlip(),
        T.MultiRandomSelect(
            T.MultiRandomResize(sizes=[640], max_size=640),
            T.MultiCompose([
                T.MultiRandomCrop(
                    min_size=384,
                    max_size=600,
                    overflow_bbox=overflow_bbox
                ),
                T.MultiRandomResize(sizes=[640], max_size=640)
            ])
        ),
        T.MultiHSV(),
        T.MultiCompose([
            T.MultiToTensor(),
            T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        T.MultiReverseClip(reverse=reverse_clip)
    ])


def transforms_for_eval():
    return T.MultiCompose([
        T.MultiRandomResize(sizes=[800], max_size=1333),
        T.MultiCompose([
            T.MultiToTensor(),
            T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ])


def build(cfg, split):
    if split == "train":
        return G2MOT(
            cfg=cfg,
            split=split,
            transform=transfroms_for_train(
            coco_size=False,
            overflow_bbox=False,
            reverse_clip=0.0
            )
        )
    elif split == "test":
        return G2MOT(config=cfg, split=split, transform=transforms_for_eval())
    else:
        raise ValueError(f"Data split {split} is not supported for DanceTrack dataset.")
