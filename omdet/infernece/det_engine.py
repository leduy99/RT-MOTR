import os
import copy
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Dict
from omdet.utils.tools import chunks
from omdet.utils.plots import Annotator
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer as Trainer
from omdet.utils.cache import LRUCache
from omdet.infernece.base_engine import BaseEngine
from detectron2.utils.logger import setup_logger
from detectron2.modeling import detector_postprocess
from omdet.omdet_v2_turbo.detector import OmDetV2Turbo
from omdet.omdet_v2_turbo.config import add_omdet_v2_turbo_config


class DetEngine(BaseEngine):
    def __init__(self, model_dir='resources/', device='cpu', batch_size=10):
        self.model_dir = model_dir
        self._models = LRUCache(10)
        self.device = device
        self.batch_size = batch_size
        self.logger = setup_logger(name=__name__)

    def _init_cfg(self, cfg, model_id):
        cfg.MODEL.WEIGHTS = "./exp_1/checkpoint.pth" #os.path.join(self.model_dir, model_id+'.pth')
        cfg.MODEL.DEVICE = self.device
        cfg.INPUT.MAX_SIZE_TEST = 640
        cfg.INPUT.MIN_SIZE_TEST = 640
        cfg.MODEL.DEPLOY_MODE = True
        cfg.freeze()
        return cfg

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters())

    def _load_model(self, model_id):
        if not self._models.has(model_id):
            cfg = get_cfg()
            add_omdet_v2_turbo_config(cfg)
            cfg.merge_from_file(os.path.join('configs', model_id + '.yaml'))
            cfg = self._init_cfg(cfg, model_id)
            model = OmDetV2Turbo(cfg)
            self.logger.info("Model:\n{}".format(model))
            checkpoint = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
            load_res = model.load_state_dict(checkpoint['model'], strict=False)
            print(load_res)
            print("Loading a OmDet model {}".format(cfg.MODEL.WEIGHTS))
            model.eval()
            model.to(cfg.MODEL.DEVICE)
            print("Total parameters: {}".format(self.count_parameters(model)))
            self._models.put(model_id, (model, cfg))

        return self._models.get(model_id)

    def inf_predict(self, model_id,
                  dir: str,
                  save_img_dir: str,
                  task: Union[str, List],
                  labels: List[str],
                  src_type: str = 'local',
                  conf_threshold: float = 0.5,
                  nms_threshold: float = 0.5
                  ):

        if len(task) == 0:
            raise Exception("Task cannot be empty.")

        model, cfg = self._load_model(model_id)

        res = []
        resp = []
        flat_labels = labels
        track_instances = None

        count = 0
        for img in sorted(os.listdir(dir)):
            count += 1
            data = [os.path.join(dir, img)]
            with torch.no_grad():
                for batch in chunks(data, self.batch_size):
                    batch_image = self._load_data(src_type, cfg, batch)
                    for img in batch_image:
                        img['label_set'] = labels
                        img['tasks'] = task

                    batch_y = model.forward_track(batch_image, track_instances, score_thresh=conf_threshold, nms_thresh=nms_threshold)
                    track_instances = copy.deepcopy(batch_y)

                    processed_results = []
                    for results_per_image, input_per_image in zip([batch_y], batch_image):
                        height = input_per_image.get("height")
                        width = input_per_image.get("width")
                        r = detector_postprocess(results_per_image, height, width)
                        processed_results.append({"instances": r})
                    batch_y = processed_results

                    for z in batch_y:
                        temp = []
                        instances = z['instances'].to('cpu')

                        for idx, pred in enumerate(zip(instances.pred_boxes, instances.scores, instances.obj_idxes, instances.pred_classes)):
                            (x, y, xx, yy), conf, id, cls = pred
                            w = xx - x
                            h = yy - y
                            conf = float(conf)
                            cls = flat_labels[int(cls)]

                            line = f"{count}, {id}, {x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}, {conf:.2f}, -1, -1, -1\n"
                            resp.append(line)

                            temp.append({'xmin': int(x),
                                        'ymin': int(y),
                                        'xmax': int(xx),
                                        'ymax': int(yy),
                                        'conf': conf,
                                        'label': f'{id}: {cls}'})
                            
                        res.append(temp)

                        im = Image.open(batch[0])
                        a = Annotator(np.ascontiguousarray(im), font_size=16, line_width=4, pil=True, font='sample_data/simsun.ttc')
                        for R in res[count - 1]:
                            a.box_label([R['xmin'], R['ymin'], R['xmax'], R['ymax']],
                                        label=f"{R['label']} {str(int(R['conf'] * 100))}%",
                                        color='red')
                            
                        if not os.path.exists(save_img_dir):
                            os.makedirs(save_img_dir)

                        image = a.result()
                        img = Image.fromarray(image)
                        img.save(save_img_dir + '/' + batch[0].split('/')[-1])

        return resp
    
    def inf_track(self, model_id,
                  dir: str,
                  task: Union[str, List],
                  labels: List[str],
                  src_type: str = 'local',
                  conf_threshold: float = 0.5,
                  nms_threshold: float = 0.5
                  ):

        if len(task) == 0:
            raise Exception("Task cannot be empty.")

        model, cfg = self._load_model(model_id)

        resp = []
        flat_labels = labels
        track_instances = None
        out_folder = './outputs'

        count = 0
        for img in os.listdir(dir):
            count += 1
            data = [os.path.join(dir, img)]
            with torch.no_grad():
                for batch in chunks(data, self.batch_size):
                    batch_image = self._load_data(src_type, cfg, batch)
                    for img in batch_image:
                        img['label_set'] = labels
                        img['tasks'] = task

                    batch_y = model.forward_track(batch_image, track_instances, score_thresh=conf_threshold, nms_thresh=nms_threshold)
                    track_instances = copy.deepcopy(batch_y)

                    processed_results = []
                    for results_per_image, input_per_image in zip([batch_y], batch_image):
                        height = input_per_image.get("height")
                        width = input_per_image.get("width")
                        r = detector_postprocess(results_per_image, height, width)
                        processed_results.append({"instances": r})
                    batch_y = processed_results

                    for z in batch_y:
                        temp = []
                        instances = z['instances'].to('cpu')

                        for idx, pred in enumerate(zip(instances.pred_boxes, instances.scores, instances.obj_idxes, instances.pred_classes)):
                            (x, y, xx, yy), conf, id, cls = pred
                            conf = float(conf)
                            cls = flat_labels[int(cls)]

                            temp.append({'xmin': int(x),
                                        'ymin': int(y),
                                        'xmax': int(xx),
                                        'ymax': int(yy),
                                        'conf': conf,
                                        'label': f'{id}: {cls}'})
                        resp.append(temp)

                        im = Image.open(batch[0])
                        a = Annotator(np.ascontiguousarray(im), font_size=16, line_width=4, pil=True, font='sample_data/simsun.ttc')
                        for R in resp[count - 1]:
                            a.box_label([R['xmin'], R['ymin'], R['xmax'], R['ymax']],
                                        label=f"{R['label']} {str(int(R['conf'] * 100))}%",
                                        color='red')
                            
                        if not os.path.exists(out_folder):
                            os.mkdir(out_folder)

                        image = a.result()
                        img = Image.fromarray(image)
                        img.save('outputs/' + batch[0].split('/')[-1])

        return resp