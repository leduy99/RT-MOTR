import os
from datasets import build_dataset
from detectron2.config import get_cfg
from omdet.omdet_v2_turbo.config import add_omdet_v2_turbo_config

cfg = get_cfg()
add_omdet_v2_turbo_config(cfg)
cfg.merge_from_file(os.path.join('configs', 'OmDet-Turbo_tiny_SWIN_T' + '.yaml'))

data = build_dataset(cfg, 'train')