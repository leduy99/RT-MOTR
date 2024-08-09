import os

from .config import add_omdet_v2_turbo_config
from detectron2.config import get_cfg
from .detector import OmDetV2Turbo, ClipMatcher
from .ela_encoder import ELAEncoder
from .ela_decoder import ELADecoder
from .head import DINOHead
from .utils import load_pretrained_model


def build_training(args):
    cfg = get_cfg()
    add_omdet_v2_turbo_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.pretrain
    cfg.MODEL.DEVICE = 'cuda'
    cfg.INPUT.MAX_SIZE_TEST = 640
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.MODEL.DEPLOY_MODE = True
    cfg.freeze()

    model = OmDetV2Turbo(cfg)
    criterion = ClipMatcher(cfg)
