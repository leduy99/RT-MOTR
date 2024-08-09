from detectron2.config import CfgNode as CN
from omdet.modeling.backbone.config import add_backbone_config


def add_omdet_v2_turbo_config(cfg):
    """
    Add config for Modulated OmDet Turn.
    """
    cfg.MODEL.HEAD = "DINOHead"
    cfg.MODEL.LOSS = "DINOLoss"
    cfg.MODEL.TRANSFORMER_ENCODER = "ELAEncoder"
    cfg.MODEL.TRANSFORMER_DECODER = "ELADecoder"
    cfg.MODEL.TRACK_EMBEDDING_UPDATER = "QueryInteractionModule"

    cfg.MODEL.LANGUAGE_BACKBONE = CN()
    cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE = "clip"
    cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM = 512

    # Task Head
    cfg.MODEL.ELAEncoder = CN()
    cfg.MODEL.ELAEncoder.in_channels = [192, 384, 768]
    cfg.MODEL.ELAEncoder.feat_strides = [8, 16, 32]
    cfg.MODEL.ELAEncoder.hidden_dim = 256
    cfg.MODEL.ELAEncoder.use_encoder_idx = [2]
    cfg.MODEL.ELAEncoder.num_encoder_layers = 1
    cfg.MODEL.ELAEncoder.encoder_layer = 'TransformerLayer'
    cfg.MODEL.ELAEncoder.pe_temperature = 10000
    cfg.MODEL.ELAEncoder.expansion = 1.0
    cfg.MODEL.ELAEncoder.depth_mult = 1.0
    cfg.MODEL.ELAEncoder.act = 'silu'
    cfg.MODEL.ELAEncoder.eval_size = None
    cfg.MODEL.ELAEncoder.dim_feedforward=1024

    cfg.MODEL.ELADecoder = CN()
    cfg.MODEL.ELADecoder.hidden_dim = 256
    cfg.MODEL.ELADecoder.num_queries = 300
    cfg.MODEL.ELADecoder.position_embed_type = 'sine'
    cfg.MODEL.ELADecoder.backbone_feat_channels = [256, 256, 256]
    cfg.MODEL.ELADecoder.feat_strides = [8, 16, 32]
    cfg.MODEL.ELADecoder.num_levels = 3
    cfg.MODEL.ELADecoder.num_decoder_points = 4
    cfg.MODEL.ELADecoder.nhead = 8
    cfg.MODEL.ELADecoder.num_decoder_layers = 3
    cfg.MODEL.ELADecoder.dim_feedforward = 1024
    cfg.MODEL.ELADecoder.dropout = 0.0
    cfg.MODEL.ELADecoder.activation = 'relu'
    cfg.MODEL.ELADecoder.num_denoising = 100
    cfg.MODEL.ELADecoder.label_noise_ratio = 0.5
    cfg.MODEL.ELADecoder.box_noise_scale = 1.0
    cfg.MODEL.ELADecoder.learnt_init_query = True
    cfg.MODEL.ELADecoder.eval_size = None
    cfg.MODEL.ELADecoder.eval_idx = -1
    cfg.MODEL.ELADecoder.eps = 1e-2
    cfg.MODEL.ELADecoder.cls_type = 'cosine'

    cfg.MODEL.QueryInteractionModule = CN()
    cfg.MODEL.QueryInteractionModule.dim_in = 256
    cfg.MODEL.QueryInteractionModule.hidden_dim = 1024
    cfg.MODEL.QueryInteractionModule.random_drop = 0.1
    cfg.MODEL.QueryInteractionModule.fp_ratio = 0.3
    cfg.MODEL.QueryInteractionModule.update_query_pos = False
    cfg.MODEL.QueryInteractionModule.merger_dropout = 0

    cfg.MODEL.FUSE_TYPE = None

    cfg.INPUT.RANDOM_CROP = None
    cfg.INPUT.RANDOM_CONTRAST = None
    cfg.INPUT.RANDOM_BRIGHTNESS = None
    cfg.INPUT.RANDOM_SATURATION = None

    cfg.MODEL.DEPLOY_MODE = False

    cfg.TRAINING = CN()
    cfg.TRAINING.DATASET = CN()
    cfg.TRAINING.DATASET.NAME = None
    cfg.TRAINING.DATASET.json_path = ''
    cfg.TRAINING.DATASET.DATA_ROOT = ''

    cfg.TRAINING.SAMPLE_STEPS = None
    cfg.TRAINING.SAMPLE_MODES = None
    cfg.TRAINING.SAMPLE_INTERVAL = None
    cfg.TRAINING.SAMPLE_LENGTHS = None

    cfg.TRAINING.SEED = None
    cfg.TRAINING.USE_DISTRIBUTED = None
    cfg.TRAINING.AVAILABLE_GPUS = ''
    cfg.TRAINING.BATCH_SIZE = 1
    cfg.TRAINING.NUM_WORKERS = 4

    cfg.TRAINING.set_cost_class = 2
    cfg.TRAINING.set_cost_bbox = 5
    cfg.TRAINING.set_cost_giou = 2
    cfg.TRAINING.mask_loss_coef = 1
    cfg.TRAINING.dice_loss_coef = 1
    cfg.TRAINING.cls_loss_coef = 2
    cfg.TRAINING.bbox_loss_coef = 5
    cfg.TRAINING.giou_loss_coef = 2
    cfg.TRAINING.focal_alpha = 0.25

    cfg.TRAINING.LR = 2e-4
    cfg.TRAINING.LR_drop = 40
    cfg.TRAINING.clip_max_norm = 0.1
    cfg.TRAINING.weight_decay = 1e-4

    cfg.TRAINING.num_epochs = 50
    cfg.TRAINING.aux_loss = False
    add_backbone_config(cfg)