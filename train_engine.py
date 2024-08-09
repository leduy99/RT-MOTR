import sys
import math
import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets

from datasets import build_dataset, build_sampler, build_dataloader
from detectron2.config import get_cfg

from omdet.utils.cache import LRUCache
from omdet.omdet_v2_turbo.matcher import build_matcher
from omdet.omdet_v2_turbo.detector import OmDetV2Turbo, ClipMatcher
from omdet.omdet_v2_turbo.config import add_omdet_v2_turbo_config

import omdet.utils.misc as utils
from utils.utils import labels_to_one_hot, is_distributed, distributed_rank, set_seed, is_main_process, \
    distributed_world_size

from log.logger import Logger, ProgressLogger
from typing import Iterable
from log.log import MetricLog
from datasets.data_prefetcher import data_prefetcher, data_dict_to_cuda
from omdet.omdet_v2_turbo.utils import load_pretrained_model

def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)

        criterion.initialize_for_single_clip(data_dict['gt_instances'])
        outputs = model(data_dict)

        loss_dict = criterion(outputs, data_dict)
        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        # gather the stats from all processes

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main():
    cfg = get_cfg()
    add_omdet_v2_turbo_config(cfg)
    cfg.merge_from_file(os.path.join('configs', 'OmDet-Turbo_tiny_SWIN_T' + '.yaml'))
    cfg.MODEL.DEVICE = 'cuda'
    cfg.INPUT.MAX_SIZE_TEST = 640
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.MODEL.DEPLOY_MODE = True
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TRAINING.AVAILABLE_GPUS[3:]

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if cfg.TRAINING.USE_DISTRIBUTED:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(distributed_rank())

    ### Later
    # train_logger = Logger(logdir=os.path.join(config["OUTPUTS_DIR"], "train"), only_main=True)
    # train_logger.show(head="Configs:", log=config)
    # train_logger.write(log=cfg, filename="config.yaml", mode="w")
    # train_logger.tb_add_git_version(git_version=config["GIT_VERSION"])

    set_seed(cfg.TRAINING.SEED)

    model = OmDetV2Turbo(cfg)
    model.to(cfg.MODEL.DEVICE)
    # for n, p in model.named_parameters():
    #     print(n)
    model_without_ddp = model

    # Load Pretrained Model
    if cfg.MODEL.WEIGHTS is not None:
        model = load_pretrained_model(model, cfg.MODEL.WEIGHTS, show_details=False)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Data process
    dataset_train = build_dataset(cfg=cfg, split="train")
    sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
    dataloader_train = build_dataloader(dataset=dataset_train, sampler=sampler_train,
                                        batch_size=cfg.TRAINING.BATCH_SIZE, num_workers=cfg.TRAINING.NUM_WORKERS)

    # Criterion
    img_matcher = build_matcher(cfg)
    num_frames_per_batch = max(cfg.TRAINING.SAMPLE_LENGTHS)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): cfg.TRAINING.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): cfg.TRAINING.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): cfg.TRAINING.giou_loss_coef,
                            })

    # TODO this is a hack
    if cfg.TRAINING.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(cfg.MODEL.ELADecoder.num_decoder_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): cfg.TRAINING.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): cfg.TRAINING.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): cfg.TRAINING.giou_loss_coef,
                                    })

    losses = ['labels', 'boxes']
    criterion = ClipMatcher(matcher=img_matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(cfg.MODEL.DEVICE)
    model.criterion = criterion

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out
    
    #Freeze unwanted params
    freeze_names = ['backbone', 'language_backbone']
    
    for name, param in model.named_parameters():
        if match_name_keywords(name, freeze_names):
            param.requires_grad = False

    #Learning rate dict for each module
    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, "decoder") and p.requires_grad],
            "lr": cfg.TRAINING.LR,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, "track_qim") and p.requires_grad],
            "lr": cfg.TRAINING.LR,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAINING.LR,
                                  weight_decay=cfg.TRAINING.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAINING.LR_drop)

    if cfg.TRAINING.USE_DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.TRAINING.AVAILABLE_GPUS[3:]], find_unused_parameters=True)
        model_without_ddp = model.module

    train_func = train_one_epoch_mot
    dataset_train.set_epoch(0)
    for epoch in range(cfg.TRAINING.num_epochs):
        if cfg.TRAINING.USE_DISTRIBUTED:
            sampler_train.set_epoch(epoch)
        train_stats = train_func(
            model, criterion, dataloader_train, optimizer, cfg.MODEL.DEVICE, epoch, cfg.TRAINING.clip_max_norm)
        lr_scheduler.step()

        #save part
        output_dir = "./exp_g2mot/"
        os.makedirs(output_dir, exist_ok=True)
        save_period = 1
        checkpoint_paths = [output_dir + 'checkpoint.pth']
        # extra checkpoint before LR drop and every 1 epochs
        if (epoch + 1) % cfg.TRAINING.LR_drop == 0 or (epoch + 1) % save_period == 0 or (((cfg.TRAINING.num_epochs >= 45 and (epoch + 1) > 45) or cfg.TRAINING.num_epochs < 45) and (epoch + 1) % 5 == 0):
            checkpoint_paths.append(output_dir + f'checkpoint{epoch:04}.pth')
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'cfg': cfg,
            }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if utils.is_main_process():
            with open(output_dir + "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")



if __name__ == '__main__':
    main()