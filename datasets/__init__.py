from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
# from .dancetrack import build as build_dancetrack
# from .mot17 import build as build_mot17
# from .bdd100k import build as build_bbd100k
from .g2mot import build as build_g2mot
from .mot import MOTDataset
from omdet.utils.misc import mot_collate_fn
from utils.utils import is_distributed


def build_dataset(cfg, split) -> MOTDataset:
    # if config["DATASET"] == "DanceTrack":
    #     return build_dancetrack(config=config, split=split)
    # elif config["DATASET"] == "SportsMOT":
    #     return build_dancetrack(config=config, split=split)
    # elif config["DATASET"] == "MOT17":
    #     return build_mot17(config=config, split=split)
    # elif config["DATASET"] == "MOT17_SPLIT":
    #     return build_mot17(config=config, split=split)
    # elif config["DATASET"] == "BDD100K":
    #     return build_bbd100k(config=config, split=split)
    if cfg.TRAINING.DATASET.NAME == "G2MOT":
        return build_g2mot(cfg=cfg, split=split)
    else:
        raise ValueError(f"Dataset {cfg.DATASET.NAME} is not supported!")


def build_sampler(dataset: MOTDataset, shuffle: bool):
    if is_distributed():
        sampler = DistributedSampler(dataset=dataset, shuffle=shuffle)
    else:
        sampler = RandomSampler(dataset) if shuffle is True else SequentialSampler(dataset)
    return sampler


def build_dataloader(dataset: MOTDataset, sampler, batch_size: int, num_workers: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=mot_collate_fn,
        pin_memory=True
    )