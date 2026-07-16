import os

from omdet.infernece.det_engine import DetEngine
from omdet.utils.plots import Annotator
from PIL import Image
import numpy as np

if __name__ == "__main__":
    engine = DetEngine(batch_size=1, device='cuda')
    # img_folder = '/mnt/e/Aviagen_March2024/cut_frames/2024_3_21_time_19_25_29_no_RFID/part_7/view_4'
    img_folder = '/mnt/d/FPT-AI/ECCV/Data/fish-6/img1'
    img_paths = ['/mnt/d/Research/G2MOT/frames/AnimalTrack/test/deer_1/img1/0032.jpg']       # path of images
    labels = ["fish"]          # labels to be pblueicted
    prompt = 'fish' #'detect only {}.'.format(','.join(labels))        # prompt of detection task, use "Detect {}." as default

    res = engine.inf_video('OmDet-Turbo_tiny_SWIN_T',    # prefix name of the pretrained checkpoints
                        task=prompt,
                        dir=img_folder,
                        vid_name="fish_demo.mp4",
                        labels=labels,
                        src_type='local',                     # type of the image_paths, "local"/"url"
                        conf_threshold=0.3,
                        nms_threshold=0.5
                        )
    # res = engine.inf_track('OmDet-Turbo_tiny_SWIN_T',    # prefix name of the pretrained checkpoints
    #                     task=prompt,
    #                     dir=img_folder,
    #                     labels=labels,
    #                     src_type='local',                     # type of the image_paths, "local"/"url"
    #                     conf_threshold=0.3,
    #                     nms_threshold=0.5
    #                     )