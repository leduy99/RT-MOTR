import os
from omdet.infernece.det_engine import DetEngine
from omdet.utils.plots import Annotator
from PIL import Image
import numpy as np

# Parent folder containing all subfolders with 'img1' directories
parent_folder = '/mnt/d/FPT-AI/CVPR24/CVPR_datasets/G2MOT/frames/DanceTrack/val'

# Loop through all subdirectories to find 'img1' folders
count = 0
for root, dirs, files in os.walk(parent_folder):
    # Initialize the engine
    engine = DetEngine(batch_size=1, device='cuda')

    # Check if 'img1' exists in the current folder
    if 'img1' in dirs:
        img_folder = os.path.join(root, 'img1')  # Full path to the 'img1' folder
        img_paths = [os.path.join(img_folder, img) for img in os.listdir(img_folder) if img.endswith('.jpg')]  # Collect all images

        if img_paths:
            labels = ["person"]  # Labels to be predicted
            prompt = 'person'  # Prompt of detection task

            # Run inference
            res = engine.inf_video('OmDet-Turbo_tiny_SWIN_T',  # Prefix name of the pretrained checkpoints
                                   task=prompt,
                                   dir=img_folder,
                                   vid_name=f'test_dance{str(count).zfill(2)}.mp4',
                                   labels=labels,
                                   src_type='local',  # Type of the image_paths, "local"/"url"
                                   conf_threshold=0.4,
                                   nms_threshold=0.5)
            
            count += 1

            # You can add code here to save or process the results as needed
