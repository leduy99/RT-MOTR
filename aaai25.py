import os
import json

from omdet.infernece.det_engine import DetEngine
from omdet.utils.plots import Annotator
from PIL import Image
import numpy as np

if __name__ == "__main__":
    root_folder = '/mnt/d/Research/'
    json_path = '/mnt/d/Research/G2MOT/caption_queries/animaltrack_test.json'
    output_path = '/mnt/d/Research/AAAI25/animaltrack_test'
    check = False
    with open(json_path, 'r') as file:
        data = json.load(file)
        for item in data:
            if 'goose_2' in item['video_path']:
                check = True

            if (not check) or (not item['is_eval']):
                continue
            engine = DetEngine(batch_size=1, device='cuda')
            img_folder = os.path.join(root_folder, item['video_path'])
            
            material =  item['track_path'].split('/')
            output_folder = os.path.join(output_path, material[-2])

            labels = [item['caption']]          # labels to be predicted
            prompt = item['caption']

            save_img_dir = os.path.join(output_path, material[-2], material[-1][:-4])
            res = engine.inf_predict('OmDet-Turbo_tiny_SWIN_T',    # prefix name of the pretrained checkpoints
                                task=prompt,
                                dir=img_folder,
                                save_img_dir=save_img_dir,
                                labels=labels,
                                src_type='local',                     # type of the image_paths, "local"/"url"
                                conf_threshold=0.2,
                                nms_threshold=0.5
                                )
                
            output_file = os.path.join(output_folder, material[-1])
            with open(output_file, 'w') as file:
                file.writelines(res)