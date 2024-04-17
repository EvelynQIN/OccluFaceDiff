import os  
from glob import glob 
import numpy as np
import cv2 
from tqdm import tqdm
from loguru import logger
import argparse
from utils import dataset_setting
import sys
from configs.config import get_cfg_defaults
import librosa
from transformers import Wav2Vec2Processor
from utils import dataset_setting
from model.deca import EMOCA
from model.wav2vec import Wav2Vec2Model
from skimage.transform import estimate_transform, warp
from utils.data_util import crop_np
from mmseg.apis import inference_model, init_model
from collections import defaultdict
import torch


scale_facter = 1 / 1000.0 # convert mm to m
h, w = 2048, 1334
image_size = 224
scale = 1.5



# compute the landmarks and the distance between two points for each image 
def create_training_data(path_to_dataset, device, model_cfg):
    """prepare data dict for training the model

    Returns:
        data_dict: dict of training data, one per motion sequence
            image: torch.Size([bs, 3, 224, 224])
            lmk_2d: torch.Size([bs, 68, 3])
            img_mask: torch.Size([bs, 224, 224])
            audio_emb: torch.Size([bs, 768])
            shape: torch.Size([bs, 100])
            tex: torch.Size([bs, 50])
            exp: torch.Size([bs, 50])
            pose: torch.Size([bs, 6])
            cam: torch.Size([bs, 3])
            light: torch.Size([bs, 9, 3])
    """
    
    sbjs = [x.path for x in os.scandir(path_to_dataset) if x.is_dir()]
    
    # init facial segmentator
    config_file = 'face_segmentation/deeplabv3plus_r101_512x512_face-occlusion.py'
    checkpoint_file = 'face_segmentation/deeplabv3plus_r101_512x512_face-occlusion-93ec6695.pth'
    face_segment = init_model(config_file, checkpoint_file, device='cuda:0')
    print_shape = True
    for sbj in sbjs:
        to_folder = os.path.join(sbj, "processed_data")
        for motion in os.scandir(os.path.join(sbj, "images")):
            out_fname = os.path.join(to_folder, motion.name + ".pt")
            
            data = torch.load(out_fname)

            imgs = (data['image'].permute(0,2,3,1) * 255).numpy().astype(np.uint8)
            seg_mask = []
            for img in imgs:
                seg_result = inference_model(face_segment, img)
                seg_mask.append(np.asanyarray(seg_result.pred_sem_seg.values()[0].to('cpu'))[0])

            seg_mask = np.stack(seg_mask)
            data['img_mask'] = torch.from_numpy(seg_mask).float()
            if print_shape:
                for k in data:
                    print(f"shape of {k}: {data[k].shape}")
                print_shape = False
            
            torch.save(data, out_fname)
                
                
                

if __name__ == '__main__':
    path_to_dataset = 'dataset/multiface'
    device = 'cuda'
    model_cfg = get_cfg_defaults().model
    create_training_data(path_to_dataset, device, model_cfg)
        
                
                               
                