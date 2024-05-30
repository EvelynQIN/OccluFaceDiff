""" Test motion recontruction with with landmark and audio as input.
"""
import os
import random
import numpy as np
from tqdm import tqdm
import cv2
from enum import Enum
import os.path
from glob import glob
from pathlib import Path
from time import time
from collections import defaultdict
import argparse

from pathlib import Path
import torch
import imageio
from skimage.io import imread
import pickle

import sys
sys.path.append('./')
from model.deca import EMOCA
from configs.config import get_cfg_defaults

model_cfg = get_cfg_defaults().emoca
device = 'cuda'
emoca = EMOCA(model_cfg)
emoca.to(device)

def video_to_frames(video_path):
    video_cap = cv2.VideoCapture()
    if not video_cap.open(video_path):
        print(f"{video_path} open error!")
        exit(1)
    image_array = []
    while True:
        _, frame = video_cap.read()
        if frame is None:
            break
        image_array.append(frame)
    video_cap.release()
    image_array = np.stack(image_array) / 255. # (n, 224, 224, 3) in float BGR
    image_array = torch.from_numpy(image_array[:,:,:,[2,1,0]]).permute(0,3,1,2) # (n, 3, 224, 224) in RGB
    
    return  image_array.float()

video_folder = 'dataset/RAVDESS/processed/cropped_videos'
rec_folder = os.path.join('dataset/RAVDESS/processed', 'EMOCA_reconstruction')
if not os.path.exists(rec_folder):
    os.makedirs(rec_folder)
flag = True
for video in os.scandir(video_folder):
    image_array = video_to_frames(video.path).to(device)
    with torch.no_grad():
        emoca_codes = emoca(image_array)
    for key in emoca_codes:
        emoca_codes[key] = emoca_codes[key].to('cpu').numpy()
    if flag:
        for key in emoca_codes:
            print(f"{key}, {emoca_codes[key].shape}")
        flag = False
    video_id = video.name[3:-4]
    to_path = os.path.join(rec_folder, f"{video_id}.npy")
    np.save(to_path, emoca_codes)
