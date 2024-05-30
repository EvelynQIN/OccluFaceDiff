import glob
import os

import torch
import torch.nn as nn
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict
import random
from torchvision import transforms
import torchvision.transforms.functional as F 
from utils import utils_transform
import pickle
import h5py

class TrainMeadDataset(Dataset):
    def __init__(
        self,
        dataset_name, # list of dataset names
        dataset_path,
        split_data,
        n_shape=300,
        n_exp=50,
    ):
        self.split_data = split_data
        self.dataset = dataset_name
        self.n_shape = n_shape 
        self.n_exp = n_exp
        self.video_id_2_emotion_classs = video_id_2_emotion_class()

        # paths to processed folder
        self.processed_folder = os.path.join(dataset_path, dataset_name, 'processed')
        self.reconstruction_folder = os.path.join(self.processed_folder, 'reconstructions/EMICA-MEAD_flame2020')
       
    def __len__(self):
        return len(self.split_data)

    def _get_emica_codes(self, motion_path):
        code_dict = {}

        with h5py.File(os.path.join(self.reconstruction_folder, motion_path, 'shape_pose_cam.hdf5'), "r") as f:
        # cam : (1, n, 3)
        # exp : (1, n, 100)
        # global_pose : (1, n, 3)
        # jaw : (1, n, 3)
        # shape : (1, n, 300)
            for k in f.keys():
                code_dict[k] = torch.from_numpy(f[k][0]).float()
        code_dict['shape'] = code_dict['shape'][:,:self.n_shape]
        code_dict['exp'] = code_dict['exp'][:,:self.n_exp]
        emica_input = torch.cat([
            code_dict['shape'][:,:100], code_dict['exp'][:,:50], code_dict['jaw']], dim=-1
        )

        return emica_input # (n, 100+50+3)


    def __getitem__(self, idx):
        motion_path, seqlen = self.split_data[idx]

        x = self._get_emica_codes(motion_path)
        subject, view, emotion, level, sent = motion_path.split('/')
        label = self.video_id_2_emotion_classs[f"{emotion}_{level}"]
        return x, label

def video_id_2_emotion_class():
    
    emotions = ['neutral', 'happy', 'sad', 'surprised', 'fear', 'disgusted', 'angry', 'contempt']
    levels = ['level_1', 'level_2', 'level_3']
    class_id = 0
    video_id_2_emotion_class_mapping = {}
    for emotion in emotions:
        for level in levels:
            if emotion == 'neutral' and level != 'level_1':
                continue
            video_id_2_emotion_class_mapping[f"{emotion}_{level}"] = class_id
            class_id += 1
    return video_id_2_emotion_class_mapping

def get_dataloader(
    dataset,
    split,
    batch_size,
    num_workers=32,
):

    if split == "train" or split == 'val':
        shuffle = True
        drop_last = True
        num_workers = num_workers
    else:
        shuffle = False
        drop_last = False
        num_workers = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=False,
    )
    return loader