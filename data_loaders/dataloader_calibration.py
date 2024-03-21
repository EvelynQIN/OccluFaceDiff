# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import glob
import os

import torch
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from skimage.transform import estimate_transform, warp, resize, rescale
import random

class LmkDataset(Dataset):
    def __init__(
        self,
        data,
        norm_dict
    ):
        self.data = data
        self.mean = norm_dict['mean_target']
        self.std = norm_dict['std_target']

    def __len__(self):
        return len(self.data['lmk_2d'])
    
    def inv_transform(self, target):
        
        target = target * self.std + self.mean
        
        return target

    def __getitem__(self, idx):
        lmk_2d = self.data['lmk_2d'][idx]   # nx2
        verts_2d = self.data['verts_2d'][idx]   # Vx2
        target = self.data['target'][idx]

        # normalization
        target = (target - self.mean) / (self.std + 1e-8)
        
        return lmk_2d.reshape(-1).float(), target.float(), verts_2d.float()

def load_data(dataset, dataset_path):

    motion_paths = glob.glob(dataset_path + "/" + dataset + "/train" + f"/*.pt") + \
         glob.glob(dataset_path + "/" + dataset + "/val" + f"/*.pt") + \
         glob.glob(dataset_path + "/" + dataset + "/test" + f"/*.pt")
    
    lmk_2d_list, target_list= [], []
    verts2d_list = []
    print(f'[LOAD DATA] from FaMoS')
    for motion_path in tqdm(motion_paths):
        motion = torch.load(motion_path)
        lmk_2d = motion["lmk_2d"][::2]
        target = motion["target"][::2]
        # reduce flame params (shape 100, expression 50)
        target = torch.cat([
            target[:,:100], target[:,300:350], target[:,400:]], 
            dim=-1)
        
        verts2d = motion['verts_2d_cropped'][::2]

        lmk_2d_list.append(lmk_2d)  
        target_list.append(target)
        verts2d_list.append(verts2d)

    lmk_2d_list = torch.cat(lmk_2d_list, dim=0).reshape(-1, 68, 2)
    bs = lmk_2d_list.shape[0]
    target_list = torch.cat(target_list, dim=0).reshape(bs, -1)
    verts2d_list = torch.cat(verts2d_list, dim=0).reshape(bs, -1, 2)

    norm_path = 'processed_data/FaMoS_CamCalib_norm_dict.pt'
    if os.path.exists(norm_path):
        norm_dict = torch.load(norm_path)
    else:
        mean_target = torch.mean(target_list, dim=0)
        std_target = torch.std(target_list, dim=0)

        norm_dict = {
            'mean_target': mean_target,
            'std_target': std_target
        }
        torch.save(norm_dict, norm_path)

    output = {
        "lmk_2d": lmk_2d_list,
        "target": target_list,
        "verts_2d": verts2d_list
    }

    return  output, norm_dict


def get_dataloader(
    dataset,
    split,
    batch_size,
    num_workers=32,
):

    if split == "train":
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