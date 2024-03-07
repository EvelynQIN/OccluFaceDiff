# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import glob
import os

import torch
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from skimage.transform import estimate_transform, warp, resize, rescale

class LmkDataset(Dataset):
    def __init__(
        self,
        data,
        scale,
        trans_scale=0,
        image_size=224,
    ):
        self.data = data
        self.image_size = image_size
        self.original_image_size = (300, 400)
        self.trans_scale = trans_scale
        self.scale = scale  #[scale_min, scale_max]

    def __len__(self):
        return len(self.data['lmk_2d'])

    def __getitem__(self, idx):
        
        lmk_2d = self.data['lmk_2d'][idx]   # nx2
        lmk_3d = self.data['lmk_3d'][idx]   # nx3
        
        ## crop information
        tform = self.crop(lmk_2d)
        ## crop 
        cropped_lmk = torch.matmul(tform, torch.hstack([lmk_2d, torch.ones([lmk_2d.shape[0],1])]).transpose(0, 1)).transpose(0, 1)[:,:2] 

        # normalized kpt
        cropped_lmk = cropped_lmk/self.image_size * 2  - 1
        
        return cropped_lmk.reshape(-1).float(), lmk_3d.float()

    def crop(self, kpt):
        left = torch.min(kpt[:,0]); right = torch.max(kpt[:,0]); 
        top = torch.min(kpt[:,1]); bottom = torch.max(kpt[:,1])

        old_size = (right - left + bottom - top)/2
        center = torch.FloatTensor([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        # translate center
        trans_scale = (torch.rand(2)*2 -1) * self.trans_scale
        center = center + trans_scale*old_size # 0.5

        # scale = torch.rand(1) * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size*self.scale)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        return torch.from_numpy(tform.params).float()

def get_lmks(motion_list):
    # zero the global translation and pose (rigid transformation)
    lmk_2d_list, lmk_3d_list, flame_verts_list = [], [], []
    skip_frames=2   # skip continuous frames to avoid over fitting
    for motion in tqdm(motion_list):
        lmk_2d = motion["lmk_2d"][0::skip_frames]
        lmk_3d = motion["lmk_3d_cam"][0::skip_frames]
        flame_verts = motion["flame_verts_cam"][0::skip_frames]

        lmk_2d_list.append(lmk_2d)  
        lmk_3d_list.append(lmk_3d)
        flame_verts_list.append(flame_verts)
    lmk_2d_list = torch.cat(lmk_2d_list, dim=0).reshape(-1, 68, 2)
    lmk_3d_list = torch.cat(lmk_3d_list, dim=0).reshape(-1, 68, 3)
    flame_verts_list = torch.cat(flame_verts_list, dim=0).reshape(lmk_3d_list.shape[0], -1, 3)
    return lmk_2d_list, lmk_3d_list, flame_verts_list

def load_data(dataset, dataset_path, split):

    motion_paths = glob.glob(dataset_path + "/" + dataset + "/" + split + f"/*.pt")      
    
    motion_list = [torch.load(i) for i in motion_paths]

    lmk_2d_list, lmk_3d_list = get_lmks(motion_list)
            
    output = {
        "lmk_2d": lmk_2d_list,
        "lmk_3d": lmk_3d_list,
    }
    return  output


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