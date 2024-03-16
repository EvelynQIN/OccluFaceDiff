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
        norm_dict,
        occlusion_mask_prob=0.5,
        normalization=True
    ):
        self.data = data
        self.mean = norm_dict['mean']
        self.std = norm_dict['std']
        self.normalization = normalization
        self.occlusion_mask_prob = occlusion_mask_prob

    def __len__(self):
        return len(self.data['lmk_2d'])

    def __getitem__(self, idx):
        
        lmk_2d = self.data['lmk_2d'][idx]   # nx2
        verts_2d = self.data['verts_2d'][idx]   # Vx2
        target = self.data['target'][idx]
        shape = target.clone()[:100]
        
        if self.normalization:
            shape = (shape - self.mean['target'][:100]) / (self.std['target'][:100] + 1e-8)
        
        # generate occlusion mask
        occlusion_mask = (1 - self.add_random_occlusion_mask(lmk_2d)).bool()
        occluded_lmk2d = lmk_2d.clone()
        occluded_lmk2d[occlusion_mask,] = 0

        # ## crop information
        # tform = self.crop(lmk_2d)
        # ## crop 
        # cropped_lmk = torch.matmul(tform, torch.hstack([lmk_2d, torch.ones([lmk_2d.shape[0],1])]).transpose(0, 1)).transpose(0, 1)[:,:2] 

        # # normalized kpt
        # cropped_lmk = cropped_lmk/self.image_size * 2  - 1
        
        return occluded_lmk2d.reshape(-1).float(), shape, lmk_2d.float(), verts_2d.float(), target.float()
    
    def add_random_occlusion_mask(self, lmk_2d):
        num_lmks = lmk_2d.shape[0]
        occlusion_mask = torch.zeros(num_lmks) # (n, v)
        add_mask = torch.bernoulli(torch.ones(1) * self.occlusion_mask_prob)[0]
        if add_mask == 0:
            return occlusion_mask
        
        occlude_center_lmk_id = torch.randint(low=0, high=num_lmks, size=(1,))[0]
        occlude_radius = torch.rand(1)[0] * 1.5
        lmk_2d_dist_to_center = torch.norm(
            lmk_2d - lmk_2d[occlude_center_lmk_id][None],
            2,
            -1
        )
        occlude_lmks = lmk_2d_dist_to_center < occlude_radius
        occlusion_mask[occlude_lmks] = 1
    
        return occlusion_mask

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

def compute_norm_dict_for_train(dataset, dataset_path):
    norm_dict_path = os.path.join(dataset_path, f'{dataset}_norm_dict.pt')
    split = 'train'
    motion_paths = glob.glob(dataset_path + "/" + dataset + "/" + split + f"/*.pt")   
    target_list, lmk3d_normed = [], []
    norm_dict = {}
    print('Compute the norm dict from training dataset.')
    for motion_path in tqdm(motion_paths):
        motion = torch.load(motion_path)
        target = motion["target"]
        lmk3d = motion['lmk_3d_normed']

        lmk3d_normed.append(lmk3d)  
        target_list.append(target)

    lmk_3d_list = torch.cat(lmk3d_normed, dim=0).reshape(-1, 3)
    target_list = torch.cat(target_list, dim=0)
    assert target_list.shape[1] == 300 + 100 + 5 * 6 

    norm_dict['mean'] = {
            "lmk_3d_normed": lmk_3d_list.mean(dim=0).float(),
            "target": target_list.mean(dim=0).float()
        }
    norm_dict['std'] = {
        "lmk_3d_normed": lmk_3d_list.std(dim=0).float(),
        "target": target_list.std(dim=0).float(),
    }
    
    with open(norm_dict_path, "wb") as f:
        torch.save(norm_dict, f)
    print(f'[Norm Dict] has saved at {norm_dict_path}')

def load_data(dataset, dataset_path, split):

    norm_dict_path = os.path.join(dataset_path, f'{dataset}_norm_dict.pt')
    if not os.path.exists(norm_dict_path):
        compute_norm_dict_for_train(dataset, dataset_path)
    norm_dict = torch.load(norm_dict_path)

    motion_paths = glob.glob(dataset_path + "/" + dataset + "/" + split + f"/*.pt")      
    lmk_2d_list, target_list= [], []
    verts2d_list = []
    print(f'[LOAD DATA] from FaMoS')
    for motion_path in tqdm(motion_paths):
        motion = torch.load(motion_path)
        skip_frames = 2 # avoid data overlap
        lmk_2d = motion["lmk_2d"][0::skip_frames]
        target = motion["target"][0::skip_frames]
        verts2d = motion['verts_2d_cropped'][0::skip_frames]

        lmk_2d_list.append(lmk_2d)  
        target_list.append(target)
        verts2d_list.append(verts2d)

    lmk_2d_list = torch.cat(lmk_2d_list, dim=0).reshape(-1, 68, 2)
    bs = lmk_2d_list.shape[0]
    target_list = torch.cat(target_list, dim=0).reshape(bs, -1)
    verts2d_list = torch.cat(verts2d_list, dim=0).reshape(bs, -1, 2)
            
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
        shuffle = True
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