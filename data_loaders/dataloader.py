# Copyri[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_motion_length]ion_length][start_id:start_id + input_motion_length][start_id:start_id + input_motion_length][start_id:start_id + input_motion_targetght (c) Meta Platforms, Inc. All Rights Reserved
import glob
import os

import torch
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict
import random
from utils.landmark_mask import REGIONS

class TrainDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        norm_dict,
        data,
        input_motion_length=120,
        train_dataset_repeat_times=1,
        no_normalization=True,
        occlusion_mask_prob=0.5,
        mixed_occlusion_prob=0.3,
        fps=30
    ):
        self.dataset_name = dataset_name
        self.mean = norm_dict['mean']
        self.std = norm_dict['std']
        self.data = data
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization
        self.input_motion_length = input_motion_length
        self.occlusion_mask_prob = occlusion_mask_prob
        self.num_mask_regions = len(REGIONS)
        self.mask_regions = list(REGIONS.keys())
        self.mixed_occlusion_prob = mixed_occlusion_prob
        self.fps = fps
        self.skip_frames = 2 if self.fps == 30 else 1 # downsample to half

    def __len__(self):
        return len(self.data['target']) * self.train_dataset_repeat_times

    def inv_transform(self, target):
        
        target = target * self.std["target"] + self.mean["target"]
        
        return target

    def __getitem__(self, idx):
        id = idx % len(self.data['target'])

        seqlen = self.data['target'][id].shape[0]
        
        if self.train_dataset_repeat_times == 1:
            # do not repeat
            input_motion_length = seqlen 
        elif self.input_motion_length is None:  
            # in transformer, randomly clip a subseq
            input_motion_length = torch.randint(min(50, seqlen), seqlen + 1, (1,))[0]
        else:
            # fix motion len
            input_motion_length = self.input_motion_length 

        if seqlen == input_motion_length: 
            start_id = 0
        else:
            start_id = torch.randint(0, int(seqlen - input_motion_length), (1,))[0]     # random crop a motion seq
        
        lmk_2d = self.data['lmk_2d'][id][start_id:start_id + input_motion_length:self.skip_frames]  # (n, 68, 2)
        verts_2d = self.data['verts_2d'][id][start_id:start_id + input_motion_length:self.skip_frames]  # (n, 5023, 2)
        lmk_3d_normed = self.data['lmk_3d_normed'][id][start_id:start_id + input_motion_length:self.skip_frames] # (n, 68, 3)
        target = self.data['target'][id][start_id:start_id + input_motion_length:self.skip_frames] # (n, shape300 + exp100 + rot6d30)
        motion_len_downsampled = target.shape[0]
        
        img_mask = self.data['img_mask'][id]
        n_imgs = torch.sum(img_mask[start_id:start_id + input_motion_length:self.skip_frames])
        
        if n_imgs > 0:
            # taking the average
            img_start_fid = torch.sum(img_mask[:start_id])
            mica_shape = self.data['mica_shape'][id][img_start_fid:img_start_fid+n_imgs] # (n_imgs, 100)
        else:
            # using other frames
            mica_shape = self.data['mica_shape'][id]
        
        mica_shape_avg = torch.mean(mica_shape, dim=0)  # (100,)
            
        # Normalization 
        if not self.no_normalization:    
            lmk_3d_normed = ((lmk_3d_normed.reshape(-1, 3) - self.mean['lmk_3d_normed']) / (self.std['lmk_3d_normed'] + 1e-8)).reshape(motion_len_downsampled, -1, 3)
            target = (target - self.mean['target']) / (self.std['target'] + 1e-8)
            mica_shape_avg = (mica_shape_avg - self.mean['target'][:100]) / (self.std['target'][:100] + 1e-8)
        
        # add random occlusion mask
        add_mask = torch.bernoulli(torch.ones(1) * self.occlusion_mask_prob)[0]
        mixed_occlusion = torch.bernoulli(torch.ones(1) * self.mixed_occlusion_prob)[0]
        occlusion_mask = self.add_random_occlusion_mask(lmk_2d, add_mask)
        if add_mask and mixed_occlusion:
            occlusion_mask += self.add_random_occlusion_mask(lmk_2d, add_mask)
            occlusion_mask[occlusion_mask > 1] = 1
        
        return target.float(), lmk_2d.float(), lmk_3d_normed.float(), verts_2d.float(), mica_shape_avg.float(), occlusion_mask
    
    def add_random_occlusion_mask(self, lmk_2d, add_mask):
        input_motion_length, num_lmks = lmk_2d.shape[:2]
        occlusion_mask = torch.zeros(input_motion_length, num_lmks) # (n, v)
        
        if add_mask == 0:
            return occlusion_mask
        
        # select occlusion type
        occlusion_type = torch.randint(low=0, high=4, size=(1,))[0]

        if occlusion_type == 0:
            # occlude fixed set of lmks
            occlude_center_lmk_id = torch.randint(low=0, high=num_lmks, size=(1,))[0]
            occlude_radius = torch.rand(1)[0] * 1.5
            lmk_2d_dist_to_center = torch.norm(
                lmk_2d[0] - lmk_2d[0, occlude_center_lmk_id][None],
                2,
                -1
            )
            occlude_lmks = lmk_2d_dist_to_center < occlude_radius
            occlusion_mask[:, occlude_lmks] = 1
        elif occlusion_type == 1:
            # occlude random set of lmks for each frame
            for i in range(input_motion_length):
                occlude_center_lmk_id = torch.randint(low=0, high=num_lmks, size=(1,))[0]
                occlude_radius = torch.rand(1)[0] * 1.5
                lmk_2d_dist_to_center = torch.norm(
                    lmk_2d[i] - lmk_2d[i, occlude_center_lmk_id][None],
                    2,
                    -1
                )
                occlude_lmks = lmk_2d_dist_to_center < occlude_radius
                occlusion_mask[i, occlude_lmks] = 1
        elif occlusion_type == 2:
            # mask out predefined regions for all frames
            mask_region_id = torch.randint(low=0, high=self.num_mask_regions, size=(1,))[0]
            mask_lmk_ids = torch.LongTensor(REGIONS[self.mask_regions[mask_region_id]])
            occlusion_mask[:, mask_lmk_ids] = 1
            
        else:
            # occlude random num of frames
            num_occluded_frames = torch.randint(low=1, high=input_motion_length//2, size=(1,))[0]
            occlude_frame_ids =  torch.LongTensor(random.sample(range(input_motion_length), num_occluded_frames))
            occlusion_mask[occlude_frame_ids] = 1
        return occlusion_mask
    
class TestDataset(Dataset):
    def __init__(
        self,
        dataset,
        norm_dict,
        motions,
        no_normalization=True,
        occlusion_mask_prob=0,
    ):
        self.dataset = dataset
        self.mean = norm_dict['mean']
        self.std = norm_dict['std']
        self.motions = motions
        self.no_normalization = no_normalization
        self.occlusion_mask_prob = occlusion_mask_prob

    def __len__(self):
        return len(self.motion_path_list)

    def inv_transform(self, target):
        
        target = target * self.std["target"] + self.mean["target"]
        
        return target

    def __getitem__(self, idx):
        
        id = idx % len(self.motion_path_list)

        motion_dict =self.motions[id]
        
        seqlen = motion_dict['target'].shape[0]
        
        lmk_2d = motion_dict['lmk_2d']  # (n, 68, 2)
        lmk_3d_normed = motion_dict['lmk_3d_normed'] # (n, 68, 3)
        target = motion_dict['target'] # (n, shape300 + exp100 + rot30 + trans3)
        motion_id = os.path.split(self.motion_path_list[id])[1].split('.')[0]
        
        
        n_imgs = torch.sum(motion_dict['img_mask'])
        img_arr = motion_dict['arcface_input'] # (n_imgs, 3, 112, 112)

        # make sure there are always 4 images 
        needed_imgs = 4 - n_imgs
        if needed_imgs < 0:
            # sample 4 images with equal intervals
            img_ids = torch.arange(0, n_imgs, n_imgs // 4)[:4]
            img_arr = img_arr[img_ids]
        elif needed_imgs > 0:
            # repeat needed images
            img_arr_added_ids = torch.randint(0, n_imgs, size=(needed_imgs,))
            img_arr_repeated = motion_dict['arcface_input'][img_arr_added_ids]
            img_arr = torch.cat([img_arr, img_arr_repeated], dim=0)
        assert (not img_arr.isnan().any()) and img_arr.shape[0] == 4
            
        # Normalization 
        if not self.no_normalization:    
            lmk_3d_normed = ((lmk_3d_normed.reshape(-1, 3) - self.mean['lmk_3d_normed']) / (self.std['lmk_3d_normed'] + 1e-8)).reshape(seqlen, -1, 3)
            target = (target - self.mean['target']) / (self.std['target'] + 1e-8)
        
        # add random occlusion mask
        occlusion_mask = self.add_random_occlusion_mask(lmk_2d)  
        
        return target.float(), lmk_2d.float(), lmk_3d_normed.float(), img_arr.float(), occlusion_mask, motion_id

    def add_random_occlusion_mask(self, lmk_2d, **model_kwargs):
        input_motion_length, num_lmks = lmk_2d.shape[:2]
        occlusion_mask = torch.zeros(input_motion_length, num_lmks) # (n, v)
        add_mask = torch.bernoulli(torch.ones(1) * self.occlusion_mask_prob)[0]
        if add_mask == 0:
            return occlusion_mask
        
        # select occlusion type
        occlusion_type = model_kwargs.get("occlusion_type", torch.randint(low=0, high=3, size=(1,))[0])

        if occlusion_type == 0:
            # occlude fixed set of lmks
            if "occlude_lmks" in model_kwargs:
                occlude_lmks  = model_kwargs["occlude_lmks"]
            else:
                occlude_center_lmk_id = model_kwargs.get("occlude_center_lmk_id", torch.randint(low=0, high=num_lmks, size=(1,))[0])
                occlude_radius = model_kwargs.get("occlude_radius", torch.rand(1)[0] * 1.5)
                lmk_2d_dist_to_center = torch.norm(
                    lmk_2d[0] - lmk_2d[0, occlude_center_lmk_id][None],
                    2,
                    -1
                )
                occlude_lmks = lmk_2d_dist_to_center < occlude_radius
            occlusion_mask[:, occlude_lmks] = 1
        elif occlusion_type == 1:
            # occlude random set of lmks for each frame
            for i in range(input_motion_length):
                occlude_center_lmk_id = torch.randint(low=0, high=num_lmks, size=(1,))[0]
                occlude_radius = torch.rand(1)[0] * 1.5
                lmk_2d_dist_to_center = torch.norm(
                    lmk_2d[i] - lmk_2d[i, occlude_center_lmk_id][None],
                    2,
                    -1
                )
                occlude_lmks = lmk_2d_dist_to_center < occlude_radius
                occlusion_mask[i, occlude_lmks] = 1
        else:
            # occlude random num of frames
            if "occlude_frame_ids" in model_kwargs:
                occlude_frame_ids = model_kwargs["occlusion_type"]
            else:
                num_occluded_frames = torch.randint(low=1, high=input_motion_length//2, size=(1,))[0]
                occlude_frame_ids =  torch.LongTensor(random.sample(range(input_motion_length), num_occluded_frames))

            occlusion_mask[occlude_frame_ids] = 1
        return occlusion_mask


def get_path(dataset_path, dataset, split, subject_id=None, motion_list=None):
    if subject_id is None:
        if motion_list is None:
            files = glob.glob(dataset_path + "/" + dataset + "/" + split + "/*pt")
        else:
            files = []
            for motion_id in motion_list:
                f = glob.glob(dataset_path + "/" + dataset + "/" + split + f"/*{motion_id}.pt")
                files.extend(f)        
    else:
        if motion_list is None:
            files = glob.glob(dataset_path + "/" + dataset + "/" + split + f"/subject_{subject_id}*pt")
        else:
            files = []
            for motion_id in motion_list:
                f = glob.glob(dataset_path + "/" + dataset + "/" + split + f"/subject_{subject_id}_{motion_id}*pt")
                files.extend(f)
    return files

def get_mean_std_path(dataset):
    return dataset + "_norm_dict.pt"

def get_face_motion(motion_paths, n_shape):
    motion_list = defaultdict(list)

    print(f"Load motions from processed data.")
    for motion_path in tqdm(motion_paths):
        motion = torch.load(motion_path)
        nframes = motion['target'].shape[0]
        if nframes < 50:
            continue
        motion_id = os.path.split(motion_path)[-1].split(".")[0]
        motion_list['motion_id'].append(motion_id)
        motion_list['target'].append(motion['target'])
        motion_list['lmk_3d_normed'].append(motion['lmk_3d_normed'])
        motion_list['lmk_2d'].append(motion['lmk_2d'])
        motion_list['verts_2d'].append(motion['verts_2d_cropped'])
        motion_list['mica_shape'].append(motion['mica_shape'][:,:n_shape])
        motion_list['img_mask'].append(motion['img_mask'])    
        
        if 'head_rotation' in motion_id: # augment the rotation data sequence
            motion_list['motion_id'].append(motion_id)
            motion_list['target'].append(motion['target'])
            motion_list['lmk_3d_normed'].append(motion['lmk_3d_normed'])
            motion_list['lmk_2d'].append(motion['lmk_2d'])
            motion_list['verts_2d'].append(motion['verts_2d_cropped'])
            motion_list['mica_shape'].append(motion['mica_shape'][:,:n_shape])
            motion_list['img_mask'].append(motion['img_mask'])    
        assert torch.sum(motion['img_mask']) ==  motion['mica_shape'].shape[0], f"{motion_path} mica prediciton not aligned with frames!"
       
    return motion_list

def load_data(args, dataset, dataset_path, split, subject_id = None, selected_motion_ids = None):
    """
    Collect the data for the given split

    Args:
        - For test:
            dataset : the name of the testing dataset
            split : test or train
        - For train:
            dataset : the name of the training dataset
            split : train or test
            input_motion_length : the input motion length
    """

    motion_paths = get_path(dataset_path, dataset, split, subject_id, selected_motion_ids)
    n_shape = args.n_shape 
    motions = get_face_motion(motion_paths, n_shape)

    # compute the mean and std for the training data
    norm_dict_path = get_mean_std_path(dataset)
    if os.path.exists(os.path.join(dataset_path, norm_dict_path)):
        print(f"Norm dict found.")
        norm_dict = torch.load(os.path.join(dataset_path, norm_dict_path))
    else:
        norm_dict = {}
        verts_3d_list = torch.cat(motions['lmk_3d_normed'], dim=0).reshape(-1, 3)
        target_list = torch.cat(motions['target'], dim=0)
        norm_dict['mean'] = {
            "lmk_3d_normed": verts_3d_list.mean(dim=0).float(),
            "target": target_list.mean(dim=0).float()
        }
        norm_dict['std'] = {
            "lmk_3d_normed": verts_3d_list.std(dim=0).float(),
            "target": target_list.std(dim=0).float(),
        }
        
        with open(os.path.join(dataset_path, norm_dict_path), "wb") as f:
            torch.save(norm_dict, f)

    return  motions, norm_dict

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
    
