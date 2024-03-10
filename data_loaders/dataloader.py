# Copyri[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_motion_length]ion_length][start_id:start_id + input_motion_length][start_id:start_id + input_motion_length][start_id:start_id + input_motion_targetght (c) Meta Platforms, Inc. All Rights Reserved
import glob
import os

import torch
import numpy as np 
from utils.image_process import batch_crop_lmks
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict
import random

class TrainDataset(Dataset):
    def __init__(
        self,
        dataset,
        norm_dict,
        motion_path_list,
        input_motion_length=120,
        train_dataset_repeat_times=1,
        no_normalization=True,
    ):
        self.dataset = dataset
        self.mean = norm_dict['mean']
        self.std = norm_dict['std']
        self.motion_path_list = motion_path_list
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization
        self.input_motion_length = input_motion_length

    def __len__(self):
        return len(self.motion_path_list) * self.train_dataset_repeat_times

    def inv_transform(self, target):
        
        target = target * self.std["target"] + self.mean["target"]
        
        return target

    def __getitem__(self, idx):
        id = idx % len(self.motion_path_list)

        motion_dict = torch.load(self.motion_path_list[id])
        
        seqlen = motion_dict['target'].shape[0]
        
        if self.train_dataset_repeat_times == 1:
            # do not repeat
            input_motion_length = seqlen 
        elif self.input_motion_length is None:  
            # in transformer, randomly clip a subseq
            input_motion_length = torch.randint(min(50, seqlen), seqlen+1, (1,))[0]
        else:
            # fix motion len
            input_motion_length = self.input_motion_length 

        if seqlen == input_motion_length: 
            start_id = 0
        else:
            start_id = torch.randint(0, int(seqlen - input_motion_length), (1,))[0]     # random crop a motion seq
        
        lmk_2d = motion_dict['lmk_2d'][start_id:start_id + input_motion_length].reshape(input_motion_length, -1)  # (n, 68x2)
        lmk_3d_normed = motion_dict['lmk_3d_normed'][start_id:start_id + input_motion_length].reshape(input_motion_length, -1) # (n, 68x3)
        lmk_3d_cam = motion_dict['lmk_3d_cam'][start_id:start_id + input_motion_length].reshape(input_motion_length, -1) # (n, 68x3)
        target = motion_dict['target'][start_id:start_id + input_motion_length] # (n, rot30 + shape300 + exp100)
        
        
        n_imgs = torch.sum(motion_dict['img_mask'][start_id:start_id + input_motion_length])
        img_start_fid = torch.sum(motion_dict['img_mask'][:start_id])
        img_arr = motion_dict['arcface_input'][img_start_fid:img_start_fid+n_imgs] # (n_imgs, 3, 112, 112)

        # make sure there are always 4 images within the clipped sequence
        needed_imgs = 4 - n_imgs
        if needed_imgs < 0:
            # randomly select 4 images wo replacement
            img_ids = torch.LongTensor(random.sample(range(n_imgs), 4))
            img_arr = img_arr[img_ids]
        elif needed_imgs > 0:
            # repeat needed images
            n_img_available = torch.sum(motion_dict['img_mask'])
            assert n_img_available > 0
            img_arr_added_ids = torch.randint(0, n_img_available, size=(needed_imgs,))
            img_arr_repeated = motion_dict['arcface_input'][img_arr_added_ids]
            img_arr = torch.cat([img_arr, img_arr_repeated], dim=0) if needed_imgs < 4 else img_arr_repeated
        assert (not img_arr.isnan().any()) and img_arr.shape[0] == 4
            
        # Normalization 
        if not self.no_normalization:    
            lmk_3d_normed = ((lmk_3d_normed.reshape(-1, 3) - self.mean['lmk_3d_normed']) / (self.std['lmk_3d_normed'] + 1e-8)).reshape(input_motion_length, -1)
            target = (target - self.mean['target']) / (self.std['target'] + 1e-8)
        
        assert (not target.isnan().any()) 
        assert (not lmk_2d.isnan().any()) 
        assert (not lmk_3d_normed.isnan().any()) 
        assert (not lmk_3d_cam.isnan().any())
        
        return target.float(), lmk_2d.float(), lmk_3d_normed.float(), img_arr.float(), lmk_3d_cam.float()
    
class TestDataset(Dataset):
    def __init__(
        self,
        dataset,
        norm_dict,
        motion_path_list,
        no_normalization=True,
    ):
        self.dataset = dataset
        self.mean = norm_dict['mean']
        self.std = norm_dict['std']
        self.motion_path_list = motion_path_list
        self.no_normalization = no_normalization

    def __len__(self):
        return len(self.motion_path_list)

    def inv_transform(self, target):
        
        target = target * self.std["target"] + self.mean["target"]
        
        return target

    def __getitem__(self, idx):
        
        id = idx % len(self.motion_path_list)

        motion_dict = torch.load(self.motion_path_list[id])
        
        seqlen = motion_dict['target'].shape[0]
        
        lmk_2d = motion_dict['lmk_2d'].reshape(seqlen, -1)  # (n, 68x2)
        lmk_3d_normed = motion_dict['lmk_3d_normed'].reshape(seqlen, -1) # (n, 68x3)
        target = motion_dict['target'] # (n, shape300 + exp100 + rot30 + trans3)
        motion_id = os.path.split(self.motion_path_list[id])[-1].split(os.sep)[0]
        
        
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
            lmk_3d_normed = ((lmk_3d_normed.reshape(-1, 3) - self.mean['lmk_3d_normed']) / (self.std['lmk_3d_normed'] + 1e-8)).reshape(seqlen, -1)
            target = (target - self.mean['target']) / (self.std['target'] + 1e-8)
        
        assert (not target.isnan().any()) 
        assert (not lmk_2d.isnan().any()) 
        assert (not lmk_3d_normed.isnan().any()) 
        
        return target.float(), lmk_2d.float(), lmk_3d_normed.float(), img_arr.float(), motion_id


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

def get_face_motion(motion_paths, skip_frame, split, args):
    motion_list = defaultdict(list)
    # discard the motion sequence shorter than the specified length for train / val
    input_motion_length = args.input_motion_length
    discard_shorter_seq = True if (input_motion_length is not None) and (split != "test") else False

    # image crop information
    # scale = args.scale
    # trans_scale = args.trans_scale
    # image_size = args.image_size
    for motion_path in tqdm(motion_paths):
        motion = torch.load(motion_path)
        nframes = motion['target'].shape[0]
        if discard_shorter_seq and nframes < input_motion_length*skip_frame:
            continue
        motion_list['motion_id'].append(os.path.split(motion_path)[-1].split(".")[0])

        # crop and normalize 2d lmks
        # lmks_2d_cropped = batch_crop_lmks(lmks_2d, trans_scale, scale, image_size)

        # motion_list['lmk_2d'].append(motion['lmk_2d'][0::skip_frame])
        motion_list['lmk_3d_normed'].append(motion['lmk_3d_normed'][0::skip_frame])
        # motion_list['lmk_3d_cam'].append(motion['lmk_3d_cam'][0::skip_frame])
        # motion_list['img_arr'].append(motion['arcface_input'][0::skip_frame])
        # motion_list['img_mask'].append(motion['img_mask'][0::skip_frame].bool())

        # ensure shape remains same for all frames
        motion_list['target'].append(motion['target'][0::skip_frame])
        
    return motion_list

def get_valid_motion_path_list(motion_paths, skip_frame, input_motion_length):
    motion_path_list = []
    # discard the motion sequence shorter than the specified length for train / val

    for motion_path in tqdm(motion_paths):
        motion = torch.load(motion_path)
        nframes = motion['target'].shape[0]
        if nframes < input_motion_length*skip_frame:
            continue
        motion_path_list.append(motion_path)

    return motion_path_list

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
    # TODO: train all datasets at fps=60
    if dataset == "FaMoS":
        # downsample FaMoS to half
        skip_frame = 1
    else:
        skip_frame = 1
    
    input_motion_length = args.input_motion_length if "input_motion_length" in args.keys() else None
    motion_paths_fname = os.path.join(dataset_path, dataset, 'valid_motion_paths', f'{split}.npy')
    
    if os.path.exists(motion_paths_fname):
        motion_paths = np.load(motion_paths_fname, allow_pickle=True)
    else:
        if not os.path.exists(os.path.join(dataset_path, dataset, 'valid_motion_paths')):
            os.makedirs(os.path.join(dataset_path, dataset, 'valid_motion_paths'))
        discard_shorter_seq = True if (input_motion_length is not None) and (split != "test") else False
        motion_paths = get_path(dataset_path, dataset, split, subject_id, selected_motion_ids)
        if discard_shorter_seq:
            motion_paths = get_valid_motion_path_list(motion_paths, skip_frame, input_motion_length) 
        np.save(motion_paths_fname, motion_paths, allow_pickle=True)

    # compute the mean and std for the training data
    norm_dict_path = get_mean_std_path(dataset)
    if os.path.exists(os.path.join(dataset_path, norm_dict_path)):
        norm_dict = torch.load(os.path.join(dataset_path, norm_dict_path))
    else:
        norm_dict = {}
        lmk_3d_normed = []
        target = []
        for path in motion_paths:
            motion = torch.load(path)
            lmk_3d_normed.append(motion['lmk_3d_normed'])
            target.append(motion['target'])
        verts_3d_list = torch.cat(lmk_3d_normed, dim=0).reshape(-1, 3)
        target_list = torch.cat(target, dim=0)
        norm_dict['mean'] = {
            "lmk_3d_normed": verts_3d_list.mean(dim=0).float(),
            "target": target_list.mean(dim=0).float(),
            "trans": torch.FloatTensor([0.004, 0.222, 1.200])
        }
        norm_dict['std'] = {
            "lmk_3d_normed": verts_3d_list.std(dim=0).float(),
            "target": target_list.std(dim=0).float(),
        }
        
        with open(os.path.join(dataset_path, norm_dict_path), "wb") as f:
            torch.save(norm_dict, f)

    return  motion_paths, norm_dict


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
    
    
    
    