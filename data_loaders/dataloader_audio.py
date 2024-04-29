import glob
import os

import torch
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from skimage.transform import estimate_transform, warp
from collections import defaultdict
import random
from utils import dataset_setting
from utils import utils_transform

class TrainDataset(Dataset):
    def __init__(
        self,
        split_data,
        input_motion_length=60,
        train_dataset_repeat_times=30,
        fps=30,
        
    ):
        self.split_data = split_data
        self.input_motion_length = input_motion_length
        self.train_dataset_repeat_times = train_dataset_repeat_times 
        self.fps = fps 
        self.audio_per_frame = 16000 // self.fps
        
       
    def __len__(self):
        return len(self.split_data['shape']) * self.train_dataset_repeat_times

    def __getitem__(self, idx):
        id = idx % len(self.split_data['shape'])
        seqlen = self.split_data['shape'][id].shape[0]
        
        if seqlen == self.input_motion_length: 
            start_id = 0
        else:
            start_id = torch.randint(0, int(seqlen - self.input_motion_length), (1,))[0]     # random crop a motion seq

        target = self.split_data['target'][id][start_id:start_id+self.input_motion_length]
        shape = self.split_data['shape'][id][start_id:start_id+self.input_motion_length]
        
        # cut audio into corresponding position
        start_audio = start_id * self.audio_per_frame 
        end_audio = (start_id+self.input_motion_length) * self.audio_per_frame
        audio_input = self.split_data['shape'][id][start_audio:end_audio]

        return {
            'audio_input': audio_input.float(),
            'shape': shape.float(),
            'target': target.float()
        }

def load_data_all(datasets, split, input_motion_length):
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
    split_data = defaultdict(list)
    for dataset in datasets:
        split_path = os.path.join('./processed', dataset, split+'.npy')
        if os.path.exists(split_path):
            processed_paths = np.load(split_path, allow_pickle=True)[()]['processed_paths']
        else:
            raise ValueError(f"{dataset} processed path not existed!")
        print(f"Loading [{dataset}] ...")
        for path in tqdm(processed_paths):
            if dataset == 'multiface' and 'SEN' not in path:
                continue
            processed_data = torch.load(path)
            if dataset == 'multiface':
                if processed_data['shape'].shape[0] < input_motion_length:
                    continue
                split_data['audio_input'].append(processed_data['audio_input'])
                split_data['shape'].append(processed_data['shape'])
                pose = processed_data['pose']
                exp = processed_data['exp']
                jaw_6d = utils_transform.aa2sixd(pose[...,3:])
                target = torch.cat([jaw_6d, exp], dim=-1)
                split_data['target'].append(target)
            else:
                if (processed_data['shape'].shape[0] + 1) // 2 < input_motion_length:
                    continue
                split_data['audio_input'].append(processed_data['audio_input'])
                split_data['shape'].append(processed_data['shape'][::2])
                split_data['target'].append(processed_data['target'][::2])
    
    return split_data

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