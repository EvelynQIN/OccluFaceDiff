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
        dataset_name,
        split_data
    ):
        self.split_data = split_data
        self.skip_frames = 1 if dataset_name == 'multiface' else 2
       
    def __len__(self):
        return len(self.split_data['processed_paths'])

    def __getitem__(self, idx):
        processed_data = torch.load(self.split_data['processed_paths'][idx])

        target = processed_data['target'][::self.skip_frames]
        shape = processed_data['shape'][::self.skip_frames]
        audio_input = processed_data['audio_input']

        return {
            'audio_input': audio_input.float(),
            'shape': shape.float(),
            'target': target.float()
        }

def load_data(dataset, dataset_path, split):
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
    split_path = os.path.join('./processed', dataset, split+'.npy')
    if os.path.exists(split_path):
        split_data = np.load(split_path, allow_pickle=True)[()]
    
    return  split_data

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