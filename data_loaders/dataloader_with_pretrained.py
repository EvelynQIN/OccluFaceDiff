# Copyri[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_motion_length]ion_length][start_id:start_id + input_motion_length][start_id:start_id + input_motion_length][start_id:start_id + input_motion_targetght (c) Meta Platforms, Inc. All Rights Reserved
import glob
import os
from memory_profiler import profile
import torch
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from collections import defaultdict
import random
from utils.landmark_mask import REGIONS
from utils import dataset_setting
from utils import utils_transform
from torchvision import transforms
import torchvision.transforms.functional as F 

def random_occlusion(occlusion_type, mask_array):
    occlusion_types = [
        'downsample_frame',
        'bottom_right',
        'bottom_left',
        'top_left',
        'top_right',
        'right_half',
        'left_half',
        'top_half' ,
        'bottom_half',
        'all_occ'
    ]
    if occlusion_type == 'random_occlusion':
        occlusion_id = torch.randint(low=0, high=9, size=(1,))[0]
        occlusion_type = occlusion_types[occlusion_id]
    
    num_frames = mask_array.shape[0]
    if occlusion_type == 'downsample_frame':
        mask_array[:num_frames:3] = 0
    elif occlusion_type == 'bottom_right':
        mask_array[:,90:,90:] = 0
    elif occlusion_type == 'bottom_left':
        mask_array[:,90:,:100] = 0
    elif occlusion_type == 'top_left':
        mask_array[:,:100,:100] = 0
    elif occlusion_type == 'top_right':
        mask_array[:,:100,112:] = 0
    elif occlusion_type == 'right_half':
        mask_array[:,:,112:] = 0
    elif occlusion_type == 'left_half':
        mask_array[:,:,:112] = 0
    elif occlusion_type == 'top_half':
        mask_array[:,:112,:] = 0
    elif occlusion_type == 'bottom_half':
        mask_array[:,112:,:] = 0
    elif occlusion_type == 'all_occ':
        mask_array[:,:,:] = 0
    return mask_array


def random_color_jitter_to_video(imgs, brightness, contrast, saturation, hue, order):
    #imgs of shape [N, 3, h, w]
    vid_transforms = []
    vid_transforms.append(lambda img: F.adjust_brightness(img, brightness))
    vid_transforms.append(lambda img: F.adjust_contrast(img, contrast))
    vid_transforms.append(lambda img: F.adjust_saturation(img, saturation))
    vid_transforms.append(lambda img: F.adjust_hue(img, hue))
    
    transform = transforms.Compose([vid_transforms[id] for id in order])

    return transform(imgs)

class TrainDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        processed_path, # path to .pt file
        input_motion_length=120,
        train_dataset_repeat_times=1,
        no_normalization=True,
        occlusion_mask_prob=0.5,
        mixed_occlusion_prob=0.3,
        fps=30
    ):
        self.dataset_name = dataset_name
        self.original_image_size = dataset_setting.image_size[self.dataset_name]
        self.processed_path = processed_path
        self.image_size = 224 
        self.scale = 1.5
        self.data_fps_original = 30 if self.dataset_name == 'multiface' else 60
        
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization
        self.input_motion_length = input_motion_length
        self.occlusion_mask_prob = occlusion_mask_prob
        self.fps = fps
        self.skip_frames = int(self.data_fps_original / self.fps) 
        self.input_motion_length = input_motion_length * self.skip_frames

    def __len__(self):
        return len(self.processed_path) * self.train_dataset_repeat_times

    def get_occlusion_mask(self, num_frames, mask_array, with_audio):
        # add random occlusion mask
        add_mask = torch.bernoulli(torch.ones(1) * self.occlusion_mask_prob)[0]
        if not add_mask:
            return mask_array

        num_frames = mask_array.shape[0]

        # select occlusion type
        occlusion_type = torch.randint(low=0, high=3, size=(1,))[0]

        if with_audio:
            if occlusion_type == 0:
                # occlude all visual cues
                mask_array[:,:,:] = 0
            elif occlusion_type == 1:
                # occlude the whole mouth region
                mask_array[:,100:,100:] = 0
            else:
                # occlude random regions for each frame
                mask_bbx = torch.randint(low=4, high=220, size=(num_frames,4)) 
                for i in range(num_frames):
                    mask_array[i, mask_bbx[i,0]:mask_bbx[i,1], mask_bbx[i,2]:mask_bbx[i,3]] = 0
        else:
            if occlusion_type == 0:
                # occlude fixed region: top left coords and w, h of occlusion rectangle
                x, y = torch.randint(low=10, high=200, size=(2,)) 
                dx, dy = torch.randint(low=20, high=112, size=(2,))    
                mask_array[:, y:y+dy, x:x+dx] = 0
            elif occlusion_type == 1:
                # occlude random regions for each frame
                mask_bbx = torch.randint(low=20, high=200, size=(num_frames,4)) 
                for i in range(num_frames):
                    mask_array[i, mask_bbx[i,0]:mask_bbx[i,1], mask_bbx[i,2]:mask_bbx[i,3]] = 0
            else:
                # occlude random num of frames
                occluded_frame_ids = torch.randint(low=0, high=num_frames, size=(num_frames // 2,))
                mask_array[occluded_frame_ids] = 0
        return mask_array

    def get_lmk_mask(self, lmk2d, img_mask):
        lmk_mask = []
        pix_pos = ((lmk2d + 1) * self.image_size / 2).long()
        pix_pos = torch.clamp(pix_pos, min=0, max=self.image_size-1)
        for i in range(lmk2d.shape[0]):
            lmk_mask.append(img_mask[i, pix_pos[i, :, 1], pix_pos[i, :, 0]])
        return torch.stack(lmk_mask)
    
    def image_augment(self, image):
         # image augmentation
        transf_order, b, c, s, h = transforms.ColorJitter.get_params(
            brightness=(1, 2.5),
            contrast=(1, 2),
            saturation=(1, 1),
            hue=(-0.1,0.1))
        
        return random_color_jitter_to_video(image, b, c, s, h, transf_order)

    def __getitem__(self, idx):
        id = idx % len(self.processed_path)
        processed_data = torch.load(self.processed_path[id])
        seqlen = processed_data['lmk_2d'].shape[0]
        with_audio = 'SEN' in self.processed_path[id]
        
        if self.train_dataset_repeat_times == 1:
            # do not repeat
            input_motion_length = 20 * self.skip_frames
            
        elif self.input_motion_length is None:  
            # in transformer, randomly clip a subseq
            input_motion_length = torch.randint(min(20, seqlen), min(seqlen+1, 41), (1,))[0]
        else:
            # fix motion len
            input_motion_length = self.input_motion_length 

        if seqlen == input_motion_length: 
            start_id = 0
        else:
            start_id = torch.randint(0, int(seqlen - input_motion_length), (1,))[0]     # random crop a motion seq
        
        batch = {}
        
        for k in ['image', 'img_mask', 'lmk_2d', 'audio_emb', 'shape', 'tex', 'cam', 'light']:
            batch[k] = processed_data[k][start_id:start_id + input_motion_length:self.skip_frames]
        
        batch['image'] = self.image_augment(batch['image'])
        batch['img_mask'] = self.get_occlusion_mask(batch['lmk_2d'].shape[0], batch['img_mask'], with_audio)
        batch['lmk_mask'] = self.get_lmk_mask(batch['lmk_2d'], batch['img_mask'])
        pose = processed_data['pose'][start_id:start_id + input_motion_length:self.skip_frames]
        exp = processed_data['exp'][start_id:start_id + input_motion_length:self.skip_frames]
        jaw_6d = utils_transform.aa2sixd(pose[...,3:])
        batch['target'] = torch.cat([jaw_6d, exp], dim=-1)
        batch['R'] = pose[...,:3]

        return batch

class TestDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        processed_path, # path to .pt file
        input_motion_length=120,
        no_normalization=True,
        occlusion_mask_prob=0.5,
        fps=30,
        occlusion_type=None
    ):
        self.occlusion_type = occlusion_type
        self.dataset_name = dataset_name
        self.original_image_size = dataset_setting.image_size[self.dataset_name]
        self.processed_path = processed_path
        self.image_size = 224 
        self.scale = 1.5
        self.data_fps_original = 30 if self.dataset_name == 'multiface' else 60
        
        self.no_normalization = no_normalization
        self.input_motion_length = input_motion_length
        self.occlusion_mask_prob = occlusion_mask_prob

    def __len__(self):
        return len(self.processed_path)


    def get_occlusion_mask(self, mask_array, occlusion_type='downsample_frame'):
        # add random occlusion mask
        add_mask = torch.bernoulli(torch.ones(1) * self.occlusion_mask_prob)[0]
        # mask_array = torch.ones(num_frames, self.image_size, self.image_size)
        if not add_mask:
            return mask_array
        return random_occlusion(occlusion_type, mask_array)

    def get_lmk_mask(self, lmk2d, img_mask):
        lmk_mask = []
        pix_pos = ((lmk2d + 1) * self.image_size / 2).long()
        pix_pos = torch.clamp(pix_pos, min=0, max=self.image_size-1)
        for i in range(lmk2d.shape[0]):
            lmk_mask.append(img_mask[i, pix_pos[i, :, 1], pix_pos[i, :, 0]])
        return torch.stack(lmk_mask)

    def __getitem__(self, idx):
        id = idx % len(self.processed_path)
        batch = torch.load(self.processed_path[id])

        subject, _, motion_id = self.processed_path[id].split('/')[-3:]
        batch['motion_id'] = motion_id[:-3]
        batch['subject_id'] = subject.split('--')[3]
        if 'SEN' in motion_id:
            batch['audio_path'] = '/'.join(self.processed_path[id].split('/')[:-2]) + f'/audio/{motion_id[:-3]}.wav'
        batch['img_mask'] = self.get_occlusion_mask(batch['img_mask'], self.occlusion_type)
        batch['lmk_mask'] = self.get_lmk_mask(batch['lmk_2d'], batch['img_mask'])
        jaw_6d = utils_transform.aa2sixd(batch['pose'][...,3:])
        batch['target'] = torch.cat([jaw_6d, batch['exp']], dim=-1)

        return batch

def get_path(dataset_path, dataset, split, input_motion_length):
    if dataset == 'multiface':
        split_id = dataset_setting.multiface_split[split]
        selected_motions = [dataset_setting.multiface_motion_id[i] for i in split_id]
    subjects = [subject.path for subject in os.scandir(os.path.join(dataset_path, dataset)) if subject.is_dir()]
    processed_path =[]
    for subject in subjects:
        for motion_path in glob.glob(os.path.join(subject, 'processed_data', '*.pt')):
            motion_name = os.path.split(motion_path)[1][:-3]
            if motion_name in selected_motions:
                if len(glob.glob(os.path.join(subject, 'images', f'{motion_name}/*/*.png'))) < input_motion_length:
                    continue
                processed_path.append(motion_path)
    
    return processed_path

def load_data(dataset, dataset_path, split, input_motion_length):
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
        processed_path = np.load(split_path, allow_pickle=True)[()]
    else:
        folder_path = os.path.join('./processed', dataset)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        processed_path = get_path(dataset_path, dataset, split, input_motion_length)
        np.save(split_path, processed_path)

    return np.array(processed_path)

def load_motion_for_subject(dataset, dataset_path, subject_id, split=None, motion_id=None):
    sbj_path = glob.glob(os.path.join(dataset_path, dataset, f'*{subject_id}*/processed_data'))[0]
    if dataset == 'multiface':
        split_id = dataset_setting.multiface_split[split]
        selected_motions = [dataset_setting.multiface_motion_id[i] for i in split_id]
    if motion_id is None:
        motion_path = []
        for path in glob.glob(sbj_path + '/*.pt'):
            motion_name = os.path.split(path)[1][:-3]
            if motion_name in selected_motions:
                motion_path.append(path)
    else:
        motion_path = glob.glob(sbj_path + f'/{motion_id}.pt')
    return motion_path

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

if __name__ == "__main__":
    
    load_data(
        'dataset', 
        'multiface',
        'train'
    )
