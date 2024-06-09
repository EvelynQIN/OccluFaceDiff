import glob
import os

import torch
import torch.nn as nn
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from skimage.transform import estimate_transform, warp
from collections import defaultdict
import random
from utils.occlusion import MediaPipeFaceOccluder, get_test_img_occlusion_mask
import cv2 
from torchvision import transforms
import torchvision.transforms.functional as F 
from utils import utils_transform
import pickle
import h5py
from utils.MediaPipeLandmarkLists import *

MEDIAPIPE_LANDMARK_NUMBER = 478

class TestRAVDESSDataset(Dataset):
    def __init__(
        self,
        dataset_name, # list of dataset names
        dataset_path,
        split_data,
        fps=25,
        occ_type='non_occ',
        use_iris=False,
        load_audio_input=True,
        vis=False,
        mask_path=None
    ):
        self.split_data = split_data
        self.fps = fps
        self.dataset = dataset_name
        self.use_iris = use_iris # whether to use iris landmarks from mediapipe (last 10)
        self.load_audio_input = load_audio_input
        self.occ_type = occ_type # occlusion type
        self.vis = vis

        # image process
        self.image_size = 224 
        self.wav_per_frame = int(16000 / self.fps)

        # paths to processed folder
        self.processed_folder = os.path.join(dataset_path, dataset_name, 'processed')
        self.audio_input_folder = os.path.join(self.processed_folder, 'audio_inputs')
        self.cropped_landmark_folder = os.path.join(self.processed_folder,'cropped_landmarks_mediapipe')
        self.video_folder = os.path.join(self.processed_folder, 'cropped_videos')
        self.emoca_rec_folder = os.path.join(self.processed_folder, 'EMOCA_reconstruction')
        self.mask_path = mask_path
       
    def __len__(self):
        return len(self.split_data)
    
    def _get_lmk_mediapipe(self, motion_id):
        """ get mediapipe landmarks normlized to [-1, 1]
        """
        fname = '02-' + motion_id + '.npy'
        lmk_path = os.path.join(self.cropped_landmark_folder, fname)
        lmk_2d = np.load(lmk_path, allow_pickle=True)[()]

        if not self.use_iris:
            lmk_2d = lmk_2d[:,:468] # exclude pupil parts

        return torch.from_numpy(lmk_2d).float() # (n,478,2)

    def _get_audio_input(self, motion_id, num_frames):
        fname = '03-' + motion_id + '.pt'
        audio_path = os.path.join(self.audio_input_folder, fname)
        audio_input = torch.load(audio_path)[0]

        remain_audio_len = num_frames * self.wav_per_frame - len(audio_input)
        if remain_audio_len > 0:
            audio_input = nn.functional.pad(audio_input, (0, remain_audio_len))
        elif remain_audio_len < 0:
            audio_input = audio_input[:num_frames*self.wav_per_frame]
        
        audio_input = audio_input.reshape(num_frames, -1)

        return audio_input  # (n, wav_per_frame)

    def _get_gt_input(self, motion_id):
        fname = motion_id + '.npy'
        rec_path = os.path.join(self.emoca_rec_folder, fname)
        rec_dict = np.load(rec_path, allow_pickle=True)[()]
        return rec_dict

    def _get_image_info(self, motion_id):
        fname = '02-' + motion_id + '.mp4'
        video_path = os.path.join(self.video_folder, fname)
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
    
    def _get_occlusion_mask(self, img_mask, lmk_2d):
        kpts = (lmk_2d.clone() * 112 + 112).long()
        img_mask = get_test_img_occlusion_mask(img_mask, kpts, self.occ_type)
        lmk_mask = self._get_lmk_mask_from_img_mask(img_mask, kpts)
        return img_mask, lmk_mask
    
    def _get_lmk_mask_from_img_mask(self, img_mask, kpts):
        n, v = kpts.shape[:2]
        lmk_mask = torch.ones((n,v))
        for i in range(n):
            for j in range(v):
                x, y = kpts[i,j]
                if x<0 or x >=self.image_size or y<0 or y>=self.image_size or img_mask[i,y,x]==0:
                    lmk_mask[i,j] = 0
        return lmk_mask

    def __getitem__(self, idx):
        motion_id = self.split_data[idx]
        batch = {}
        batch['lmk_2d'] = self._get_lmk_mediapipe(motion_id)
        seqlen = batch['lmk_2d'].shape[0]
        if self.load_audio_input:
            batch['audio_input'] = self._get_audio_input(motion_id, seqlen)
        if self.mask_path is not None and os.path.exists(os.path.join(self.mask_path, f"{motion_id}_mask.npy")):
            mask_path_motion = os.path.join(self.mask_path, f"{motion_id}_mask.npy")
            img_mask = np.load(mask_path_motion, allow_pickle=True)[()]
            batch['img_mask'] = torch.from_numpy(img_mask).float()
        else:
            batch['img_mask'] = torch.ones((seqlen, self.image_size, self.image_size))
        if self.vis:
            batch['image'] = self._get_image_info(motion_id)
            
        batch['img_mask'], batch['lmk_mask'] = self._get_occlusion_mask(batch['img_mask'], batch['lmk_2d'])
        
        gt_rec = self._get_gt_input(motion_id)
        batch['shape'] = torch.from_numpy(gt_rec['shape']).float()
        batch['cam'] = torch.from_numpy(gt_rec['cam']).float()
        batch['global_pose'] = torch.from_numpy(gt_rec['pose'][:,:3]).float()
        batch['exp'] = torch.from_numpy(gt_rec['exp_emoca']).float()
        batch['jaw'] = torch.from_numpy(gt_rec['pose'][:,3:]).float()
        
        return batch, motion_id

def load_RAVDESS_test_data(
        dataset, 
        dataset_path, 
        subject_list=None, 
        emotion_list=None,
        level_list=None, 
        sent_list=None):
    """
    Load motin list for specified split, ensuring all motions >= input_motion_length
    Return:
        motion_list: list of motion path (sbj/view/emotion/level/sent)
    """
    processed_folder = os.path.join(dataset_path, dataset, 'processed')
    
    motion_list = []
    for video in os.scandir(os.path.join(processed_folder, 'cropped_videos')):
        # check split and motion length
        video_id = video.name[:-4]
        modality, vocal, emotion, intensity, sent, rep, sbj = video_id.split('-')
        if (subject_list is not None and sbj not in subject_list) or \
            (emotion_list is not None and emotion not in emotion_list) or \
            (level_list is not None and intensity not in level_list) or \
            (sent_list is not None and int(sent) not in sent_list):
            continue

        motion_list.append('-'.join([vocal, emotion, intensity, sent, rep, sbj]))

    motion_list = np.array(motion_list)

    return  motion_list