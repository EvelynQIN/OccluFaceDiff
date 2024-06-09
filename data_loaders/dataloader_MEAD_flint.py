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

# pad the sequence with init / endding poses
class TrainMeadDatasetPad(Dataset):
    def __init__(
        self,
        dataset_name, # list of dataset names
        dataset_path,
        split_data,
        input_motion_length=120,
        train_dataset_repeat_times=1,
        no_normalization=True,
        fps=25,
        n_shape=300,
        n_exp=50,
        load_tex=False,
        use_iris=False
    ):
        self.split_data = split_data
        self.fps = fps
        self.dataset = dataset_name
        self.load_tex = load_tex # whether to use texture from emica
        self.use_iris = use_iris # whether to use iris landmarks from mediapipe (last 10)
        self.n_shape = n_shape 
        self.n_exp = n_exp
        # for audio alignment
        sampling_rate = 16000
        self.wav_per_frame = int(sampling_rate / self.fps)

        # apply occlusion
        # self.occluder = MediaPipeFaceOccluder()

        # image process
        self.image_size = 224 
        
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization
        self.input_motion_length = input_motion_length

        # paths to processed folder
        self.processed_folder = os.path.join(dataset_path, dataset_name, 'processed')
        self.audio_input_folder = os.path.join(self.processed_folder, 'audio_inputs')
        self.cropped_landmark_folder = os.path.join(self.processed_folder,'cropped_landmarks')
        self.reconstruction_folder = os.path.join(self.processed_folder, 'reconstructions/EMICA-MEAD_flame2020')
        self.emotion_folder = os.path.join(self.processed_folder, 'emotions/resnet50')
        self.image_folder = os.path.join(self.processed_folder,'images')
       
    def __len__(self):
        return len(self.split_data) * self.train_dataset_repeat_times
    
    def _get_lmk_mediapipe(self, motion_path, start_id, motion_len):
        """ get mediapipe landmarks normlized to [-1, 1]
        """
        lmk_path = os.path.join(self.cropped_landmark_folder, motion_path, 'landmarks_mediapipe.hdf5')
        with h5py.File(lmk_path, "r") as f:
            lmk_2d = torch.from_numpy(f['lmk_2d'][start_id:start_id+motion_len]).float()

        if not self.use_iris:
            lmk_2d = lmk_2d[:,:468] # exclude pupil parts

        return lmk_2d # (n,478,2)

    def _get_images(self, motion_path, start_id, motion_len):
        """ get mediapipe landmarks normlized to [-1, 1]
        """
        img_path = os.path.join(self.image_folder, motion_path, 'cropped_frames.hdf5')
        with h5py.File(img_path, "r") as f:
            image = torch.from_numpy(f['images'][start_id:start_id+motion_len]).float()
            # img_mask = torch.from_numpy(f['img_masks'][start_id:start_id+self.input_motion_length]).float()

        return image

    def _get_audio_input(self, motion_path, start_id, motion_len):
        path_sep = motion_path.split('/')
        sbj = path_sep[0]
        emotion, level, sent = path_sep[-3:]
        audio_path = os.path.join(self.audio_input_folder, f'{sbj}/{emotion}/{level}/{sent}.pt')
        audio_input = torch.load(audio_path)

        sid = start_id * self.wav_per_frame
        audio_len = motion_len * self.wav_per_frame 
        audio_split = audio_input[0, sid:sid + audio_len]

        # pad zero to the end if not long enough
        remain_audio_len = audio_len - len(audio_split)
        if remain_audio_len > 0:
            audio_split = nn.functional.pad(audio_split, (0, remain_audio_len))
            
        return audio_split # (n*wavperframe,)

    def _get_emica_codes(self, motion_path, start_id, motion_len):
        code_dict = {}

        with h5py.File(os.path.join(self.reconstruction_folder, motion_path, 'shape_pose_cam.hdf5'), "r") as f:
        # cam : (1, n, 3)
        # exp : (1, n, 100)
        # global_pose : (1, n, 3)
        # jaw : (1, n, 3)
        # shape : (1, n, 300)
            for k in f.keys():
                code_dict[k] = torch.from_numpy(f[k][0,start_id:start_id+motion_len]).float()
        if self.load_tex:
            with h5py.File(os.path.join(self.reconstruction_folder, motion_path, 'appearance.hdf5'), "r") as f:
            # light : (1, n, 27)
            # tex : (1, n, 50)
                for k in f.keys():
                    code_dict[k] = torch.from_numpy(f[k][0,start_id:start_id+motion_len]).float()
        code_dict['shape'] = code_dict['shape'][:,:self.n_shape]
        code_dict['exp'] = code_dict['exp'][:,:self.n_exp]

        return code_dict 

    def __getitem__(self, idx):
        id = idx % len(self.split_data)
        motion_path, seqlen = self.split_data[id]
        seqlen = int(seqlen)

        pad_prob = np.random.rand()
        if pad_prob < 0.6:
            motion_len = torch.randint(low=self.input_motion_length-25, high=60, size=(1,))[0]
            if np.random.rand() < 0.5:
                start_id = 0
            else:
                start_id = seqlen-motion_len
        else:
            motion_len = self.input_motion_length
            start_id = torch.randint(0, int(seqlen - motion_len), (1,))[0] if seqlen > motion_len else 0 # random crop a motion seq
        
        pad_size = self.input_motion_length - motion_len       

        batch = self._get_emica_codes(motion_path, start_id, motion_len)
        batch['lmk_2d'] = self._get_lmk_mediapipe(motion_path, start_id, motion_len)
        batch['audio_input'] = self._get_audio_input(motion_path, start_id, motion_len)

        # pading applied
        if pad_size > 0:
            batch['audio_input'] = batch['audio_input'].reshape(motion_len, -1)
            if start_id == 0:
                for key in batch:
                    tmp = torch.repeat_interleave(batch[key][:1], pad_size, dim=0)
                    batch[key] = torch.cat([tmp, batch[key]], dim=0)
            else:
                for key in batch:
                    tmp = torch.repeat_interleave(batch[key][-1:], pad_size, dim=0)
                    batch[key] = torch.cat([batch[key], tmp], dim=0)
            batch['audio_input'] = batch['audio_input'].reshape(-1)

        
        return batch

class TrainMeadDataset(Dataset):
    def __init__(
        self,
        dataset_name, # list of dataset names
        dataset_path,
        split_data,
        input_motion_length=120,
        train_dataset_repeat_times=1,
        no_normalization=True,
        fps=25,
        n_shape=300,
        n_exp=50,
        load_tex=False,
        use_iris=False
    ):
        self.split_data = split_data
        self.fps = fps
        self.dataset = dataset_name
        self.load_tex = load_tex # whether to use texture from emica
        self.use_iris = use_iris # whether to use iris landmarks from mediapipe (last 10)
        self.n_shape = n_shape 
        self.n_exp = n_exp
        # for audio alignment
        sampling_rate = 16000
        self.wav_per_frame = int(sampling_rate / self.fps)

        # apply occlusion
        # self.occluder = MediaPipeFaceOccluder()

        # image process
        self.image_size = 224 
        
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization
        self.input_motion_length = input_motion_length

        # paths to processed folder
        self.processed_folder = os.path.join(dataset_path, dataset_name, 'processed')
        self.audio_input_folder = os.path.join(self.processed_folder, 'audio_inputs')
        self.cropped_landmark_folder = os.path.join(self.processed_folder,'cropped_landmarks')
        self.reconstruction_folder = os.path.join(self.processed_folder, 'reconstructions/EMICA-MEAD_flame2020')
        self.emotion_folder = os.path.join(self.processed_folder, 'emotions/resnet50')
        self.image_folder = os.path.join(self.processed_folder,'images')
       
    def __len__(self):
        return len(self.split_data) * self.train_dataset_repeat_times
    
    def _get_lmk_mediapipe(self, motion_path, start_id):
        """ get mediapipe landmarks normlized to [-1, 1]
        """
        lmk_path = os.path.join(self.cropped_landmark_folder, motion_path, 'landmarks_mediapipe.hdf5')
        with h5py.File(lmk_path, "r") as f:
            lmk_2d = torch.from_numpy(f['lmk_2d'][start_id:start_id+self.input_motion_length]).float()

        if not self.use_iris:
            lmk_2d = lmk_2d[:,:468] # exclude pupil parts

        return lmk_2d # (n,478,2)

    def _get_images(self, motion_path, start_id):
        """ get mediapipe landmarks normlized to [-1, 1]
        """
        img_path = os.path.join(self.image_folder, motion_path, 'cropped_frames.hdf5')
        with h5py.File(img_path, "r") as f:
            image = torch.from_numpy(f['images'][start_id:start_id+self.input_motion_length]).float()
            # img_mask = torch.from_numpy(f['img_masks'][start_id:start_id+self.input_motion_length]).float()

        return image

    def _get_audio_input(self, motion_path, start_id):
        path_sep = motion_path.split('/')
        sbj = path_sep[0]
        emotion, level, sent = path_sep[-3:]
        audio_path = os.path.join(self.audio_input_folder, f'{sbj}/{emotion}/{level}/{sent}.pt')
        audio_input = torch.load(audio_path)

        sid = start_id * self.wav_per_frame
        audio_len = self.input_motion_length * self.wav_per_frame 
        audio_split = audio_input[0, sid:sid + audio_len]

        # pad zero to the end if not long enough
        remain_audio_len = audio_len - len(audio_split)
        if remain_audio_len > 0:
            audio_split = nn.functional.pad(audio_split, (0, remain_audio_len))
            
        return audio_split # (n*wavperframe,)

    def _get_emica_codes(self, motion_path, start_id):
        code_dict = {}

        with h5py.File(os.path.join(self.reconstruction_folder, motion_path, 'shape_pose_cam.hdf5'), "r") as f:
        # cam : (1, n, 3)
        # exp : (1, n, 100)
        # global_pose : (1, n, 3)
        # jaw : (1, n, 3)
        # shape : (1, n, 300)
            for k in f.keys():
                code_dict[k] = torch.from_numpy(f[k][0,start_id:start_id+self.input_motion_length]).float()
        if self.load_tex:
            with h5py.File(os.path.join(self.reconstruction_folder, motion_path, 'appearance.hdf5'), "r") as f:
            # light : (1, n, 27)
            # tex : (1, n, 50)
                for k in f.keys():
                    code_dict[k] = torch.from_numpy(f[k][0,start_id:start_id+self.input_motion_length]).float()
        code_dict['shape'] = code_dict['shape'][:,:self.n_shape]
        code_dict['exp'] = code_dict['exp'][:,:self.n_exp]

        return code_dict 
    
    def _get_emotion_features(self, motion_path, start_id):
        with h5py.File(os.path.join(self.emotion_folder, motion_path, 'features.pkl'), 'r') as f:
            feature = f['data']['feature'][0, start_id:start_id+self.input_motion_length]
        return torch.from_numpy(feature).float() # (n, 2048)

    def __getitem__(self, idx):
        id = idx % len(self.split_data)
        motion_path, seqlen = self.split_data[id]
        seqlen = int(seqlen)
        
        if seqlen == self.input_motion_length: 
            start_id = 0
        else:
            start_id = torch.randint(0, int(seqlen - self.input_motion_length), (1,))[0]     # random crop a motion seq

        batch = self._get_emica_codes(motion_path, start_id)
        batch['lmk_2d'] = self._get_lmk_mediapipe(motion_path, start_id)
        batch['audio_input'] = self._get_audio_input(motion_path, start_id)
        # batch['image'] = self._get_images(motion_path, start_id)
        # batch['lmk_mask'] = self.occluder.get_lmk_occlusion_mask(batch['lmk_2d'])
        # for key in batch:
        #     print(f"shape of {key}: {batch[key].shape}")
        
        return batch

class TestMeadDataset(Dataset):
    def __init__(
        self,
        dataset_name, # list of dataset names
        dataset_path,
        split_data,
        fps=25,
        n_shape=300,
        n_exp=50,
        occ_type='non_occ',
        load_tex=False,
        use_iris=False,
        load_audio_input=True,
        vis=False,
        use_segmask=False,
        mask_path=None
    ):
        self.split_data = split_data
        self.fps = fps
        self.dataset = dataset_name
        self.load_tex = load_tex # whether to use texture from emica
        self.use_iris = use_iris # whether to use iris landmarks from mediapipe (last 10)
        self.load_audio_input = load_audio_input
        self.n_shape = n_shape 
        self.n_exp = n_exp
        self.occ_type = occ_type # occlusion type
        self.vis = vis
        self.use_segmask = use_segmask
        # # apply occlusion
        # self.occluder = MediaPipeFaceOccluder()

        # image process
        self.image_size = 224 
        self.wav_per_frame = int(16000 / self.fps)

        # paths to processed folder
        self.processed_folder = os.path.join(dataset_path, dataset_name, 'processed')
        self.audio_input_folder = os.path.join(self.processed_folder, 'audio_inputs')
        self.cropped_landmark_folder = os.path.join(self.processed_folder,'cropped_landmarks')
        self.reconstruction_folder = os.path.join(self.processed_folder, 'reconstructions/EMICA-MEAD_flame2020')
        self.emotion_folder = os.path.join(self.processed_folder, 'emotions/resnet50')
        self.image_folder = os.path.join(self.processed_folder, 'images')
        self.mask_path = mask_path
       
    def __len__(self):
        return len(self.split_data)
    
    def _get_lmk_mediapipe(self, motion_path):
        """ get mediapipe landmarks normlized to [-1, 1]
        """
        lmk_path = os.path.join(self.cropped_landmark_folder, motion_path, 'landmarks_mediapipe.hdf5')
        with h5py.File(lmk_path, "r") as f:
            lmk_2d = torch.from_numpy(f['lmk_2d'][:]).float()

        # # validity check
        # landmark_validity = np.ones((len(lmk_2d), 1), dtype=np.float32)
        # for i in range(len(lmk_2d)): 
        #     if len(lmk_2d[i]) == 0: # dropped detection
        #         lmk_2d[i] = np.zeros((MEDIAPIPE_LANDMARK_NUMBER, 2))
        #         landmark_validity[i] = 0.
        #     else: # multiple faces detected or one face detected
        #         lmk_2d[i] = lmk_2d[i][0] # just take the first one for now
        # lmk_2d = np.stack(lmk_2d, axis=0)

        if not self.use_iris:
            lmk_2d = lmk_2d[:,:468] # exclude pupil parts

        return lmk_2d # (n,478,2)

    def _get_audio_input(self, motion_path, num_frames):
        path_sep = motion_path.split('/')
        sbj = path_sep[0]
        emotion, level, sent = path_sep[-3:]
        audio_path = os.path.join(self.audio_input_folder, f'{sbj}/{emotion}/{level}/{sent}.pt')
        audio_input = torch.load(audio_path)[0]

        remain_audio_len = num_frames * self.wav_per_frame - len(audio_input)
        if remain_audio_len > 0:
            audio_input = nn.functional.pad(audio_input, (0, remain_audio_len))
        
        audio_input = audio_input.reshape(num_frames, -1)

        return audio_input  # (n, wav_per_frame)

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
        if self.load_tex:
            with h5py.File(os.path.join(self.reconstruction_folder, motion_path, 'appearance.hdf5'), "r") as f:
            # light : (1, n, 27)
            # tex : (1, n, 50)
                for k in f.keys():
                    code_dict[k] = torch.from_numpy(f[k][0]).float()
        code_dict['shape'] = code_dict['shape'][:,:self.n_shape]
        code_dict['exp'] = code_dict['exp'][:,:self.n_exp]

        return code_dict 
    
    def _get_emotion_features(self, motion_path):
        with h5py.File(os.path.join(self.emotion_folder, motion_path, 'features.pkl'), 'r') as f:
            feature = f['data']['feature'][0]
        return torch.from_numpy(feature).float() # (n, 2048)

    def _get_image_info(self, motion_path):
        data_dict = {}
        with h5py.File(os.path.join(self.image_folder, motion_path, 'cropped_frames.hdf5'), "r") as f:
            data_dict['image'] = torch.from_numpy(f['images'][:]).float()
            if self.use_segmask and 'img_mask' in f:
                data_dict['img_mask'] = torch.from_numpy(f['img_masks'][:]).float()
            else:
                data_dict['img_mask'] = torch.ones((data_dict['image'].shape[0], self.image_size, self.image_size))
        return data_dict
    
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
        motion_path, seqlen = self.split_data[idx]
        seqlen = int(seqlen)

        batch = self._get_emica_codes(motion_path)
        batch['lmk_2d'] = self._get_lmk_mediapipe(motion_path)
        if self.load_audio_input:
            batch['audio_input'] = self._get_audio_input(motion_path, seqlen)
        if self.vis:
            batch.update(self._get_image_info(motion_path))
            if self.mask_path is not None and os.path.exists(os.path.join(self.mask_path, f"{motion_path}_mask.npy")):
                mask_path_motion = os.path.join(self.mask_path, f"{motion_path}_mask.npy")
                img_mask = np.load(mask_path_motion, allow_pickle=True)[()]
                batch['img_mask'] = torch.from_numpy(img_mask).float()
            else:
                batch['img_mask'], batch['lmk_mask'] = self._get_occlusion_mask(batch['img_mask'], batch['lmk_2d'])
        else:
            batch['img_mask'] = torch.ones((seqlen, self.image_size, self.image_size))
            batch['img_mask'], batch['lmk_mask'] = self._get_occlusion_mask(batch['img_mask'], batch['lmk_2d'])
        # for key in batch:
        #     print(f"shape of {key}: {batch[key].shape}")
        
        return batch, motion_path

def get_split_MEAD(split):

    MEAD_subject_split = {
    # 38 subjects
    "train": ['M011', 'M012', 'M013', 'M019', 'M022', 
              'M023', 'M024', 'M025', 'M026', 'M027', 
              'M028', 'M029', 'M030', 'M031', 'M033', 
              'M034', 'M037', 'M039', 'M040', 'M041', 
              'W009', 'W014', 'W015', 'W016', 'W018', 
              'W019', 'W023', 'W024', 'W025', 'W026', 
              'W028', 'W029', 'W033', 'W035', 'W036', 
              'W037', 'W038', 'W040'],
    "test": [
        'M003', 'M005', 'M007', 'M009', 'W011']
    }      
    
    MEAD_sentence_split = {
        "train": [i for i in range(10, 200)],
        "test": [i for i in range(0, 10)]
    }    

    return MEAD_subject_split[split], MEAD_sentence_split[split]

def load_test_data(
        dataset, 
        dataset_path, 
        split, 
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
    video_id_to_sent_id = np.load(os.path.join(dataset_path, dataset, 'processed/video_id_to_sent_id.npy'), allow_pickle=True)[()]
        
    folder_path = os.path.join(processed_folder, 'split')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    subjects, MEAD_sentence_split = get_split_MEAD(split)
    if subject_list is not None:
        subjects = subject_list
    if level_list is None:
        level_list = ['level_1', 'level_2', 'level_3']
    if sent_list is None:
        sent_list = MEAD_sentence_split
    with open(os.path.join(processed_folder, 'video_list_test.pkl'), 'rb') as f:
        video_list = pickle.load(f)
    motion_list = []
    for video_id, num_frames in video_list:
        # check split and motion length
        sbj, view, emotion, level, sent = video_id.split('/')
        sent_id = video_id_to_sent_id[video_id]
        if sbj not in subjects or \
            sent_id not in sent_list or \
            level not in level_list:
            continue 
        if emotion_list and emotion not in emotion_list:
            continue

        motion_list.append((video_id, num_frames))

    motion_list = np.array(motion_list)

    return  motion_list


def load_data(dataset, dataset_path, split, input_motion_length):
    """
    Load motin list for specified split, ensuring all motions >= input_motion_length
    Return:
        motion_list: list of motion path (sbj/view/emotion/level/sent)
    """
    processed_folder = os.path.join(dataset_path, dataset, 'processed')
    video_id_to_sent_id = np.load(os.path.join(dataset_path, dataset, 'processed/video_id_to_sent_id.npy'), allow_pickle=True)[()]
        
    folder_path = os.path.join(processed_folder, 'split')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    MEAD_subject_split, MEAD_sentence_split = get_split_MEAD(split)
    with open(os.path.join(processed_folder, 'video_list_woimg.pkl'), 'rb') as f:
        video_list = pickle.load(f)
    motion_list = []
    for video_id, num_frames in video_list:
        # check split and motion length
        sbj, view, emotion, level, sent = video_id.split('/')
        sent_id = video_id_to_sent_id[video_id]
        if sbj not in MEAD_subject_split or \
            sent_id not in MEAD_sentence_split or \
            num_frames < input_motion_length:
            continue 

        motion_list.append((video_id, num_frames))

    motion_list = np.array(motion_list)


    return  motion_list

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