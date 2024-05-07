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
from utils.occlusion import MediaPipeFaceOccluder
import cv2 
from torchvision import transforms
import torchvision.transforms.functional as F 
from utils import utils_transform
import pickle
import h5py

MEDIAPIPE_LANDMARK_NUMBER = 478

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
        self.occluder = MediaPipeFaceOccluder()

        # image process
        self.image_size = 224 
        
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization
        self.input_motion_length = input_motion_length

        # paths to processed folder
        self.processed_folder = os.path.join(dataset_path, dataset_name, 'processed')
        self.audio_input_folder = os.path.join(self.processed_folder, 'audio_inputs')
        self.cropped_folder = os.path.join(self.processed_folder,'images')
        self.reconstruction_folder = os.path.join(self.processed_folder, 'reconstructions/EMICA-MEAD_flame2020')
        self.emotion_folder = os.path.join(self.processed_folder, 'emotions/resnet50')
       
    def __len__(self):
        return len(self.split_data) * self.train_dataset_repeat_times
    
    def _get_lmk_mediapipe(self, motion_path, start_id):
        """ get mediapipe landmarks normlized to [-1, 1]
        """
        lmk_path = os.path.join(self.cropped_folder, motion_path, 'cropped_frames.hdf5')
        with h5py.File(lmk_path, "r") as f:
            lmk_2d = torch.from_numpy(f['lmk_2d'][start_id:start_id+self.input_motion_length]).float()

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

        return torch.from_numpy(lmk_2d).float() # (n,478,2)

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

        # compose target 
        jaw_6d = utils_transform.aa2sixd(code_dict['jaw'])
        code_dict['target'] = torch.cat([jaw_6d, code_dict['exp']], dim=-1)
        code_dict.pop('exp', None)

        return code_dict 
    
    def _get_emotion_features(self, motion_path, start_id):
        with h5py.File(os.path.join(self.emotion_folder, motion_path, 'features.pkl'), 'r') as f:
            feature = f['data']['feature'][0, start_id:start_id+self.input_motion_length]
        return torch.from_numpy(feature).float() # (n, 2048)
    
    def _get_lmk_occlusion_mask(self, lmk_2d):
        n, v = lmk_2d.shape[:2]
        lmk_mask = torch.ones(n, v)
        sid = torch.randint(low=0, high=n-10, size=(1,))[0]
        occ_num_frames = torch.randint(low=10, high=n-sid+1, size=(1,))[0]
        frame_id = torch.arange(sid, sid+occ_num_frames)
        for occ_region, occ_prob in self.occluder.occlusion_regions_prob.items():
            prob = np.random.rand()
            if prob < occ_prob:
                lmk_mask = self.occluder.occlude_lmk_batch(lmk_2d, lmk_mask, occ_region, frame_id)
        
        # occlude random frames
        prob = np.random.rand()
        if prob < 0.2:
            frame_id = torch.randint(low=0, high=n, size=(n // 2,))
            lmk_mask[frame_id,:] = 0

        return lmk_mask


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
        # batch['emo_feature'] = self._get_emotion_features(motion_path, start_id)
        batch['lmk_mask'] = self._get_lmk_occlusion_mask(batch['lmk_2d'])
        # for key in batch:
        #     print(f"shape of {key}: {batch[key].shape}")
        
        return batch

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
        'M003', 'M005']
    }      
    
    MEAD_sentence_split = {
        "train": ["%03d" % i for i in range(1, 26)] +  ["%03d" % i for i in range(36, 100)],
        "test": ["%03d" % i for i in range(26, 36)]
    }    

    return MEAD_subject_split[split], MEAD_sentence_split[split]

def load_data(dataset, dataset_path, split, input_motion_length):
    """
    Load motin list for specified split, ensuring all motions >= input_motion_length
    Return:
        motion_list: list of motion path (sbj/view/emotion/level/sent)
    """
    processed_folder = os.path.join(dataset_path, dataset, 'processed')
    motion_split_path = os.path.join(processed_folder, f'split/motion_list_{split}.npy')
    
    if os.path.exists(motion_split_path):
        motion_list = np.load(motion_split_path, allow_pickle=True)[()]
        
    else:
        folder_path = os.path.join(processed_folder, 'split')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        MEAD_subject_split, MEAD_sentence_split = get_split_MEAD(split)
        with open(os.path.join(processed_folder, 'video_list_woimg.pkl'), 'rb') as f:
            video_list = pickle.load(f)
        motion_list = []
        for video_id, num_frames in video_list.items():
            # check split and motion length
            sbj, view, emotion, level, sent = video_id.split('/')
            if sbj not in MEAD_subject_split or \
                sent not in MEAD_sentence_split or \
                num_frames < input_motion_length:
                continue 

            motion_list.append((video_id, num_frames))

        motion_list = np.array(motion_list)

        np.save(motion_split_path, motion_list)

    return  motion_list

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