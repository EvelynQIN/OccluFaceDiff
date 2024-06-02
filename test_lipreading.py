# Borrowed from SPETRE https://github.com/filby89/spectre/blob/12835d595b8e2f85b9ac1d64474f8f7827071ec7/utils/lipread_utils.py#L75
from argparse import Namespace
from fairseq import checkpoint_utils, tasks, utils
import os
from fairseq.dataclass.configs import GenerationConfig
from jiwer import wer, cer
import cv2
import tempfile
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
separator = Separator(phone='-', word=' ')
backend = EspeakBackend('en-us', words_mismatch='ignore', with_stress=False)


import random
import numpy as np
from tqdm import tqdm
from enum import Enum
import os.path
from glob import glob
from pathlib import Path
import subprocess
from loguru import logger
from time import time
from utils.parser_util import test_args
from model.FLAME import FLAME_mediapipe, FLAMETex
from configs.config import get_cfg_defaults
from utils import dataset_setting
from utils.renderer import SRenderY
from utils.data_util import batch_orth_proj, face_vertices
from pathlib import Path
from collections import defaultdict
import argparse

import torch
import torchvision.transforms.functional as F_v
import torch.nn.functional as F
from pytorch3d.io import load_obj
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams
from pytorch3d.structures import Meshes
# pretrained

import imageio
from skimage.io import imread
from data_loaders.dataloader_MEAD_flint import load_test_data
import ffmpeg
import pickle
import h5py
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append("external/Visual_Speech_Recognition_for_Multiple_Languages")
from external.Visual_Speech_Recognition_for_Multiple_Languages.dataloader.transform import Compose, Normalize, CenterCrop, SpeedRate, Identity
from configparser import ConfigParser
from utils.MediaPipeLandmarkLists import *

def get_phoneme_to_viseme_map():
    pho2vi = {}
    # pho2vi_counts = {}
    all_vis = []

    p2v = "pretrained/av_hubert/phonemes2visemes.csv"

    with open(p2v) as file:
        lines = file.readlines()
        # for line in lines[2:29]+lines[30:50]:
        for line in lines:
            if line.split(",")[0] in pho2vi:
                if line.split(",")[4].strip() != pho2vi[line.split(",")[0]]:
                    print('error')
            pho2vi[line.split(",")[0]] = line.split(",")[4].strip()

            all_vis.append(line.split(",")[4].strip())
            # pho2vi_counts[line.split(",")[0]] = 0
    return pho2vi, all_vis

pho2vi, all_vis = get_phoneme_to_viseme_map()

def save2avi(filename, data=None, fps=25):
    """save2avi. - function taken from Visual Speech Recognition repository

    :param filename: str, the filename to save the video (.avi).
    :param data: numpy.ndarray, the data to be saved.
    :param fps: the chosen frames per second.
    """
    assert data is not None, "data is {}".format(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc("F", "F", "V", "1")
    writer = cv2.VideoWriter(filename, fourcc, fps, (data[0].shape[1], data[0].shape[0]), 0)
    for frame in data:
        writer.write(frame)
    writer.release()

def convert_text_to_visemes(text):
    phonemized = backend.phonemize([text], separator=separator)[0]

    text = ""
    for word in phonemized.split(" "):
        visemized = []
        for phoneme in word.split("-"):
            if phoneme == "":
                continue
            try:
                visemized.append(pho2vi[phoneme.strip()])
                if pho2vi[phoneme.strip()] not in all_vis:
                    all_vis.append(pho2vi[phoneme.strip()])
                # pho2vi_counts[phoneme.strip()] += 1
            except:
                print('Count not find', phoneme)
                continue
        text += " " + "".join(visemized)
    return text

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TestRAVDESSDataset(Dataset):
    def __init__(
        self,
        dataset_name, 
        dataset_path,
        rec_path,
        split_data,
        model_type='diffusion'
    ):
        self.split_data = split_data
        self.rec_path = rec_path
        self.dataset = dataset_name
        self.model_type = model_type

        # paths to processed folder
        self.processed_folder = os.path.join(dataset_path, dataset_name, 'processed')
        self.cropped_landmark_folder = os.path.join(self.processed_folder,'cropped_landmarks_mediapipe')
        self.emoca_rec_folder = os.path.join(self.processed_folder, 'EMOCA_reconstruction')
        self.video_id_to_sent = {
            '01': 'KIDS ARE TALKING BY THE DOOR',
            '02': 'DOGS ARE SITTING BY THE DOOR'
        }
        self.video_folder = os.path.join(self.processed_folder, 'cropped_videos')

    def __len__(self):
        return len(self.split_data)
    
    def _get_transcript(self, motion_id):
        vocal, emotion, intensity, sent, rep, sbj = motion_id.split('-')
        return self.video_id_to_sent[sent]
    
    def _get_video_input(self, motion_id):
        fname = '02-' + motion_id + '.npy'
        lmk_path = os.path.join(self.cropped_landmark_folder, fname)
        lmk_2d = np.load(lmk_path, allow_pickle=True)[()][:,EMBEDDING_INDICES]
    
        rec_dict = {}
        rec_dict['lmk_2d'] = torch.from_numpy(lmk_2d).float() # (n,105,2)

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
        rec_dict['image'] = torch.from_numpy(image_array[:,:,:,[2,1,0]]).permute(0,3,1,2).float() # (n, 3, 224, 224) in RGB
        return rec_dict

    def _get_emica_codes(self, motion_id):
        fname = motion_id + '.npy'
        rec_path = os.path.join(self.emoca_rec_folder, fname)
        rec_dict = np.load(rec_path, allow_pickle=True)[()]

        rec_dict['global_pose'] = rec_dict['pose'][:,:3]
        rec_dict['jaw'] = rec_dict['pose'][:,3:]
        rec_dict.pop('pose', None)
        for key in rec_dict:
            rec_dict[key] = torch.from_numpy(rec_dict[key]).float()

        return rec_dict
    
    def _get_diffusion_reconstruction(self, motion_path):
        sample_path = os.path.join(self.rec_path, f"{motion_path}.npy")
        if not os.path.exists(sample_path):
            return None
        diffusion_sample = np.load(sample_path, allow_pickle=True)[()]
        diffusion_codes = {}
        diffusion_codes['jaw'] = torch.from_numpy(diffusion_sample[:,100:]).float()
        diffusion_codes['exp'] = torch.from_numpy(diffusion_sample[:,:100]).float()
        return diffusion_codes
    
    def _get_emoca_reconstruction(self, motion_path):
        sample_path = os.path.join(self.rec_path, f"{motion_path}.npy")
        if not os.path.exists(sample_path):
            return None
        emoca_codes = np.load(sample_path, allow_pickle=True)[()]

        rec_dict = {}
        rec_dict['jaw'] = torch.from_numpy(emoca_codes['pose'][:,3:]).float()
        if self.model_type in ['deca', 'spectre']:
            rec_dict['exp'] = torch.from_numpy(emoca_codes['exp']).float()
        else:
            rec_dict['exp'] = torch.from_numpy(emoca_codes['exp_emoca']).float()
        return rec_dict
    
    def _get_verts_reconstruction(self, motion_path):
        sample_path = os.path.join(self.rec_path, f"{motion_path}.npy")
        if not os.path.exists(sample_path):
            return None
        rec_dict = {}
        verts_reconstruciton = np.load(sample_path, allow_pickle=True)[()]
        rec_dict['verts'] = torch.from_numpy(verts_reconstruciton).float()

        # fix a dummy camera
        # rec_dict['cam'] = torch.FloatTensor([[ 8.8514, -0.0149,  0.0190]]).repeat(verts_reconstruciton.shape[0], 1)

        return rec_dict
    
    def __getitem__(self, idx):
        motion_path = self.split_data[idx]
        transcript = self._get_transcript(motion_path)
        if self.model_type in ['faceformer', 'facediffuser', 'codetalker', 'voca']:
            gt_flame = self._get_emica_codes(motion_path)
            rec_dict = self._get_verts_reconstruction(motion_path)
            n_verts = rec_dict['verts'].shape[0]

            num_frames = gt_flame['shape'].shape[0]
            if num_frames < n_verts:
                rec_dict['verts'] = rec_dict['verts'][:num_frames]
            elif num_frames > n_verts:
                rec_dict['verts'] = torch.cat([rec_dict['verts'], rec_dict['verts'][-1:].repeat(num_frames-n_verts, 1, 1)], dim=0)
            
            assert gt_flame['shape'].shape[0] == rec_dict['verts'].shape[0]
            for key in ['cam', 'shape', 'tex', 'light', 'global_pose']:
                rec_dict[key] = gt_flame[key]
        elif self.model_type == 'emica':
            rect_dict = self._get_emica_codes(motion_path)
        
        elif self.model_type in ['diffusion', 'emote']:
            rec_dict = self._get_emica_codes(motion_path)
            new_dict = self._get_diffusion_reconstruction(motion_path)
            if new_dict is None:
                return None, None
            rec_dict['exp'] = new_dict['exp']
            rec_dict['jaw'] = new_dict['jaw']
            seqlen = min(rec_dict['exp'].shape[0], rec_dict['global_pose'].shape[0])
            for key in rec_dict:
                rec_dict[key] = rec_dict[key][:seqlen]
        elif self.model_type in ['emoca', 'deca', 'spectre']:
            rec_dict = self._get_emica_codes(motion_path)
            new_dict = self._get_emoca_reconstruction(motion_path)
            if new_dict is None:
                return None, None
            rec_dict['exp'] = new_dict['exp']
            rec_dict['jaw'] = new_dict['jaw']
        elif self.model_type == 'video':
            rec_dict = self._get_video_input(motion_path)
        else:
            raise ValueError(f"{self.model_type} not supported!")
        if rec_dict is None:
            return None, None
    
        return rec_dict, transcript

class TestMeadDataset(Dataset):
    def __init__(
        self,
        dataset_name, # list of dataset names
        dataset_path,
        rec_path,
        split_data,
        n_shape=100,
        n_exp=50,
        model_type='diffusion'
    ):
        self.split_data = split_data
        self.rec_path = rec_path
        self.dataset = dataset_name
        self.n_shape = n_shape 
        self.n_exp = n_exp
        self.model_type = model_type

        # paths to processed folder
        self.processed_folder = os.path.join(dataset_path, dataset_name, 'processed')
        self.reconstruction_folder = os.path.join(self.processed_folder, 'reconstructions/EMICA-MEAD_flame2020')
        transcript_path = 'dataset/mead_25fps/processed/video_id_to_sent.npy'
        self.video_id_to_sent = np.load(transcript_path, allow_pickle=True)[()]

    def __len__(self):
        return len(self.split_data)

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
        
        with h5py.File(os.path.join(self.reconstruction_folder, motion_path, 'appearance.hdf5'), "r") as f:
            # light : (1, n, 27)
            # tex : (1, n, 50)
                for k in f.keys():
                    code_dict[k] = torch.from_numpy(f[k][0]).float()

        code_dict['shape'] = code_dict['shape'][:,:self.n_shape]
        code_dict['exp'] = code_dict['exp'][:,:self.n_exp]
        code_dict['light'] = code_dict['light'].reshape(-1, 9, 3)

        return code_dict 

    def _get_diffusion_reconstruction(self, motion_path):
        sample_path = os.path.join(self.rec_path, f"{motion_path}.npy")
        if not os.path.exists(sample_path):
            return None
        diffusion_sample = np.load(sample_path, allow_pickle=True)[()]
        diffusion_codes = {}
        diffusion_codes['jaw'] = torch.from_numpy(diffusion_sample[:,self.n_exp:]).float()
        diffusion_codes['exp'] = torch.from_numpy(diffusion_sample[:,:self.n_exp]).float()
        return diffusion_codes

    def _get_emoca_reconstruction(self, motion_path):
        sample_path = os.path.join(self.rec_path, f"{motion_path}.npy")
        if not os.path.exists(sample_path):
            return None
        emoca_codes = np.load(sample_path, allow_pickle=True)[()]

        rec_dict = {}
        rec_dict['jaw'] = torch.from_numpy(emoca_codes['pose'][:,3:]).float()
        if self.model_type in ['deca', 'spectre']:
            rec_dict['exp'] = torch.from_numpy(emoca_codes['exp']).float()
        else:
            rec_dict['exp'] = torch.from_numpy(emoca_codes['exp_emoca']).float()
        return rec_dict
    
    def _get_verts_reconstruction(self, motion_path):
        sample_path = os.path.join(self.rec_path, f"{motion_path}.npy")
        if not os.path.exists(sample_path):
            return None
        rec_dict = {}
        verts_reconstruciton = np.load(sample_path, allow_pickle=True)[()]
        rec_dict['verts'] = torch.from_numpy(verts_reconstruciton).float()

        # fix a dummy camera
        # rec_dict['cam'] = torch.FloatTensor([[ 8.8514, -0.0149,  0.0190]]).repeat(verts_reconstruciton.shape[0], 1)

        return rec_dict

    def _get_transcript(self, motion_path):
        return self.video_id_to_sent[motion_path]

    def __getitem__(self, idx):
        motion_path, _ = self.split_data[idx]
        transcript = self._get_transcript(motion_path)
        if self.model_type in ['faceformer', 'facediffuser', 'codetalker', 'voca']:
            gt_flame = self._get_emica_codes(motion_path)
            rec_dict = self._get_verts_reconstruction(motion_path)
            n_verts = rec_dict['verts'].shape[0]

            num_frames = gt_flame['shape'].shape[0]
            if num_frames < n_verts:
                rec_dict['verts'] = rec_dict['verts'][:num_frames]
            elif num_frames > n_verts:
                rec_dict['verts'] = torch.cat([rec_dict['verts'], rec_dict['verts'][-1:].repeat(num_frames-n_verts, 1, 1)], dim=0)
            
            assert gt_flame['shape'].shape[0] == rec_dict['verts'].shape[0]
            for key in ['cam', 'shape', 'tex', 'light', 'global_pose']:
                rec_dict[key] = gt_flame[key]
        elif self.model_type == 'emica':
            rect_dict = self._get_emica_codes(motion_path)
        elif self.model_type == 'diffusion':
            rec_dict = self._get_emica_codes(motion_path)
            new_dict = self._get_diffusion_reconstruction(motion_path)
            if new_dict is None:
                return None, None
            rec_dict['exp'] = new_dict['exp']
            rec_dict['jaw'] = new_dict['jaw']
        elif self.model_type in ['emoca', 'deca', 'spectre']:
            rec_dict = self._get_emica_codes(motion_path)
            new_dict = self._get_emoca_reconstruction(motion_path)
            if new_dict is None:
                return None, None
            rec_dict['exp'] = new_dict['exp']
            rec_dict['jaw'] = new_dict['jaw']
        else:
            raise ValueError(f"{self.model_type} not supported!")
        if rec_dict is None:
            return None, None
    
        return rec_dict, transcript

class LipReadEval:
    
    def __init__(self, config, model_cfg, test_data, device='cuda'):
        
        self.config = config
        self.model_cfg = model_cfg
        self.device = device
        self.test_data = test_data
        self.original_data_folder = 'dataset/mead_25fps/original_data'
        self.fps = 25
        self.sld_wind_size = 128
        self.model_type = config.model_type
        # IO setups
                        
        # name of the tested motion sequence
        self.output_folder = self.config.output_folder
        self.video_path = 'test_video_mouth.avi'
        
        logger.add(os.path.join(self.output_folder, f'test_mead_lipread_hubert_{config.model_type}.log'))
        logger.info(f"Using device {self.device}.")
        
        self._create_flame()
        self._setup_renderer()
        self.load_hubert()

        self.total_wer = AverageMeter()
        self.total_cer = AverageMeter()
        self.total_werv = AverageMeter()
        self.total_cerv = AverageMeter()

        # cut mouth utils
        self._crop_width = 96
        self._crop_height = 96
        self._window_margin = 12
        self._lip_idx = torch.from_numpy(LIP_EM).long()
        # self._start_idx = 48
        # self._stop_idx = 68

        self.fourcc = cv2.VideoWriter_fourcc("F", "F", "V", "1")    # format to write mouth sequence

        # # default render setting for verts only methods
        # with h5py.File('dataset/mead_25fps/processed/reconstructions/EMICA-MEAD_flame2020/M003/front/neutral/level_1/002/shape_pose_cam.hdf5', "r") as f:
        # # cam : (1, n, 3)
        # # exp : (1, n, 100)
        # # global_pose : (1, n, 3)
        # # jaw : (1, n, 3)
        # # shape : (1, n, 300)
        #     self.cam = torch.from_numpy(f['cam'][0,0]).float()[None,:]
        #     self.pose = torch.from_numpy(f['global_pose'][0,0]).float()[None,:]
        # with h5py.File('dataset/mead_25fps/processed/reconstructions/EMICA-MEAD_flame2020/M003/front/neutral/level_1/002/appearance.hdf5', "r") as f:
        #     # light : (1, n, 27)
        #     # tex : (1, n, 50)
        #     self.light = torch.from_numpy(f['light'][0,0]).float().reshape(1, 9, 3)
        #     self.tex = torch.from_numpy(f['tex'][0,0]).float()[None,:]

    def cut_mouth_vectorized(
        self,
        images, 
        landmarks, 
        convert_grayscale=True
        ):
                
        with torch.no_grad():

            landmarks = landmarks * 112 + 112
            # #1) smooth the landmarks with temporal convolution
            # landmarks are of shape (T, 68, 2) 
            landmarks_t = landmarks.reshape(*landmarks.shape[:2], -1)   # （bs, t, 468x2）
            # make temporal dimension last 
            landmarks_t = landmarks_t.permute(0, 2, 1)  # (bs, 468x2, t)

            # smooth with temporal convolution
            temporal_filter = torch.ones(self._window_margin, device=landmarks_t.device) / self._window_margin
            # pad the the landmarks 
            landmarks_t_padded = F.pad(landmarks_t, (self._window_margin // 2, self._window_margin // 2), mode='replicate')
            # convolve each channel separately with the temporal filter
            num_channels = landmarks_t.shape[1]
            smooth_landmarks_t = F.conv1d(landmarks_t_padded, 
                temporal_filter.unsqueeze(0).unsqueeze(0).expand(num_channels,1,temporal_filter.numel()), 
                groups=num_channels, padding='valid'
            )
            smooth_landmarks_t = smooth_landmarks_t[..., 0:landmarks_t.shape[-1]]

            # reshape back to the original shape 
            smooth_landmarks_t = smooth_landmarks_t.permute(0, 2, 1).view(landmarks.shape)
            smooth_landmarks_t = smooth_landmarks_t + landmarks.mean(dim=2, keepdims=True) - smooth_landmarks_t.mean(dim=2, keepdims=True)

            # #2) get the mouth landmarks
            mouth_landmarks_t = smooth_landmarks_t[..., self._lip_idx, :]
            
            # #3) get the mean of the mouth landmarks
            mouth_landmarks_mean_t = mouth_landmarks_t.mean(dim=-2, keepdims=True)
        
            # #4) get the center of the mouth
            center_x_t = mouth_landmarks_mean_t[..., 0]
            center_y_t = mouth_landmarks_mean_t[..., 1]

            # #5) use grid_sample to crop the mouth in every image 
            # create the grid
            height = self._crop_height//2
            width = self._crop_width//2

            # torch.arange(0, mouth_crop_width, device=images.device)

            grid = torch.stack(torch.meshgrid(torch.linspace(-height, height, self._crop_height).to(self.device) / 112,
                                            torch.linspace(-width, width, self._crop_width).to(self.device) / 112 ), 
                               dim=-1)
            grid = grid[..., [1, 0]]
            grid = grid.unsqueeze(0).unsqueeze(0).repeat(*images.shape[:2], 1, 1, 1)    # (bs, n, 1, 9, 2)

            # normalize the center to [-1, 1]
            center_x_t = (center_x_t - 112) / 112
            center_y_t = (center_y_t - 112) / 112

            center_xy =  torch.cat([center_x_t, center_y_t ], dim=-1).unsqueeze(-2).unsqueeze(-2)
            if center_xy.ndim != grid.ndim:
                center_xy = center_xy.unsqueeze(-2)
            assert grid.ndim == center_xy.ndim, f"grid and center_xy have different number of dimensions: {grid.ndim} and {center_xy.ndim}"
            grid = grid + center_xy
        B, T = images.shape[:2]
        images = images.view(B*T, *images.shape[2:])
        grid = grid.view(B*T, *grid.shape[2:])

        if convert_grayscale: 
            images = F_v.rgb_to_grayscale(images)
        
        image_crops = F.grid_sample(
            images, 
            grid,  
            align_corners=True, 
            padding_mode='zeros',
            mode='bicubic'
        )
        
        # image_crops = image_crops.view(B, T, *image_crops.shape[1:])

        if convert_grayscale:
            image_crops = image_crops.squeeze(1)
        return image_crops  # (bs*t,3, 96, 96)

    def load_hubert(self):
        ckpt_path = "pretrained/av_hubert/self_large_vox_433h.pt" # download this from https://facebookresearch.github.io/av_hubert/

        utils.import_user_module(Namespace(user_dir='external/av_hubert/avhubert'))

        modalities = ["video"]
        self.gen_subset = "test"
        self.gen_cfg = GenerationConfig(beam=1)
        models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.models = [model.eval().cuda() for model in models]
        self.saved_cfg.task.modalities = modalities

    def _create_flame(self):
        self.flame = FLAME_mediapipe(self.model_cfg).to(self.device)
        self.flametex = FLAMETex(self.model_cfg).to(self.device)    
    
    def _setup_renderer(self):
        self.render = SRenderY(
            self.model_cfg.image_size, 
            obj_filename=self.model_cfg.topology_path, 
            uv_size=self.model_cfg.uv_size).to(self.device)
        # face mask for rendering details
        mask = imread(self.model_cfg.face_eye_mask_path).astype(np.float32)/255. 
        mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)
        mask = imread(self.model_cfg.face_mask_path).astype(np.float32)/255.
        mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_mask = F.interpolate(mask, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)
        mean_texture = imread(self.model_cfg.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)

    def vis_motion_split(self, motion_split):
        
        if 'image' in motion_split:
            render_images = motion_split['image']
            lmk_2d = motion_split['lmk_2d']
        else:
            # prepare vis data dict 
            cam = motion_split['cam']
            if 'verts' in motion_split:
                # shape template
                n = motion_split['verts'].shape[0]
                null_exp = torch.zeros((n, 100)).to(self.device)
                null_jaw = torch.zeros((n, 3)).to(self.device)
                pose = torch.cat([motion_split['global_pose'], null_jaw], dim=-1)
                shape_template, _ = self.flame(motion_split['shape'], null_exp, pose)
                verts = motion_split['verts'] + shape_template
                
                lmk_3d = self.flame.select_lmk3d_mediapipe(verts)
                
                # orthogonal projection
                trans_verts = batch_orth_proj(verts, cam)
                trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

                lmk_2d = batch_orth_proj(lmk_3d, cam)
                lmk_2d[:, :, 1:] = -lmk_2d[:, :, 1:]
                
            else:
                shape = motion_split['shape']
                exp = motion_split['exp']
                pose = torch.cat([motion_split['global_pose'], motion_split['jaw']], dim=-1)
                # flame decoder
                verts, lmk_3d = self.flame(shape, exp, pose)
            
                # orthogonal projection
                trans_verts = batch_orth_proj(verts, cam)
                trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

                lmk_2d = batch_orth_proj(lmk_3d, cam)
                lmk_2d[:, :, 1:] = -lmk_2d[:, :, 1:]

            # # render
            albedo = self.flametex(motion_split['tex']).detach()
            light = motion_split['light']
            
            render_images = self.render(verts, trans_verts, albedo, light)['images']

        # cut the mouth region
        t = render_images.shape[0]
        render_images = render_images.view(1, t, *render_images.shape[1:])
        lmk_2d = lmk_2d.view(1, t, *lmk_2d.shape[1:])
        mouths_sequence = self.cut_mouth_vectorized(render_images, lmk_2d[...,:2], convert_grayscale=False) # (bs*t, 3, 96, 96)

        # tensor to video
        mouths_sequence = mouths_sequence.detach().cpu().numpy() * 255.
        mouths_sequence = np.maximum(np.minimum(mouths_sequence, 255), 0).transpose(0,2,3,1) # (bs*t, 96, 96, 3)
        mouths_sequence = mouths_sequence.astype(np.uint8).copy()[:,:,:,[2,1,0]] # RGB  

        # write to video
        for frame in mouths_sequence:
            self.writer.write(frame)

    
    def video_reconstruction(self, batch):
        
        self.writer = cv2.VideoWriter(self.video_path, self.fourcc, self.fps, (self._crop_width, self._crop_height))
        keys = list(batch.keys())
        frame_num = batch[keys[0]].shape[0]
        for start_id in range(0, frame_num, self.sld_wind_size):
            motion_split = {}
            for key in batch:
                motion_split[key] = batch[key][start_id:start_id+self.sld_wind_size].to(self.device)
            self.vis_motion_split(motion_split)
        self.writer.release()

        num_video_frames = int(cv2.VideoCapture(self.video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        assert num_video_frames == frame_num

    def run_lipreading(self, transcription):
        """
        :param transcriptions: transcript of the the current video
        :return:
        """
        num_frames = int(cv2.VideoCapture(self.video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        data_dir = tempfile.mkdtemp()
        tsv_cont = ["/\n", f"test-0\t{self.video_path}\t{None}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]
        label_cont = ["DUMMY\n"]
        with open(f"{data_dir}/test.tsv", "w") as fo:
            fo.write("".join(tsv_cont))
        with open(f"{data_dir}/test.wrd", "w") as fo:
            fo.write("".join(label_cont))
        self.saved_cfg.task.data = data_dir
        self.saved_cfg.task.label_dir = data_dir
        task = tasks.setup_task(self.saved_cfg.task)
        task.load_dataset(self.gen_subset, task_cfg=self.saved_cfg.task)
        generator = task.build_generator(self.models, self.gen_cfg)

        def decode_fn(x):
            dictionary = task.target_dictionary
            symbols_ignore = generator.symbols_to_strip_from_output
            symbols_ignore.add(dictionary.pad())
            return task.datasets[self.gen_subset].label_processors[0].decode(x, symbols_ignore)

        itr = task.get_batch_iterator(dataset=task.dataset(self.gen_subset)).next_epoch_itr(shuffle=False)
        sample = next(itr)
        sample = utils.move_to_cuda(sample)
        hypos = task.inference_step(generator, self.models, sample)
        hypo = hypos[0][0]['tokens'].int().cpu()
        hypo = decode_fn(hypo).upper()
        groundtruth = transcription.upper()


        w = wer(groundtruth, hypo)
        c = cer(groundtruth, hypo)


        # ---------- convert to visemes -------- #
        vg = convert_text_to_visemes(groundtruth)
        v = convert_text_to_visemes(hypo)
        print(hypo)
        print(groundtruth)
        print(v)
        print(vg)
        # -------------------------------------- #
        wv = wer(vg, v)
        cv = cer(vg, v)

        if w > 2.:
            return
        
        self.total_wer.update(w)
        self.total_cer.update(c)
        self.total_werv.update(wv)
        self.total_cerv.update(cv)

        logger.info(
            f"progress: {self.idx + 1}/\tcur WER: {self.total_wer.val * 100:.1f}\t"
            f"cur CER: {self.total_cer.val * 100:.1f}\t"
            f"count: {self.total_cer.count}\t"
            f"avg WER: {self.total_wer.avg * 100:.1f}\tavg CER: {self.total_cer.avg * 100:.1f}\t"
            f"avg WERV: {self.total_werv.avg * 100:.1f}\tavg CERV: {self.total_cerv.avg * 100:.1f}"
        )
    
    def run_eval(self):
        num_test_motions = len(self.test_data)
        self.idx = 0
        for i in tqdm(range(num_test_motions)):
            batch, transcript = self.test_data[i]
            if transcript is None:
                continue
            self.video_reconstruction(batch)
            self.run_lipreading(transcript)
            torch.cuda.empty_cache()
            self.idx += 1

def main():
    # sample use:
    # python3 test_lipreading.py --output_folder vis_result/EMOCA/non_occ --model_type emoca --rec_folder EMOCA_reconstruction
    # python3 test_lipreading.py --output_folder vis_result/diffusion_Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/all --model_type diffusion --rec_folder diffusion_sample
    # python3 test_lipreading.py --output_folder vis_result/EMOCA/non_occ --model_type emica --rec_folder dummy
    # python3 test_lipreading.py --output_folder vis_result/SPECTRE/non_occ --model_type spectre --rec_folder SPECTRE_reconstruction
    # python3 test_lipreading.py --output_folder vis_result/FaceFormer --model_type faceformer --rec_folder reconstruction
    # python3 test_lipreading.py --output_folder vis_result/FaceDiffuser --model_type facediffuser --rec_folder reconstruction
    # python3 test_lipreading.py --output_folder vis_result/CodeTalker --model_type codetalker --rec_folder reconstruction
    # python3 test_lipreading.py --output_folder vis_result/VOCA --model_type voca --rec_folder reconstruction
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, help='folder to store diffusion sample.', required=True)
    parser.add_argument('--split', type=str, help='mead split for evaluation.', default='test')
    parser.add_argument('--dataset', type=str, default='mead_25fps', help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='./dataset',help='dataset name')
    parser.add_argument('--rec_folder', type=str, default='', required=True, help='folder to store reconstruction results.')    # when model_type=emica, should set a dummy folder
    parser.add_argument('--model_type', type=str, default='diffusion', required=True, help='should be in [diffusion, deca, emoca, emica, spectre, facediffuser, codetalker, emote]')

    args = parser.parse_args()
    pretrained_args = get_cfg_defaults()

    rec_folder = os.path.join(args.output_folder, args.rec_folder)
    if args.model_type in ['diffusion', 'emica', 'faceformer', 'facediffuser', 'codetalker', 'emote', 'video']:
        model_cfg = pretrained_args.model 
    elif args.model_type in ['deca', 'emoca', 'spectre']:
        model_cfg = pretrained_args.emoca
    else:
        raise ValueError(f"{args.model_type} not supported!")
    
    print("loading test data...")
    
    if args.dataset == 'mead_25fps':
        test_video_list = load_test_data(
            args.dataset, 
            args.dataset_path, 
            args.split)

        print(f"number of test sequences: {len(test_video_list)}")
        
        test_dataset = TestMeadDataset(
            args.dataset,
            args.dataset_path,
            rec_folder,
            test_video_list,
            n_shape=model_cfg.n_shape,
            n_exp=model_cfg.n_exp,
            model_type=args.model_type
        )
    elif args.dataset == 'RAVDESS':
        from data_loaders.dataloader_RAVDESS import load_RAVDESS_test_data
        test_video_list = load_RAVDESS_test_data(
            args.dataset, 
            args.dataset_path)
        
        print(f"number of test sequences: {len(test_video_list)}")
        test_dataset = TestRAVDESSDataset(
            args.dataset,
            args.dataset_path,
            rec_folder,
            test_video_list,
            args.model_type
        )


    lip_reader = LipReadEval(args, model_cfg, test_dataset, 'cuda')
    
    lip_reader.run_eval()

if __name__ == "__main__":
    main()

