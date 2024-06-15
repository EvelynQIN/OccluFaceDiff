from argparse import Namespace
import os
import cv2

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

import sys
sys.path.append('./')

from utils import utils_transform, utils_visualize
from utils.parser_util import test_args
from model.FLAME import FLAME_mediapipe, FLAMETex
from configs.config import get_cfg_defaults
from utils import dataset_setting
from utils.renderer import SRenderY
from utils.data_util import batch_orth_proj, face_vertices
from pathlib import Path
from collections import defaultdict
import argparse
from pathlib import Path
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
from utils.MediaPipeLandmarkLists import *
from utils.occlusion import RandomOcclusion

class TestMeadDataset(Dataset):
    def __init__(
        self,
        dataset_name, # list of dataset names
        dataset_path,
        rec_paths,
        split_data,
        model_types=['diffusion']
    ):
        self.split_data = split_data
        self.rec_paths = rec_paths # (dict of motion_type: rec_path)
        self.dataset = dataset_name
        self.model_types = model_types

        # paths to processed folder
        self.processed_folder = os.path.join(dataset_path, dataset_name, 'processed')
        self.image_folder = os.path.join(self.processed_folder, 'images')
        self.reconstruction_folder = os.path.join(self.processed_folder, 'reconstructions/EMICA-MEAD_flame2020')
        self.cropped_landmark_folder = os.path.join(self.processed_folder, 'cropped_landmarks')
        transcript_path = 'dataset/mead_25fps/processed/video_id_to_sent.npy'
        self.video_id_to_sent = np.load(transcript_path, allow_pickle=True)[()]
        self.random_occluder = RandomOcclusion()

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

        code_dict['light'] = code_dict['light'].reshape(-1, 9, 3)

        lmk_path = os.path.join(self.cropped_landmark_folder, motion_path, 'landmarks_mediapipe.hdf5')
        with h5py.File(lmk_path, "r") as f:
            lmk_2d = torch.from_numpy(f['lmk_2d'][:]).float()

        code_dict['lmk_2d'] = lmk_2d[:,:468] # exclude pupil parts

        # with h5py.File(os.path.join(self.image_folder, motion_path, 'cropped_frames.hdf5'), "r") as f:
        #     code_dict['original_image'] = torch.from_numpy(f['images'][:]).float()

        return code_dict 

    def _get_diffusion_reconstruction(self, motion_path, rec_path):
        sample_path = os.path.join(rec_path, f"{motion_path}.npy")
        if not os.path.exists(sample_path):
            return None
        diffusion_sample = np.load(sample_path, allow_pickle=True)[()]
        diffusion_codes = {}
        diffusion_codes['jaw'] = torch.from_numpy(diffusion_sample[:,100:]).float()
        diffusion_codes['exp'] = torch.from_numpy(diffusion_sample[:,:100]).float()

        # load img_mask
        mask_path = os.path.join(rec_path, f"{motion_path}_mask.npy")
        print(mask_path)
        if os.path.exists(mask_path):
            print(f"found mask")
            mask_info = np.load(mask_path, allow_pickle=True)[()]
            diffusion_codes['mask_info'] = mask_info
        return diffusion_codes

    def _get_emoca_reconstruction(self, motion_path, model_type, rec_path):
        sample_path = os.path.join(rec_path, f"{motion_path}.npy")
        if not os.path.exists(sample_path):
            return None
        emoca_codes = np.load(sample_path, allow_pickle=True)[()]

        rec_dict = {}
        rec_dict['jaw'] = torch.from_numpy(emoca_codes['pose'][:,3:]).float()
        if model_type in ['deca', 'spectre']:
            rec_dict['exp'] = torch.from_numpy(emoca_codes['exp']).float()
        else:
            rec_dict['exp'] = torch.from_numpy(emoca_codes['exp_emoca']).float()
        return rec_dict

    def __getitem__(self, idx):
        motion_path, _ = self.split_data[idx]
        model_reconstructions = {}
        gt_flame = self._get_emica_codes(motion_path)
        model_reconstructions['gt'] = gt_flame
        for model_type in self.model_types:
            rec_path = self.rec_paths[model_type]
            if model_type in ['diffusion']:
                rec_dict = self._get_diffusion_reconstruction(motion_path, rec_path)
            elif model_type in ['emoca', 'deca', 'spectre']:
                rec_dict = self._get_emoca_reconstruction(motion_path, model_type, rec_path)
            else:
                raise ValueError(f"{model_type} not supported!")
            if rec_dict is None:
                return None, None
            model_reconstructions[model_type] = rec_dict
        mask_info = model_reconstructions['diffusion']['mask_info']
        # model_reconstructions['gt']['image'] = self.random_occluder.get_image_with_mask(model_reconstructions['gt']['original_image'], mask_info['mask_path'], mask_info['frame_ids'])
        model_reconstructions['gt']['lmk_mask'] = self.random_occluder.get_landmark_mask_from_mask_path(model_reconstructions['gt']['lmk_2d'], mask_info['frame_ids'], mask_info['mask_path'])
        print(model_reconstructions.keys())
        return model_reconstructions, motion_path

class GridVis:
    
    def __init__(self, config, model_cfg, test_data, device='cuda'):
        
        self.config = config
        self.model_cfg = model_cfg
        self.device = device
        self.test_data = test_data
        self.original_data_folder = 'dataset/mead_25fps/original_data' if config.dataset == 'mead_25fps' else 'dataset/RAVDESS'
        self.fps = 25
        self.sld_wind_size = 64
        self.model_types = config.model_types
        self.n_views = len(self.model_types) + 2 # add gt image
        self.with_audio = config.with_audio
        self.image_size = 224

        # IO setups
                        
        # name of the tested motion sequence
        self.output_folder = os.path.join(self.config.output_folder, 'acceleration')
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        logger.add(os.path.join(self.output_folder, f'acc.log'))
        logger.info(f"Using device {self.device}.")
        
        self._create_flame()
        self._setup_renderer()

    def _create_flame(self):
        self.flame_deca = FLAME_mediapipe(self.model_cfg.emoca).to(self.device)
        self.flame_emica = FLAME_mediapipe(self.model_cfg.model).to(self.device)
        # self.flametex = FLAMETex(self.model_cfg.emoca).to(self.device)  
        flame_vmask_path = "flame_2020/FLAME_masks.pkl"
        with open(flame_vmask_path, 'rb') as f:
            self.flame_v_mask = pickle.load(f, encoding="latin1")

        for k, v in self.flame_v_mask.items():
            self.flame_v_mask[k] = torch.from_numpy(v)  
    
    def _setup_renderer(self):
        self.render = SRenderY(
            self.model_cfg.model.image_size, 
            obj_filename=self.model_cfg.model.topology_path, 
            uv_size=self.model_cfg.model.uv_size,
            v_mask=self.flame_v_mask['face']).to(self.device)   # v_mask=self.flame_v_mask['face']
        # face mask for rendering details
        mask = imread(self.model_cfg.model.face_eye_mask_path).astype(np.float32)/255. 
        mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [self.model_cfg.model.uv_size, self.model_cfg.model.uv_size]).to(self.device)
        mask = imread(self.model_cfg.model.face_mask_path).astype(np.float32)/255.
        mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_mask = F.interpolate(mask, [self.model_cfg.model.uv_size, self.model_cfg.model.uv_size]).to(self.device)
        mean_texture = imread(self.model_cfg.model.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [self.model_cfg.model.uv_size, self.model_cfg.model.uv_size]).to(self.device)

    def acceleration_for_one_seq(self, lmk_3d, lmk_mask_all, fps):
        # lmk_3d: (n, V, 3)
        lmk_mask = lmk_mask_all == 0.
        acceleration = (
            (lmk_3d[1:] - lmk_3d[:-1] ).norm(dim=-1)
            * fps
        )   # (n-1, V)

        avg_acc = acceleration.mean(dim=1)  # (n-1)

        avg_acc_invis = []
        for i in range(acceleration.shape[0]):
            avg_acc_invis_frame = acceleration[i][lmk_mask[i+1]].mean()
            if avg_acc_invis_frame is None:
                avg_acc_invis_frame = torch.FloatTensor([0])
            avg_acc_invis.append(avg_acc_invis_frame)
        avg_acc_invis = torch.stack(avg_acc_invis)
        return avg_acc.numpy(), avg_acc_invis.numpy()
        
    def acceleration_computation(self, batch):
        
        frame_num = batch['gt']['shape'].shape[0]
        global_pose = batch['gt']['global_pose']
        shape = batch['gt']['shape']
        first_occ_frame_id = batch['diffusion']['mask_info']['frame_ids'][0]
        lmk_mask = batch['gt']['lmk_mask'][:, self.flame_emica.landmark_indices_mediapipe]
        acc_dict = {}

        for model_type in batch:
            batch_model = batch[model_type]
            if model_type in ['diffusion', 'gt']:
                exp = batch_model['exp']
                pose = torch.cat([global_pose, batch_model['jaw']], dim=-1)
                # flame decoder
                verts, lmk_3d = self.flame_emica(shape, exp, pose)
            elif model_type in ['emoca', 'deca', 'spectre']:
                exp = batch_model['exp']
                pose = torch.cat([global_pose, batch_model['jaw']], dim=-1)
                # flame decoder
                verts, lmk_3d = self.flame_deca(shape[:,:100], exp, pose)
            
            avg_acc, avg_acc_invis = self.acceleration_for_one_seq(lmk_3d, lmk_mask, 25)

            acc_dict[model_type] = {
                'avg_acc': avg_acc,
                'avg_acc_invis': avg_acc_invis,
            }
            if model_type == 'gt':
                acc_dict[model_type]['frame_ids'] = batch['diffusion']['mask_info']['frame_ids']
        print(acc_dict.keys())
        return acc_dict
             
            
    
    def run_acc_computation(self):
        num_test_motions = len(self.test_data)
        mode = 'visual' if 'deca' in self.model_types else 'speech'
        for i in tqdm(range(num_test_motions)):
            batch, motion_id = self.test_data[i]
            subject, view, emotion, level, sent = motion_id.split('/')
            if batch is None:
                continue

            acc_dict = self.acceleration_computation(batch)

            # save the acc dict
            save_path = os.path.join(self.output_folder, f"{motion_id}_acc.npy")
            Path(save_path).parent.mkdir(exist_ok=True, parents=True)

            np.save(save_path, acc_dict)

def main():
    # sample use: python3 test_with_random_mask/compute_acceleration.py --exp_name mouth --split test --with_audio
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, help='folder to store vis cases.', default='vis_result')
    parser.add_argument('--exp_name', type=str, help='.', default='non_occ')
    parser.add_argument('--vis_folder', type=str, help='folder to store vis cases.', default='vis_result')
    parser.add_argument('--split', type=str, help='mead split for evaluation.', default='test')
    parser.add_argument('--dataset', type=str, default='mead_25fps', help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='./dataset',help='dataset name')
    parser.add_argument('--model_types', type=list, default=['diffusion'])
    parser.add_argument(
        "--subject_id",
        default=None,
        type=str,
        help="subject id.",
    )

    parser.add_argument(
        "--level",
        default=None,
        type=str,
        help="emotion level.",
    )

    parser.add_argument(
        "--sent",
        default=None,
        type=int,
        help="sent id in MEAD [1 digit].",
    )

    parser.add_argument(
        "--emotion",
        default=None,
        type=str,
        help="emotion id in MEAD.",
    )

    parser.add_argument(
        "--with_audio",
        action="store_true",
        help="whether the input with audio.",
    )

    args = parser.parse_args()
    pretrained_args = get_cfg_defaults()

    args.model_types = ['diffusion', 'deca', 'emoca', 'spectre'] 

    diffusion_model_base = 'diffusion_Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc'    # base model to get the image mask
    
    subject_list = [args.subject_id] if args.subject_id else None
    level_list = [args.level] if args.level else None
    sent_list = [args.sent] if args.sent else None
    emotion_list = [args.emotion] if args.emotion else None
    args.output_folder = os.path.join(args.vis_folder, diffusion_model_base , args.exp_name)
    
    print(args.output_folder)
    assert os.path.exists(args.output_folder)
    exp_name = args.exp_name

    print("loading test data...")
    rec_folder_name = 'occ_mask'
    if args.dataset == 'mead_25fps':
        rec_paths = {
            'diffusion': os.path.join(args.vis_folder, diffusion_model_base , exp_name, rec_folder_name),
            'emoca': os.path.join(args.vis_folder, 'EMOCA', exp_name, rec_folder_name ),
            'deca': os.path.join(args.vis_folder, 'EMOCA', exp_name, rec_folder_name ),
            'spectre': os.path.join(args.vis_folder,'SPECTRE', exp_name, rec_folder_name ),
        }
        
        test_video_list_all = load_test_data(
            args.dataset, 
            args.dataset_path, 
            args.split, 
            subject_list=subject_list,
            emotion_list=emotion_list, 
            level_list=level_list, 
            sent_list=sent_list)

        test_video_list = []
        for motion_path, seqlen in test_video_list_all:
            if os.path.exists(os.path.join(rec_paths['diffusion'], f"{motion_path}_mask.npy")):
                test_video_list.append((motion_path, seqlen))
        
        print(f"number of test sequences: {len(test_video_list)}")

    
        test_dataset = TestMeadDataset(
            args.dataset,
            args.dataset_path,
            rec_paths,
            test_video_list,
            model_types=args.model_types
        )
    elif args.dataset == 'RAVDESS':
        exit(0)
    

    grid_vis = GridVis(args, pretrained_args, test_dataset, 'cpu')
    grid_vis.run_acc_computation()

if __name__ == "__main__":
    main()

