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

from matplotlib import cm

class TestRAVDESSDataset(Dataset):
    def __init__(
        self,
        dataset_name, 
        dataset_path,
        rec_paths,
        split_data,
        model_types=['diffusion']
    ):
        self.split_data = split_data
        self.rec_paths = rec_paths
        self.dataset = dataset_name
        self.model_types = model_types

        # paths to processed folder
        self.processed_folder = os.path.join(dataset_path, dataset_name, 'processed')
        self.cropped_landmark_folder = os.path.join(self.processed_folder,'cropped_landmarks_mediapipe')
        self.emoca_rec_folder = os.path.join(self.processed_folder, 'EMOCA_reconstruction')
        self.video_id_to_sent = {
            '01': 'KIDS ARE TALKING BY THE DOOR',
            '02': 'DOGS ARE SITTING BY THE DOOR'
        }
        self.video_folder = os.path.join(self.processed_folder, 'cropped_videos')
        self.random_occluder = RandomOcclusion()

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
        
        # get image
        rec_dict['original_image'] = self._get_video_input(motion_id)['image']

        return rec_dict
    
    def _get_diffusion_reconstruction(self, motion_path, rec_path):
        sample_path = os.path.join(rec_path, f"{motion_path}.npy")
        if not os.path.exists(sample_path):
            return None
        diffusion_sample = np.load(sample_path, allow_pickle=True)[()]
        diffusion_codes = {}
        diffusion_codes['jaw'] = torch.from_numpy(diffusion_sample[:,100:]).float()
        diffusion_codes['exp'] = torch.from_numpy(diffusion_sample[:,:100]).float()

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
        motion_path = self.split_data[idx]

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
                raise ValueError(f"{self.model_type} not supported!")
            
            if rec_dict is None:
                print(f'{model_type} is None')
                return None, None
            model_reconstructions[model_type] = rec_dict
        
        # for model in model_reconstructions:
        #         print(f"====={model}=====")
        #         for key in model_reconstructions[model]:
        #             print(f'shape {key} {model_reconstructions[model][key].shape}')
        mask_info = model_reconstructions['diffusion']['mask_info']
        model_reconstructions['gt']['image'] = self.random_occluder.get_image_with_mask(model_reconstructions['gt']['original_image'], mask_info['mask_path'], mask_info['frame_ids'])              
        print(model_reconstructions.keys())
        return model_reconstructions, motion_path


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

        with h5py.File(os.path.join(self.image_folder, motion_path, 'cropped_frames.hdf5'), "r") as f:
            code_dict['original_image'] = torch.from_numpy(f['images'][:]).float()

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
        model_reconstructions['gt']['image'] = self.random_occluder.get_image_with_mask(model_reconstructions['gt']['original_image'], mask_info['mask_path'], mask_info['frame_ids'])
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
        self.n_views = len(self.model_types) + 1 # add gt image
        self.with_audio = config.with_audio
        self.image_size = 224

        # IO setups
                        
        # name of the tested motion sequence
        self.output_folder = os.path.join(self.config.output_folder, 'grid_vis_vel_heatmaps')
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        logger.add(os.path.join(self.output_folder, f'grid_vis_vel_error.log'))
        logger.info(f"Using device {self.device}.")
        
        self._create_flame()
        self._setup_renderer()

        # setting of heatmaps
        self.colormap = cm.get_cmap('jet')
        self.min_error = 0.
        self.max_error = 500.

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

    def get_vertex_heat_color(self, vertex_error, faces=None):
        """
        Args:
            vertex_error: per vertex error [B, V]
        Return:
            face_colors: [B, nf, 3, 3]
        """
        B = vertex_error.shape[0]
        if faces is None:
            faces = self.render.faces.repeat(B, 1, 1)

        vertex_color_code = (((vertex_error - self.min_error) / (self.max_error - self.min_error)) * 255.).long().cpu().numpy()
        verts_rgb = torch.from_numpy(self.colormap(vertex_color_code)[:,:,:3]).to(self.device) # (B, V, 3)
        face_colors = face_vertices(verts_rgb, faces)
        return face_colors
    
    def compute_velocity_error(self, verts_pred, verts_gt, fps=25):

        gt_velocity = (verts_gt[1:, ...] - verts_gt[:-1, ...]) * fps
        pred_velocity = (verts_pred[1:, ...] - verts_pred[:-1, ...]) * fps
        vel_error = torch.sqrt(torch.sum(torch.square(gt_velocity - pred_velocity), axis=-1)) * 1000.0
        # vel_error = (pred_velocity - gt_velocity).abs() * 1000.0 # in mm/s [n-1, V]
        n, V = vel_error.shape[:2]
        vel_error = torch.cat([torch.zeros((1, V), device=vel_error.device), vel_error], dim=0)    # [n, V]

        return vel_error

    def vis_motion_split(self, motion_split):
        # prepare vis data dict 
        render_images = {}
        cam = motion_split['gt']['cam']
        global_pose = motion_split['gt']['global_pose']
        shape = motion_split['gt']['shape']
        # original_image = motion_split['gt']['original_image']
        image = motion_split['gt']['image']

        # gt mesh 
        exp_gt = motion_split['gt']['exp']
        pose_gt = torch.cat([global_pose, motion_split['gt']['jaw']], dim=-1)
        verts_gt, _ = self.flame_emica(shape, exp_gt, pose_gt)

        for model_type in self.model_types:
            motion_split_model = motion_split[model_type]
            if model_type in ['faceformer', 'facediffuser', 'codetalker', 'voca']:
                # shape template
                n = motion_split_model['verts'].shape[0]
                exp = torch.zeros((n, 100)).to(self.device)
                jaw = torch.zeros((n, 3)).to(self.device)
                pose = torch.cat([global_pose, jaw], dim=-1)
                shape_template, _ = self.flame_emica(shape,  exp, pose)
                verts = motion_split_model['verts'] + shape_template
                lmk_3d = self.flame_emica.select_lmk3d_mediapipe(verts)

            elif model_type in ['diffusion', 'emote']:
                exp = motion_split_model['exp']
                pose = torch.cat([global_pose, motion_split_model['jaw']], dim=-1)
                # flame decoder
                verts, lmk_3d = self.flame_emica(shape, exp, pose)
            elif model_type in ['emoca', 'deca', 'spectre']:
                exp = motion_split_model['exp']
                pose = torch.cat([global_pose, motion_split_model['jaw']], dim=-1)
                # flame decoder
                verts, lmk_3d = self.flame_deca(shape[:,:100], exp, pose)
        
            # orthogonal projection
            trans_verts = batch_orth_proj(verts, cam)
            trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
            
            vel_error = self.compute_velocity_error(verts, verts_gt)
            print(torch.max(vel_error))
            face_error_colors = self.get_vertex_heat_color(vel_error).to(self.device)
            heat_maps = self.render.render_shape(verts, trans_verts, colors=face_error_colors)
        
            render_images[model_type] = heat_maps

        for i in range(image.shape[0]):
            vis_dict = {}
            vis_dict['occ_img'] = image[i].detach().cpu()
            # vis_dict['gt_img'] = original_image[i].detach().cpu()
            for model_type in self.model_types:
                vis_dict[model_type] = render_images[model_type][i].detach().cpu()
            grid_image = self.visualize(vis_dict)
            if self.with_audio:
                self.writer.write(grid_image[:,:,[2,1,0]])
            else:
                self.writer.append_data(grid_image)
    
    def visualize(self, visdict, dim=2):
        '''s
        image range should be [0,1]
        dim: 2 for horizontal. 1 for vertical
        '''
        assert dim == 1 or dim==2
        # grids = {}
        # for key in visdict:
        #     _,h,w = visdict[key].shape
            # if dim == 2:
            #     new_h = size; new_w = int(w*size/h)
            # elif dim == 1:
            #     new_h = int(h*size/w); new_w = size
            # grids[key] = F.interpolate(visdict[key].unsqueeze(0), [new_h, new_w]).detach().cpu().squeeze(0)
        grid = torch.cat(list(visdict.values()), dim)
        grid_image = (grid.numpy().transpose(1,2,0).copy()*255)
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        # grid_image = cv2.resize(grid_image, (self.view_w,self.view_h))
        return grid_image

    
    def video_reconstruction(self, batch, v_name):
        
        if self.with_audio:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_fname = v_name + '.mp4'
            Path(video_fname).parent.mkdir(exist_ok=True, parents=True)
            self.writer = cv2.VideoWriter(video_fname, fourcc, self.fps, (self.image_size*self.n_views, self.image_size))
        else:
            gif_fname = v_name + '.gif'
            Path(gif_fname).parent.mkdir(exist_ok=True, parents=True)
            self.writer = imageio.get_writer(gif_fname, mode='I')
        
        frame_num = batch['gt']['shape'].shape[0]
        for start_id in range(0, frame_num, self.sld_wind_size):
            motion_split = {}
            for model_type in batch:
                motion_split[model_type] = {}
                for k in batch[model_type]:
                    if k not in ['mask_info']:
                        motion_split[model_type][k] = batch[model_type][k][start_id:start_id+self.sld_wind_size].to(self.device)
            self.vis_motion_split(motion_split)
        if self.with_audio:
            self.writer.release()
    
    def run_vis(self):
        num_test_motions = len(self.test_data)
        self.idx = 0
        mode = 'visual' if 'deca' in self.model_types else 'speech'
        for i in tqdm(range(num_test_motions)):
            batch, motion_id = self.test_data[i]
            if batch is None:
                continue
            video_path = self.output_folder + f'/{motion_id}'
            self.video_reconstruction(batch, video_path)
            if self.with_audio:
                if self.config.dataset == 'mead_25fps':
                    subject, view, emotion, level, sent = motion_id.split('/')
                    audio_path = os.path.join(self.original_data_folder, subject, 'audio', emotion, level, f"{sent}.m4a")
                    os.system(f"ffmpeg -i {video_path}.mp4 -i {audio_path} -c:v copy -c:a copy {video_path}_{mode}.mp4")
                else:
                    vocal, emotion, level, sent, rep, subject = motion_id.split('-')
                    audio_path = os.path.join(self.original_data_folder, 'audio','03-' + motion_id + '.wav')
                    os.system(f"ffmpeg -i {video_path}.mp4 -i {audio_path} -c:v copy {video_path}_{mode}.mp4")
                # assert os.path.exists(audio_path)
                os.system(f"rm {video_path}.mp4")
            torch.cuda.empty_cache()
            self.idx += 1

def main():
    # sample use: python3 test_with_random_mask/grid_visualization_vel.py --exp_name mouth --split test --with_audio
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
        pretrained_args.model.n_shape = 100
        rec_paths = {
            'diffusion': os.path.join(args.vis_folder, diffusion_model_base , exp_name, rec_folder_name),
            'emoca': os.path.join(args.vis_folder, 'EMOCA', exp_name, rec_folder_name ),
            'deca': os.path.join(args.vis_folder, 'EMOCA', exp_name, rec_folder_name ),
            'spectre': os.path.join(args.vis_folder,'SPECTRE', exp_name, rec_folder_name ),
        }

        from data_loaders.dataloader_RAVDESS import load_RAVDESS_test_data
        test_video_list_all = load_RAVDESS_test_data(
            args.dataset, 
            args.dataset_path,
            subject_list=subject_list,
            emotion_list=emotion_list, 
            level_list=level_list, 
            sent_list=sent_list)
        
        test_video_list = []
        for motion_path in test_video_list_all:
            if os.path.exists(os.path.join(rec_paths['diffusion'], f"{motion_path}_mask.npy")):
                test_video_list.append(motion_path)
        
        print(f"number of test sequences: {len(test_video_list)}")

        test_dataset = TestRAVDESSDataset(
            args.dataset,
            args.dataset_path,
            rec_paths,
            test_video_list,
            model_types=args.model_types
        )
    

    grid_vis = GridVis(args, pretrained_args, test_dataset, 'cuda')
    grid_vis.run_vis()

if __name__ == "__main__":
    main()

