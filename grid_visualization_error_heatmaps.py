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
from matplotlib import cm

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

        # with h5py.File(os.path.join(self.image_folder, motion_path, 'cropped_frames.hdf5'), "r") as f:
        #     code_dict['image'] = torch.from_numpy(f['images'][:]).float()

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
            mask = np.load(mask_path, allow_pickle=True)[()]
            diffusion_codes['mask'] = torch.from_numpy(mask).float()
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
    
    def _get_verts_reconstruction(self, motion_path, rec_path):
        sample_path = os.path.join(rec_path, f"{motion_path}.npy")
        if not os.path.exists(sample_path):
            print(f"recons not existed for {sample_path}")
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
        model_reconstructions = {}
        gt_flame = self._get_emica_codes(motion_path)
        model_reconstructions['gt'] = gt_flame
        for model_type in self.model_types:
            rec_path = self.rec_paths[model_type]
            if model_type in ['faceformer', 'facediffuser', 'codetalker', 'voca']:
                rec_dict = self._get_verts_reconstruction(motion_path, rec_path)
                n_verts = rec_dict['verts'].shape[0]
                num_frames = gt_flame['shape'].shape[0]
                if num_frames < n_verts:
                    rec_dict['verts'] = rec_dict['verts'][:num_frames]
                elif num_frames > n_verts:
                    rec_dict['verts'] = torch.cat([rec_dict['verts'], rec_dict['verts'][-1:].repeat(num_frames-n_verts, 1, 1)], dim=0)
                assert gt_flame['shape'].shape[0] == rec_dict['verts'].shape[0]
            elif model_type in ['diffusion', 'emote']:
                rec_dict = self._get_diffusion_reconstruction(motion_path, rec_path)
            elif model_type in ['emoca', 'deca', 'spectre']:
                rec_dict = self._get_emoca_reconstruction(motion_path, model_type, rec_path)
            else:
                raise ValueError(f"{model_type} not supported!")
            if rec_dict is None:
                return None, None
            model_reconstructions[model_type] = rec_dict

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
        self.colormap = cm.get_cmap('jet')

        # IO setups
                        
        # name of the tested motion sequence
        self.output_folder = os.path.join(self.config.output_folder, 'grid_vis')
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        logger.add(os.path.join(self.output_folder, f'grid_vis.log'))
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

    def get_vertex_heat_color(self, vertex_error, min_error, max_error, faces=None):
        """
        Args:
            vertex_error: per vertex error [B, V]
        Return:
            face_colors: [B, nf, 3, 3]
        """
        B = vertex_error.shape[0]
        if faces is None:
            faces = self.render.faces.repeat(B, 1, 1)

        vertex_error = vertex_error.cpu()
        vertex_color_code = ((1 - (vertex_error - min_error) / (max_error - min_error)) * 255.).numpy().astype(int)
        verts_rgb = torch.from_numpy(self.colormap(vertex_color_code)[:,:,:3]).to(self.device) # (B, V, 3)
        face_colors = face_vertices(verts_rgb, faces)
        return face_colors

    def vertex_heatmap(self, vertex_error_all, velocity_error_all):

        # get template mesh
        rec_path = 'dataset/RAVDESS/processed/EMOCA_reconstruction/01-04-02-01-01-04.npy'
        gt_rec = np.load(rec_path, allow_pickle=True)[()]
        shape = torch.from_numpy(gt_rec['shape'][:1]).float()
        shape = torch.cat([shape, torch.ones(1, 200)], dim=-1).to(self.device)
        cam = torch.from_numpy(gt_rec['cam'][:1]).float().to(self.device)
        global_pose = torch.from_numpy(gt_rec['pose'][:1,:3]).float().to(self.device)
        exp = torch.zeros((1, 100)).to(self.device)
        jaw = torch.zeros((1, 3)).to(self.device)
        pose = torch.cat([global_pose, jaw], dim=-1).to(self.device)

        verts, _ = self.flame_emica(shape, exp, pose) # (1, V, 3)
        trans_verts = batch_orth_proj(verts, cam)
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        min_ve, min_vel = 1000.0, 1000.0
        max_ve, max_vel = -1.0, -1.0
        heatmaps = {}

        for model_type in self.model_types:
            print(f"max_ve_all {torch.max(vertex_error_all[model_type])}")
            min_ve = min(min_ve, torch.min(vertex_error_all[model_type]))
            max_ve = max(max_ve, torch.max(vertex_error_all[model_type]))

            print(f"max_vel_all {torch.max(velocity_error_all[model_type])}")
            min_vel = min(min_vel, torch.min(velocity_error_all[model_type]))
            max_vel = max(max_vel, torch.max(velocity_error_all[model_type]))
        
        print(f"max_ve {max_ve}")
        print(f"max_vel {max_vel}")

        for model_type in self.model_types:
            vertex_error = vertex_error_all[model_type].unsqueeze(0)
            velocity_error = velocity_error_all[model_type].unsqueeze(0)

            # draw vertex error heatmap
            face_error_colors = self.get_vertex_heat_color(vertex_error, 0, 15)
            vertex_error_heat_map = self.render.render_shape(verts, trans_verts, colors=face_error_colors, black_bg=False)[0]   # (3, h, w)

            # draw velocity error heatmap
            face_error_colors = self.get_vertex_heat_color(velocity_error, 0, 80)
            velocity_error_heat_map = self.render.render_shape(verts, trans_verts, colors=face_error_colors, black_bg=False)[0]  # (3, h, w)

            # concat vertically
            heatmap = torch.cat([vertex_error_heat_map, velocity_error_heat_map], dim=1)

            heatmaps[model_type] = heatmap.detach().cpu()
        
        grid_image = self.visualize(heatmaps)

        img_path = os.path.join(self.output_folder, 'avg_error_heatmap.png')
        cv2.imwrite(img_path, grid_image)

    def visualize(self, visdict, dim=2):
        '''
        image range should be [0,1]
        dim: 2 for horizontal. 1 for vertical
        '''
        assert dim == 1 or dim==2
        grid = torch.cat(list(visdict.values()), dim)
        grid_image = (grid.numpy().transpose(1,2,0).copy()*255)
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        # grid_image = cv2.resize(grid_image, (self.view_w,self.view_h))
        return grid_image

    def compute_velocity_error(self, verts_pred, verts_gt, fps=25):

        gt_velocity = (verts_gt[1:, ...] - verts_gt[:-1, ...]) * fps
        pred_velocity = (verts_pred[1:, ...] - verts_pred[:-1, ...]) * fps
        vel_error = torch.sqrt(torch.sum(torch.square(gt_velocity - pred_velocity), dim=-1)) * 1000.0
        # vel_error = (pred_velocity - gt_velocity).abs() * 1000.0 # in mm/s [n-1, V]
        n, V = vel_error.shape[:2]
        vel_error = torch.cat([torch.zeros((1, V), device=vel_error.device), vel_error], dim=0)    # [n, V]

        return vel_error
    
    def error_computation(self, batch):
        
        frame_num = batch['gt']['shape'].shape[0]
        vertex_error = {}
        velocity_error = {}
        global_pose = batch['gt']['global_pose'].to(self.device)
        shape = batch['gt']['shape'].to(self.device)

        # gt mesh 
        exp_gt = batch['gt']['exp'].to(self.device)
        pose_gt = torch.cat([global_pose, batch['gt']['jaw'].to(self.device)], dim=-1)
        verts_gt, _ = self.flame_emica(shape, exp_gt, pose_gt)

        mask = batch['diffusion']['mask'].bool()   # (n, 224, 224)
        occ_frame_id = ~(mask.all(1).all(1))   # (n)
        occ_frame_id = occ_frame_id.to(self.device)

        for model_type in self.model_types:
            motion_split_model = batch[model_type]

            if model_type in ['diffusion', 'emote']:
                exp = motion_split_model['exp'].to(self.device)
                pose = torch.cat([global_pose, motion_split_model['jaw'].to(self.device)], dim=-1)
                # flame decoder
                verts, lmk_3d = self.flame_emica(shape, exp, pose)
            elif model_type in ['emoca', 'deca', 'spectre']:
                exp = motion_split_model['exp'].to(self.device)
                pose = torch.cat([global_pose, motion_split_model['jaw'].to(self.device)], dim=-1)
                # flame decoder
                verts, lmk_3d = self.flame_deca(shape[:,:100], exp, pose)
            
            
            vertex_error[model_type] = (verts_gt - verts).norm(dim=-1)[occ_frame_id].mean(dim=0).cpu() * 1000.0   # (V, )

            velocity_error[model_type] = self.compute_velocity_error(verts, verts_gt)[occ_frame_id].mean(dim=0).cpu()    # (V, )
        

        return vertex_error, velocity_error
    
    def run_vis(self):
        num_test_motions = len(self.test_data)
        cnt = 0
        V = 5023
        vertex_error_all = {}
        velocity_error_all = {}
        for model in self.model_types:
            velocity_error_all[model] = torch.zeros(V)
            vertex_error_all[model] = torch.zeros(V)

        for i in tqdm(range(num_test_motions)):
            batch, motion_id = self.test_data[i]
            if batch is None:
                continue
            cnt += 1
            
            vertex_error, velocity_error = self.error_computation(batch)
            for model_type in self.model_types:
                vertex_error_all[model_type] += vertex_error[model_type]
                velocity_error_all[model_type] += velocity_error[model_type]
            
            torch.cuda.empty_cache()
        
        for model_type in self.model_types:
            vertex_error_all[model_type] /= cnt
            velocity_error_all[model_type] /= cnt
            
        print(f"cnt = {cnt}")
        
        self.vertex_heatmap(vertex_error_all, velocity_error_all)

def main():
    # sample use:
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

    # args.model_types = ['diffusion', 'emote', 'faceformer', 'facediffuser', 'codetalker'] 
    if args.exp_name == 'missing_frames':
        args.model_types = ['diffusion', 'spectre'] 
    else:
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
    if args.dataset == 'mead_25fps':
        rec_paths = {
            'diffusion': os.path.join(args.vis_folder, diffusion_model_base , exp_name, 'diffusion_sample'),
            'emoca': os.path.join(args.vis_folder, 'EMOCA', exp_name, 'EMOCA_reconstruction'),
            'deca': os.path.join(args.vis_folder, 'EMOCA', exp_name, 'EMOCA_reconstruction'),
            'spectre': os.path.join(args.vis_folder,'SPECTRE', exp_name, 'SPECTRE_reconstruction'),
            'faceformer': os.path.join(args.vis_folder, 'FaceFormer', 'reconstruction'),
            'facediffuser': os.path.join(args.vis_folder, 'FaceDiffuser', 'reconstruction'),
            'codetalker': os.path.join(args.vis_folder, 'CodeTalker', 'reconstruction'),
        }
        
        test_video_list = load_test_data(
            args.dataset, 
            args.dataset_path, 
            args.split, 
            subject_list=subject_list,
            emotion_list=emotion_list, 
            level_list=level_list, 
            sent_list=sent_list)
        shuffle_id = np.random.permutation(len(test_video_list))
        print(f"number of test sequences: {len(test_video_list)}")

    
        test_dataset = TestMeadDataset(
            args.dataset,
            args.dataset_path,
            rec_paths,
            test_video_list[shuffle_id],
            model_types=args.model_types
        )
    elif args.dataset == 'RAVDESS':
        exit(0)

    grid_vis = GridVis(args, pretrained_args, test_dataset, 'cuda')
    grid_vis.run_vis()

if __name__ == "__main__":
    main()

