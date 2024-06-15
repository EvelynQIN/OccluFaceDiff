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

        with h5py.File(os.path.join(self.image_folder, motion_path, 'cropped_frames.hdf5'), "r") as f:
            code_dict['image'] = torch.from_numpy(f['images'][:]).float()

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

    def __getitem__(self, idx):
        motion_path, _ = self.split_data[idx]
        model_reconstructions = {}
        gt_flame = self._get_emica_codes(motion_path)
        model_reconstructions['gt'] = gt_flame
        for model_type in self.model_types:
            rec_path = self.rec_paths[model_type]
            rec_dict = self._get_diffusion_reconstruction(motion_path, rec_path)
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

        # IO setups
                        
        # name of the tested motion sequence
        self.output_folder = os.path.join(self.config.output_folder, 'grid_vis_ablation')
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

    def vis_motion_split(self, motion_split):
        # prepare vis data dict 
        render_images = {}
        cam = motion_split['gt']['cam']
        global_pose = motion_split['gt']['global_pose']
        shape = motion_split['gt']['shape']
        image = motion_split['gt']['image']
        mask = motion_split['full'].get('mask', None)

        for model_type in self.model_types:
            motion_split_model = motion_split[model_type]

            exp = motion_split_model['exp']
            pose = torch.cat([global_pose, motion_split_model['jaw']], dim=-1)
            # flame decoder
            verts, lmk_3d = self.flame_emica(shape, exp, pose)
        
            # orthogonal projection
            trans_verts = batch_orth_proj(verts, cam)
            trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

            # lmk_2d = batch_orth_proj(lmk_3d, cam)
            # lmk_2d[:, :, 1:] = -lmk_2d[:, :, 1:]
        
            render_image = self.render.render_shape(verts, trans_verts, images = image) # images = image
            render_images[model_type] = render_image

        if mask is not None:
            image = image * mask.unsqueeze(1)
        for i in range(image.shape[0]):
            vis_dict = {}
            vis_dict['gt_img'] = image[i].detach().cpu()
            for model_type in self.model_types:
                vis_dict[model_type] = render_images[model_type][i].detach().cpu()
            grid_image = self.visualize(vis_dict)
            if self.with_audio:
                self.writer.write(grid_image[:,:,[2,1,0]])
            else:
                self.writer.append_data(grid_image)
    
    def visualize(self, visdict, dim=2):
        '''
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
                    motion_split[model_type][k] = batch[model_type][k][start_id:start_id+self.sld_wind_size].to(self.device)
            self.vis_motion_split(motion_split)
        if self.with_audio:
            self.writer.release()
    
    def run_vis(self):
        num_test_motions = len(self.test_data)
        self.idx = 0
        mode = 'ablation' 
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

    args.model_types = ['full', 'woflint', 'woalign', 'wofilm', 'wodiffusion'] 

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
            'full': os.path.join(args.vis_folder, diffusion_model_base , exp_name, 'diffusion_sample'),
            'woflint': os.path.join(args.vis_folder, 'diffusion_Transformer_768d_cat_mediapipelmk_final', exp_name, 'diffusion_sample'),
            'woalign': os.path.join(args.vis_folder, 'diffusion_Transformer_768d_cat_mediapipelmk_FLINT_wo_align', exp_name, 'diffusion_sample'),
            'wofilm': os.path.join(args.vis_folder, 'diffusion_Transformer_768d_cat_mediapipelmk_FLINT_woFiLM', exp_name, 'diffusion_sample'),
            'wodiffusion': os.path.join(args.vis_folder, 'pureTrans', exp_name, 'reconstruction'),
        }
        
        test_video_list = load_test_data(
            args.dataset, 
            args.dataset_path, 
            args.split, 
            subject_list=subject_list,
            emotion_list=emotion_list, 
            level_list=level_list, 
            sent_list=sent_list)
        shuffle_id = np.random.permutation(len(test_video_list))[:20]
        print(f"number of test sequences: {len(test_video_list)}")

    
        test_dataset = TestMeadDataset(
            args.dataset,
            args.dataset_path,
            rec_paths,
            test_video_list[shuffle_id],
            model_types=args.model_types
        )
    else:
        print(f"dataset not supported!")
        exit(0)
    

    grid_vis = GridVis(args, pretrained_args, test_dataset, 'cuda')
    grid_vis.run_vis()

if __name__ == "__main__":
    main()

