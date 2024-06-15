""" Test motion recontruction with with landmark and audio as input.
"""
import os
import random
import numpy as np
from tqdm import tqdm
import cv2
from enum import Enum
import os.path
from glob import glob
from pathlib import Path
import subprocess
from loguru import logger
from time import time
from matplotlib import cm
from copy import deepcopy
from collections import defaultdict

import sys
sys.path.append('./')
from diffusion.cfg_sampler import ClassifierFreeSampleModel

from utils import utils_transform, utils_visualize
from utils.metrics import get_metric_function
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import test_args
from model.FLAME import FLAME_mediapipe
from configs.config import get_cfg_defaults
from utils import dataset_setting
from utils.renderer import SRenderY
from utils.data_util import batch_orth_proj, face_vertices
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
import ffmpeg
import pickle
from model.wav2vec import Wav2Vec2Model
from munch import Munch, munchify
from model.motion_prior import L2lVqVae

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class MotionTracker:
    
    def __init__(self, config, model_cfg, test_data, device='cpu'):
        
        self.config = config
        self.model_cfg = model_cfg
        self.device = device
        self.sld_wind_size = config.sld_wind_size
        self.input_motion_length = config.input_motion_length
        self.target_nfeat = config.n_exp + config.n_pose
        self.flint_dim = 128
        self.flint_factor = 8
        self.test_data = test_data
        self.original_data_folder = 'dataset/mead_25fps/original_data' if config.test_dataset == 'MEAD' else 'dataset/RAVDESS'
        self.sample_size = 5 # num of inference times for computing std

                        
        # name of the tested motion sequence
        self.save_folder = self.config.save_folder
        self.output_folder = os.path.join(self.save_folder, self.config.arch, config.exp_name)
        
        logger.add(os.path.join(self.output_folder, 'test_diversity.log'))
        logger.info(f"Using device {self.device}.")

        # vis settings
        self.to_mp4 = True # if true then to mp4, false then to gif wo audio
        self.image_size = config.image_size
        
        self.sample_time = 0

        # heatmap visualization settings
        self.colormap = cm.get_cmap('jet')
        self.min_error = 0.
        self.max_error = 3.

        # diffusion models
        self.load_diffusion_model_from_ckpt(config, model_cfg)
        
        self._create_flame()
        self._setup_renderer()
    
    def _create_flame(self):

        self.flame = FLAME_mediapipe(self.model_cfg).to('cpu')
        flame_template_file = 'flame_2020/head_template_mesh.obj'
        self.faces = load_obj(flame_template_file)[1]

        flame_vmask_path = "flame_2020/FLAME_masks.pkl"
        with open(flame_vmask_path, 'rb') as f:
            self.flame_v_mask = pickle.load(f, encoding="latin1")

        for k, v in self.flame_v_mask.items():
            self.flame_v_mask[k] = torch.from_numpy(v)
    
    def _setup_renderer(self):

        self.render = SRenderY(
            self.model_cfg.image_size, 
            obj_filename=self.model_cfg.topology_path, 
            uv_size=self.model_cfg.uv_size,
            v_mask=self.flame_v_mask['face']
            )
        # face mask for rendering details
        mask = imread(self.model_cfg.face_eye_mask_path).astype(np.float32)/255. 
        mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)
        mask = imread(self.model_cfg.face_mask_path).astype(np.float32)/255.
        mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_mask = F.interpolate(mask, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)
        # # TODO: displacement correction
        # fixed_dis = np.load(self.model_cfg.fixed_displacement_path)
        # self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        mean_texture = imread(self.model_cfg.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)
        # # dense mesh template, for save detail mesh
        # self.dense_template = np.load(self.model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()

    def load_diffusion_model_from_ckpt(self, args, model_cfg):
        logger.info("Creating model and diffusion...")
        args.arch = args.arch[len("diffusion_") :]
        self.denoise_model, self.diffusion = create_model_and_diffusion(args, model_cfg, self.device)

        logger.info(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location="cpu")
        self.denoise_model.load_state_dict(state_dict, strict=True)

        # wrap the model with cfg sampler
        if args.guidance_param_audio is not None:
            self.denoise_model = ClassifierFreeSampleModel(self.denoise_model)

        self.denoise_model.to(self.device)  # dist_util.dev())
        self.denoise_model.eval()  # disable random masking

        self.diffusion_input_keys = ['lmk_2d', 'lmk_mask', 'audio_input']
    
    
    def sample_motion_non_overlap(self, batch_split):
        sample_fn = self.diffusion.ddim_sample_loop

        sample_list = []

        with torch.no_grad():
            bs, n = batch_split['lmk_2d'].shape[:2]
            split_length = n // self.flint_factor
            
            model_kwargs = {}
            for key in self.diffusion_input_keys:
                model_kwargs[key] = batch_split[key].to(self.device)
            model_kwargs['audio_input'] = model_kwargs['audio_input'].reshape(bs, -1)

            noise = None
            
            # add CFG scale to batch
            if self.config.guidance_param_audio is not None:
                model_kwargs["y"] = {}
                model_kwargs["y"]['scale_all'] = torch.ones(bs, device=self.device) * self.config.guidance_param_all
                model_kwargs["y"]['scale_audio'] = torch.ones(bs, device=self.device) * self.config.guidance_param_audio
            
            for i in tqdm(range(self.sample_size)):
                output_sample = sample_fn(
                    self.denoise_model,
                    (bs, split_length, self.flint_dim),
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,
                    init_image=None,
                    progress=False,
                    dump_steps=None,
                    noise=noise,
                    const_noise=False,
                )
                flint_output = self.diffusion.flint_decoder(output_sample) # (bs, k, c)
                sample_list.append(flint_output.to('cpu').unsqueeze(0))

            sample_list = torch.cat(sample_list, dim=0) # (sample_size, bs, k, c)
        return sample_list
                
    def evaluate_diversity(
        self,
        batch,
        sample_list,    # (sample_size, nframes, 103)
    ):      
        global_rot_aa = batch['global_pose'].unsqueeze(0).repeat(self.sample_size, 1, 1).to('cpu')
        shape_gt = batch['shape'].unsqueeze(0).repeat(self.sample_size, 1, 1).to('cpu')
        diff_expr = sample_list[...,:self.config.n_exp]
        diff_jaw_aa = sample_list[...,self.config.n_exp:]
        diff_rot_aa = torch.cat([global_rot_aa, diff_jaw_aa], dim=-1)

        bs, n = diff_rot_aa.shape[:2]
        
        shape_gt = shape_gt.reshape(bs*n, *shape_gt.shape[2:])
        diff_expr = diff_expr.reshape(bs*n, *diff_expr.shape[2:])
        diff_rot_aa = diff_rot_aa.reshape(bs*n, *diff_rot_aa.shape[2:])   # (bsxn, c)
        # flame decoder
        verts_all, _ = self.flame(shape_gt, diff_expr, diff_rot_aa)    # (bsxn, V, 3)
        # verts_face = torch.index_select(verts_pred, 1, self.flame_v_mask['face']) # (bsxn, V_face, 3)

        verts_all = verts_all.reshape(bs, n, -1, 3) # (bs, n, V, 3)
        
        c_dist = []
        v_dist = []
        for i in range(self.sample_size-1):
            for j in range(i+1, self.sample_size):
                dist_ij = torch.norm(verts_all[i]-verts_all[j], p=2, dim=-1) * 1000.0 # (n, v) to mm
                dist_ij_face = torch.index_select(dist_ij, 1, self.flame_v_mask['face']) # (n, V_face)
                c_dist.append(torch.mean(dist_ij_face).item())
                v_dist.append(torch.mean(dist_ij, dim=0)) # (v)
        v_dist = torch.stack(v_dist) # (sample_size, V)
        diversity = sum(c_dist) / len(c_dist)
        diversity_vertex = torch.mean(v_dist, dim=0)    # (v, )
        return diversity, diversity_vertex

    def prepare_chunk_diffusion_input(self, batch, start_id, num_frames):
        motion_split = {}
        flag_index = 0
        for key in self.diffusion_input_keys:
            motion_split[key] = batch[key][start_id:start_id+self.input_motion_length]

        # if sequnce is short, pad with same motion to the beginning
        if motion_split['lmk_2d'].shape[0] < self.input_motion_length:
            flag_index = self.input_motion_length - motion_split['lmk_2d'].shape[0]
            for key in motion_split:
                original_split = motion_split[key]
                n_dims = len(original_split.shape)
                if n_dims == 2:
                    tmp_init = original_split[:1].repeat(flag_index, 1).clone()
                elif n_dims == 3:
                    tmp_init = original_split[:1].repeat(flag_index, 1, 1).clone()
                elif n_dims == 4:
                    tmp_init = original_split[:1].repeat(flag_index, 1, 1, 1).clone()
                motion_split[key] = torch.concat([tmp_init, original_split], dim=0)
        return motion_split, flag_index

    def non_overlapping_inference(self, batch):

        if self.num_frames < self.input_motion_length:
            # pad the beginning frames
            flag_index = self.input_motion_length - self.num_frames
            batch_split = deepcopy(batch)
            batch_split['lmk_2d'] = torch.cat([batch_split['lmk_2d'][:1].repeat(flag_index, 1, 1), batch_split['lmk_2d']], dim=0).unsqueeze(0)    # (1, k, c)
            batch_split['lmk_mask'] = torch.cat([batch_split['lmk_mask'][:1].repeat(flag_index, 1), batch_split['lmk_mask']], dim=0).unsqueeze(0)    # (1, k, c)
            batch_split['audio_input'] = torch.cat([batch_split['audio_input'][:1].repeat(flag_index, 1), batch_split['audio_input']], dim=0).unsqueeze(0)    # (1, k, c)
            flag_index = [flag_index]
        else:
            start_id = 0
            batch_split = defaultdict(list)
            flag_index = []
            while start_id + self.input_motion_length <= self.num_frames:
                for key in self.diffusion_input_keys:
                    batch_split[key].append(batch[key][start_id:start_id+self.input_motion_length].unsqueeze(0))
                flag_index.append(0)
                start_id += self.input_motion_length
            
            if start_id < self.num_frames:
                flag_index.append(self.input_motion_length - self.num_frames + start_id)
                for key in self.diffusion_input_keys:
                    batch_split[key].append(batch[key][-self.input_motion_length:].unsqueeze(0))
            for key in batch_split:
                batch_split[key] = torch.cat(batch_split[key], dim=0)

        sample = self.sample_motion_non_overlap(batch_split).cpu()  # (sample_size, bs, k, c)

        final_output = []
        for i in range(sample.shape[1]):
            final_output.append(sample[:, i, flag_index[i]:])
        final_output = torch.cat(final_output, dim=1)

        assert final_output.shape[1] == self.num_frames
        return final_output 
    
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

        vertex_error = vertex_error.cpu().numpy()
        vertex_color_code = ((1 - (vertex_error - self.min_error) / (self.max_error - self.min_error)) * 255.).astype(int)
        verts_rgb = torch.from_numpy(self.colormap(vertex_color_code)[:,:,:3]) # (B, V, 3)
        face_colors = face_vertices(verts_rgb, faces)
        return face_colors

    def vertex_diversity_heatmap(self, vertex_diversity):

        # get template mesh
        rec_path = 'dataset/RAVDESS/processed/EMOCA_reconstruction/01-04-02-01-01-04.npy'
        gt_rec = np.load(rec_path, allow_pickle=True)[()]
        shape = torch.from_numpy(gt_rec['shape'][:1]).float()
        shape = torch.cat([shape, torch.ones(1, 200)], dim=-1)
        cam = torch.from_numpy(gt_rec['cam'][:1]).float()
        global_pose = torch.from_numpy(gt_rec['pose'][:1,:3]).float()
        exp = torch.zeros((1, 100))
        jaw = torch.zeros((1, 3))
        pose = torch.cat([global_pose, jaw], dim=-1)

        verts, _ = self.flame(shape, exp, pose) # (1, V, 3)
        trans_verts = batch_orth_proj(verts, cam)
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        face_error_colors = self.get_vertex_heat_color(vertex_diversity.unsqueeze(0))
        heat_map = self.render.render_shape(verts, trans_verts, colors=face_error_colors)[0]
        
        # tensor to numpy image array
        grid_image = (heat_map.numpy().transpose(1,2,0).copy()*255)
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        
        img_path = os.path.join(self.output_folder, 'vertex_diversity.png')
        cv2.imwrite(img_path, grid_image)

            
    def track(self):
        
        # make prediction by split the whole sequences into chunks
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        eval_all = {
            'diversity': 0.0,
            'diversity_vertex': torch.zeros(5023)
        }

        num_test_motions = len(self.test_data)
        eval_motion_num = 0
        for i in tqdm(range(num_test_motions)):
            batch, motion_id = self.test_data[i]
            self.num_frames = batch['lmk_2d'].shape[0]
            if batch['exp'].shape[1] != self.config.n_exp:
                batch['exp'] = torch.cat([batch['exp'], torch.ones(self.num_frames, 50)], dim=-1)
            if self.num_frames < 25:
                logger.info(f'[{motion_id}] is shorter than 1 sec, skipped.')
                continue
            eval_motion_num += 1
            logger.info(f'Process [{motion_id}]. with {self.num_frames} frames')
            # diffusion sample

            diffusion_output = self.non_overlapping_inference(batch)
            
            # start evaluation
            diversity, diversity_vertex = self.evaluate_diversity(batch, diffusion_output)
                 
            logger.info(f"diversity : {diversity} mm")
            eval_all['diversity'] += diversity
            eval_all['diversity_vertex'] += diversity_vertex
            torch.cuda.empty_cache()

        logger.info("==========Metrics for all test motion sequences:===========")
        for metric in eval_all:
            eval_all[metric] = eval_all[metric] / eval_motion_num
            if metric == 'diversity':
                logger.info(f"{metric} : {eval_all[metric]} mm")
        print('max vertex diveristy: ', torch.max(eval_all['diversity_vertex']))
        
        self.vertex_diversity_heatmap(eval_all['diversity_vertex'])

def main():
    args = test_args()

    # args.timestep_respacing = '100' # use DDIM samper
    pretrained_args = get_cfg_defaults()
    # to guanrantee reproducibility
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("loading test data...")
    subject_list = [args.subject_id] if args.subject_id else None
    level_list = [args.level] if args.level else None
    sent_list = [args.sent] if args.sent else None
    emotion_list = [args.emotion] if args.emotion else None

    if args.test_dataset == "MEAD":
        from data_loaders.dataloader_MEAD_flint import load_test_data, TestMeadDataset
        test_video_list = load_test_data(
            args.dataset, 
            args.dataset_path, 
            args.split, 
            subject_list=subject_list,
            emotion_list=emotion_list, 
            level_list=level_list, 
            sent_list=sent_list)

        print(f"number of test sequences: {len(test_video_list)}")
        test_dataset = TestMeadDataset(
            args.dataset,
            args.dataset_path,
            test_video_list[:2],
            args.fps,
            args.n_shape,
            args.n_exp,
            args.exp_name,
            args.load_tex,
            args.use_iris,
            load_audio_input=True,
            vis=args.vis,
            use_segmask=True,
            mask_path=args.mask_path
        )
    elif args.test_dataset == "RAVDESS":
        pretrained_args.model.n_shape = 100
        from data_loaders.dataloader_RAVDESS import load_RAVDESS_test_data, TestRAVDESSDataset
        test_video_list = load_RAVDESS_test_data(
            args.test_dataset, 
            args.dataset_path, 
            subject_list=subject_list,
            emotion_list=emotion_list, 
            level_list=level_list, 
            sent_list=sent_list)
        
        print(f"number of test sequences: {len(test_video_list)}")
        test_dataset = TestRAVDESSDataset(
            args.test_dataset,
            args.dataset_path,
            test_video_list,
            args.fps,
            args.exp_name,
            args.use_iris,
            load_audio_input=True,
            vis=args.vis,
            mask_path=args.mask_path
        )
    else:
        raise ValueError(f"{args.test_dataset} not supported!")

    motion_tracker = MotionTracker(args, pretrained_args.model, test_dataset, 'cuda')
    
    motion_tracker.track()

if __name__ == "__main__":
    main()
