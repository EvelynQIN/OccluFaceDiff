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
    
    def __init__(self, config, model_cfg, test_data, device='cuda'):
        
        self.config = config
        self.model_cfg = model_cfg
        self.device = device
        self.sld_wind_size = config.sld_wind_size
        self.input_motion_length = config.input_motion_length
        self.target_nfeat = config.n_exp + config.n_pose
        self.flint_dim = 128
        self.flint_factor = 8
        self.test_data = test_data
        self.original_data_folder = 'dataset/mead_25fps/original_data'
        self.vis = config.vis
        self.save_rec = config.save_rec
        # IO setups
                        
        # name of the tested motion sequence
        self.save_folder = self.config.save_folder
        self.output_folder = os.path.join(self.save_folder, self.config.arch, config.exp_name)
        self.sample_folder = os.path.join(self.output_folder, 'reconstruction')
        if self.save_rec:
            if not os.path.exists(self.sample_folder):
                os.makedirs(self.sample_folder)
        
        logger.add(os.path.join(self.output_folder, 'test_mead_wrt_gt.log'))
        logger.info(f"Using device {self.device}.")

        # vis settings
        self.to_mp4 = True # if true then to mp4, false then to gif wo audio
        self.visualization_batch = 32
        self.image_size = config.image_size
        self.resize_factor=1.0  # resize the final grid image
        self.heatmap_view = True
        if self.heatmap_view:
            self.n_views = 5
        else:
            self.n_views = 4
        self.view_h, self.view_w = int(self.image_size*self.resize_factor), int(self.image_size*self.n_views*self.resize_factor)
        
        self.sample_time = 0

        # heatmap visualization settings
        self.colormap = cm.get_cmap('jet')
        self.min_error = 0.
        self.max_error = 10.

        # diffusion models
        self.load_diffusion_model_from_ckpt(config, model_cfg)
        
        self._create_flame()
        self._setup_renderer()

        # eval metrics
        pred_metrics = [
            "pred_jitter",
            "mvpe",
            "mvve",
            "expre_error",
            "pose_error",
            "lmk_3d_mvpe",
            "mvpe_face",
            "lve",
            "mouth_closure",
            "lmk2d_reproj_error",
        ]

        # from emica pseudo gt
        gt_metrics = [
            "gt_jitter",
            "gt_mouth_closure",
        ]
        gt_metrics = []
        self.all_metrics = pred_metrics + gt_metrics
    
    def _create_flame(self):
        self.flame = FLAME_mediapipe(self.model_cfg).to(self.device)
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
            ).to(self.device)
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

    def get_vertex_error_heat_color(self, vertex_error, faces=None):
        """
        Args:
            vertex_error: per vertex error [B, V]
        Return:
            face_colors: [B, nf, 3, 3]
        """
        B = vertex_error.shape[0]
        if faces is None:
            faces = self.render.faces.cuda().repeat(B, 1, 1)
        vertex_error = vertex_error.cpu().numpy()
        vertex_color_code = (((vertex_error - self.min_error) / (self.max_error - self.min_error)) * 255.).astype(int)
        verts_rgb = torch.from_numpy(self.colormap(vertex_color_code)[:,:,:3]).to(self.device)    # (B, V, 3)
        face_colors = face_vertices(verts_rgb, faces)
        return face_colors

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
        # # create audio encoder tuned in the state_dict
        # self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # w2v_ckpt = {}
        # for key in state_dict.keys():
        #     if key.startswith('audio_encoder.'):
        #         k = key.replace("audio_encoder.","")
        #         w2v_ckpt[k] = state_dict[key]
        # if len(w2v_ckpt) > 0:
        #     self.audio_encoder.load_state_dict(w2v_ckpt, strict=True)
        #     logger.info(f"Load Audio Encoder Successfully from CKPT!")
        # self.audio_encoder.to(self.device)
        # self.audio_encoder.eval()
    
    def output_video(self, fps=30):
        utils_visualize.images_to_video(self.output_folder, fps, self.motion_name)
    
    def sample_motion_non_overlap(self, batch_split):
        sample_fn = self.diffusion.ddim_sample_loop

        with torch.no_grad():
            bs, n = batch_split['lmk_2d'].shape[:2]
            split_length = n // self.flint_factor
            
            model_kwargs = {}
            for key in self.diffusion_input_keys:
                model_kwargs[key] = batch_split[key].to(self.device)
            model_kwargs['audio_input'] = model_kwargs['audio_input'].reshape(bs, -1)

            if self.config.fix_noise:
                # fix noise seed for every frame
                noise = torch.randn(1, 1, 1).cuda()
                noise = noise.repeat(bs, split_length, self.flint_dim)
            else:
                noise = None
            
            # add CFG scale to batch
            if self.config.guidance_param_audio is not None:
                model_kwargs["y"] = {}
                model_kwargs["y"]['scale_all'] = torch.ones(bs, device=self.device) * self.config.guidance_param_all
                model_kwargs["y"]['scale_audio'] = torch.ones(bs, device=self.device) * self.config.guidance_param_audio
            
            start_time = time()
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
            self.sample_time += time() - start_time
            flint_output = self.diffusion.flint_decoder(output_sample) # (bs, k, c)

        return flint_output
                
    
    def sample_motion(self, motion_split, mem_idx):
        
        sample_fn = self.diffusion.p_sample_loop

        with torch.no_grad():
            split_length = motion_split['lmk_2d'].shape[0] // self.flint_factor
            
            model_kwargs = {}
            for key in self.diffusion_input_keys:
                model_kwargs[key] = motion_split[key].unsqueeze(0).to(self.device)
            model_kwargs['audio_input'] = model_kwargs['audio_input'].reshape(1, -1)

            if self.config.fix_noise:
                # fix noise seed for every frame
                noise = torch.randn(1, 1, 1).cuda()
                noise = noise.repeat(1, split_length, self.flint_dim)
            else:
                noise = None
            
            # add CFG scale to batch
            if self.config.guidance_param_audio is not None:
                model_kwargs["y"] = {}
                model_kwargs["y"]['scale_all'] = torch.ones(1, device=self.device) * self.config.guidance_param_all
                model_kwargs["y"]['scale_audio'] = torch.ones(1, device=self.device) * self.config.guidance_param_audio
                
            # motion inpainting with overlapping frames
            if self.motion_memory is not None:
                if 'y' not in model_kwargs:
                    model_kwargs["y"] = {}
                model_kwargs["y"]["inpainting_mask"] = torch.zeros(
                    (
                        1,
                        split_length,
                        self.flint_dim,
                    )
                ).cuda()
                model_kwargs["y"]["inpainting_mask"][:, :mem_idx, :] = 1
                model_kwargs["y"]["inpainted_motion"] = torch.zeros(
                    (
                        1,
                        split_length,
                        self.flint_dim,
                    )
                ).cuda()
                model_kwargs["y"]["inpainted_motion"][:, :mem_idx, :] = self.motion_memory[
                    :, -mem_idx:, :
                ]
            start_time = time()
            output_sample = sample_fn(
                self.denoise_model,
                (1, split_length, self.flint_dim),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=noise,
                const_noise=False,
            )
            self.sample_time += time() - start_time
            self.memory = output_sample.clone().detach()
            flint_output = self.diffusion.flint_decoder(output_sample)
            flint_output = flint_output[:, mem_idx:].reshape(-1, self.target_nfeat)

        return flint_output
    
    def vis_motion_split(self, gt_data, diffusion_sample):
        diffusion_sample = diffusion_sample.to(self.device)
        # to gpu
        for k in gt_data:
            gt_data[k] = gt_data[k].to(self.device)
        
        # prepare vis data dict 
        diff_expr = diffusion_sample[...,:self.config.n_exp]
        diff_jaw_aa = diffusion_sample[...,self.config.n_exp:]

        gt_jaw_aa = gt_data['jaw']
        gt_exp = gt_data['exp']

        cam = gt_data['cam']

        global_rot_aa = gt_data['global_pose']
        diff_rot_aa = torch.cat([global_rot_aa, diff_jaw_aa], dim=-1)
        gt_rot_aa = torch.cat([global_rot_aa, gt_jaw_aa], dim=-1)
        
        # flame decoder
        emica_verts, _ = self.flame(gt_data['shape'], gt_exp, gt_rot_aa)
        diff_verts, diff_lmk3d = self.flame(gt_data['shape'], diff_expr, diff_rot_aa)
        
        # 2d orthogonal projection
        diff_lmk2d = batch_orth_proj(diff_lmk3d, cam)
        diff_lmk2d[:, :, 1:] = -diff_lmk2d[:, :, 1:]

        diff_trans_verts = batch_orth_proj(diff_verts, cam)
        diff_trans_verts[:, :, 1:] = -diff_trans_verts[:, :, 1:]

        emica_trans_verts = batch_orth_proj(emica_verts, cam)
        emica_trans_verts[:, :, 1:] = -emica_trans_verts[:, :, 1:]
        
        # # render
        diff_render_images = self.render.render_shape(diff_verts, diff_trans_verts, images=gt_data['image'])
        emica_render_images = self.render.render_shape(emica_verts, emica_trans_verts, images=gt_data['image'])
        if self.heatmap_view:
            vertex_error = torch.norm(emica_verts - diff_verts, p=2, dim=-1) * 1000. # vertex dist in mm
            face_error_colors = self.get_vertex_error_heat_color(vertex_error).to(self.device)
            heat_maps = self.render.render_shape(diff_verts, diff_trans_verts,colors=face_error_colors)
        
        # landmarks vis
        lmk2d_vis = utils_visualize.tensor_vis_landmarks(gt_data['image'], diff_lmk2d[...,:2], gt_data['lmk_2d'])
        
        gt_img = gt_data['image'] * gt_data['img_mask'].unsqueeze(1)

        for i in range(diff_lmk2d.shape[0]):
            vis_dict = {
                'gt_img': gt_img[i].detach().cpu(),   # (3, h, w)
                'gt_mesh': emica_render_images[i].detach().cpu(),  # (3, h, w)
                'diff_mesh': diff_render_images[i].detach().cpu(),  # (3, h, w)
                'lmk': lmk2d_vis[i].detach().cpu()
            }
            if self.heatmap_view:
                vis_dict['heatmap'] = heat_maps[i].detach().cpu()
            grid_image = self.visualize(vis_dict)
            if self.to_mp4:
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
        grid_image = cv2.resize(grid_image, (self.view_w,self.view_h))
        return grid_image
    
    def evaluate_one_motion(
        self,
        diffusion_output,
        batch, 
    ):      
        global_rot_aa = batch['global_pose']
        diff_expr = diffusion_output[...,:self.config.n_exp]
        diff_jaw_aa = diffusion_output[...,self.config.n_exp:]
        diff_rot_aa = torch.cat([global_rot_aa, diff_jaw_aa], dim=-1)

        gt_jaw_aa = batch['jaw']
        gt_expr = batch['exp']
        gt_rot_aa = torch.cat([global_rot_aa, gt_jaw_aa], dim=-1)
        
        # flame decoder
        verts_gt, lmk_3d_gt = self.flame(batch['shape'], gt_expr, gt_rot_aa)
        
        # flame decoder
        verts_pred, lmk_3d_pred = self.flame(batch['shape'], diff_expr, diff_rot_aa)

        # 2d orthogonal projection
        lmk_2d_pred = batch_orth_proj(lmk_3d_pred, batch['cam'])[...,:2]
        lmk_2d_pred[:, :, 1:] = -lmk_2d_pred[:, :, 1:]

        # 2d orthogonal projection
        lmk_2d_emica = batch_orth_proj(lmk_3d_gt, batch['cam'])[...,:2]
        lmk_2d_emica[:, :, 1:] = -lmk_2d_emica[:, :, 1:]

        lmk_2d_gt = batch['lmk_2d'][:,self.flame.landmark_indices_mediapipe]

        eval_log = {}
        for metric in self.all_metrics:
            eval_log[metric] = (
                get_metric_function(metric)(
                    diff_expr, diff_jaw_aa, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_emica,
                    gt_expr, gt_jaw_aa, verts_gt, lmk_3d_gt, lmk_2d_gt,
                    self.config.fps, self.flame_v_mask 
                )
                .numpy()
            )
        
        return eval_log

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

    def overlapping_inference(self, batch):
        self.motion_memory = None   # init diffusion memory for motin infilling
        start_id = 0
        diffusion_output = []          

        while start_id == 0 or start_id + self.input_motion_length <= self.num_frames:
            motion_split, flag_index = self.prepare_chunk_diffusion_input(batch, start_id, self.num_frames)
            if start_id == 0:
                mem_idx = 0
            else:
                mem_idx = self.input_motion_length - self.sld_wind_size
            # print(f"Processing frame from No.{start_id} at mem {mem_idx}")
            output_sample = self.sample_motion(motion_split, mem_idx)
            if flag_index > 0:
                output_sample = output_sample[flag_index:]
            start_id += self.sld_wind_size
            diffusion_output.append(output_sample.cpu())
        
        if start_id < self.num_frames:
            last_start_id = self.num_frames-self.input_motion_length
            motion_split, flag_index = self.prepare_chunk_diffusion_input(batch, last_start_id, self.num_frames)
            mem_idx = self.input_motion_length - (
                self.num_frames - (start_id - self.sld_wind_size + self.input_motion_length))
            output_sample = self.sample_motion(motion_split, mem_idx)
            start_id = last_start_id
            # print(f"Processing last frame No.{start_id} at mem {mem_idx}")
            diffusion_output.append(output_sample.cpu())
        logger.info(f'DDPM sample {self.num_frames} frames used: {self.sample_time} seconds.')

        diffusion_output = torch.cat(diffusion_output, dim=0)

        assert batch['lmk_2d'].shape[0] == diffusion_output.shape[0]
        return diffusion_output

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

        sample = self.sample_motion_non_overlap(batch_split).cpu()

        final_output = []
        for i in range(sample.shape[0]):
            final_output.append(sample[i, flag_index[i]:])
        final_output = torch.cat(final_output, dim=0)

        assert final_output.shape[0] == self.num_frames
        return final_output 

            
    def track(self):
        
        # make prediction by split the whole sequences into chunks
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        eval_all = {}
        for metric in self.all_metrics:
            eval_all[metric] = 0.0
        num_test_motions = len(self.test_data)
        eval_motion_num = 0
        for i in tqdm(range(num_test_motions)):
            self.flame.to(self.device)
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

            save_path = f"{self.sample_folder}/{motion_id}.npy"
            if os.path.exists(save_path):
                diffusion_output = np.load(save_path, allow_pickle=True)[()]
                diffusion_output = torch.from_numpy(diffusion_output)
            else:
                if self.config.overlap:
                    diffusion_output = self.overlapping_inference(batch)
                else:
                    diffusion_output = self.non_overlapping_inference(batch)
            
            # visualize the output
            if self.vis:
                video_path = self.output_folder + f'/{motion_id}'
                if self.to_mp4:
                    video_fname = video_path + '.mp4'
                    Path(video_fname).parent.mkdir(exist_ok=True, parents=True)

                    self.writer = cv2.VideoWriter(
                        video_fname, fourcc, self.config.fps, 
                        (self.view_w, self.view_h))
                else:
                    gif_fname = video_path + '.gif'
                    Path(gif_fname).parent.mkdir(exist_ok=True, parents=True)
                    self.writer = imageio.get_writer(gif_fname, mode='I')

                # batch visualiza all frames
                print(diffusion_output.shape)
                for i in range(0, self.num_frames, self.visualization_batch):
                    batch_sample = diffusion_output[i:i+self.visualization_batch]
                    gt_data = {}
                    for key in batch:
                        if key != 'audio_input':
                            gt_data[key] = batch[key][i:i+self.visualization_batch]
                    self.vis_motion_split(gt_data, batch_sample)

                # concat audio 
                if self.to_mp4:
                    self.writer.release()
                    subject, view, emotion, level, sent = motion_id.split('/')
                    audio_path = os.path.join(self.original_data_folder, subject, 'audio', emotion, level, f"{sent}.m4a")
                    assert os.path.exists(audio_path)
                    os.system(f"ffmpeg -i {video_path}.mp4 -i {audio_path} -c:v copy -c:a copy {video_path}_audio.mp4")
                    os.system(f"rm {video_path}.mp4")
            
            # start evaluation
            self.flame.to('cpu')
            diffusion_output.to('cpu')
            eval_log = self.evaluate_one_motion(diffusion_output, batch)
                 
            for metric in eval_log:
                logger.info(f"{metric} : {eval_log[metric]}")
                eval_all[metric] += eval_log[metric]

            if self.save_rec and not os.path.exists(save_path):
                # save inference results
                save_path = f"{self.sample_folder}/{motion_id}.npy"
                Path(save_path).parent.mkdir(exist_ok=True, parents=True)
                np.save(save_path, diffusion_output.numpy())

                # save the occlusion mask
                if self.config.exp_name not in ['non_occ', 'all']:
                    np.save(f"{self.sample_folder}/{motion_id}_mask.npy", batch['img_mask'].cpu().numpy())
            torch.cuda.empty_cache()

        logger.info("==========Metrics for all test motion sequences:===========")
        for metric in eval_all:
            logger.info(f"{metric} : {eval_all[metric] / eval_motion_num}")

def main():
    args = test_args()

    args.timestep_respacing = '100' # use DDIM samper
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
            test_video_list,
            args.fps,
            args.n_shape,
            args.n_exp,
            args.exp_name,
            args.load_tex,
            args.use_iris,
            load_audio_input=True,
            vis=args.vis,
            use_segmask=True
        )
    elif args.test_dataset == "RAVDESS":
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
            mask_path=None
        )
    else:
        raise ValueError(f"{args.test_dataset} not supported!")

    motion_tracker = MotionTracker(args, pretrained_args.model, test_dataset, 'cuda')
    
    motion_tracker.track()

if __name__ == "__main__":
    main()
