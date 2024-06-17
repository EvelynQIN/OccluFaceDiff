""" Test motion recontruction with with landmark and audio as input.
"""
import os
import numpy as np
from tqdm import tqdm
import cv2
import os.path
from glob import glob
from pathlib import Path
from loguru import logger
from time import time
from copy import deepcopy
from collections import defaultdict

from utils import utils_visualize
from utils.model_util import create_model_and_diffusion
from utils.parser_util import predict_args
from model.FLAME import FLAME_mediapipe
from diffusion.cfg_sampler import ClassifierFreeSampleModel
from configs.config import get_cfg_defaults
from utils.renderer import SRenderY
from utils.data_util import batch_orth_proj
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch3d.io import load_obj
# pretrained

import imageio
from skimage.io import imread
import pickle
from prepare_video import VideoProcessor

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class MotionTracker:
    
    def __init__(self, config, model_cfg, test_data, device='cuda'):
        
        self.config = config
        model_cfg.n_shape = 100 # the gt shape is from EMOCA's prediction
        self.model_cfg = model_cfg
        self.device = device
        self.sld_wind_size = config.sld_wind_size
        self.input_motion_length = config.input_motion_length
        self.target_nfeat = config.n_exp + config.n_pose
        self.config.n_shape = 100
        self.flint_dim = 128
        self.flint_factor = 8
        self.test_data = test_data
        # IO setups
                        
        # name of the tested motion sequence
        self.save_folder = self.config.save_folder
        self.output_folder = os.path.join(self.save_folder, self.config.arch)
        
        logger.add(os.path.join(self.output_folder, 'predict.log'))
        logger.info(f"Using device {self.device}.")

        # vis settings
        self.to_mp4 = config.to_mp4 # if true then to mp4, false then to gif wo audio
        self.visualization_batch = 10
        self.image_size = config.image_size
        self.resize_factor=1.0  # resize the final grid image
        
        if config.occlusion_type in ['non_occ', 'audio_driven']:
            self.n_views = 3
        else:
            self.n_views = 4
        self.view_h, self.view_w = int(self.image_size*self.resize_factor), int(self.image_size*self.n_views*self.resize_factor)
        self.sample_time = 0

        # diffusion models
        self.load_diffusion_model_from_ckpt()
        
        self._create_flame()
        self._setup_renderer()
    
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
        mean_texture = imread(self.model_cfg.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)

    def load_diffusion_model_from_ckpt(self):
        logger.info("Creating model and diffusion...")
        self.denoise_model, self.diffusion = create_model_and_diffusion(self.config, self.model_cfg, self.device)
        model_path = self.config.model_path
        logger.info(f"Loading checkpoints from [{model_path}]...")
        state_dict = torch.load(model_path, map_location="cpu")
        self.denoise_model.load_state_dict(state_dict, strict=True)

        # wrap the model with cfg sampler
        if self.config.guidance_param_audio is not None:
            self.denoise_model = ClassifierFreeSampleModel(self.denoise_model)

        self.denoise_model.to(self.device) 
        self.denoise_model.eval()  

        self.diffusion_input_keys = ['lmk_2d', 'lmk_mask', 'audio_input']
    
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
                
    
    def sample_motion_overlap(self, motion_split, mem_idx):
        
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
            output_sample = self.sample_motion_overlap(motion_split, mem_idx)
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

    
    def vis_motion_split(self, gt_data, diffusion_sample):
        diffusion_sample = diffusion_sample.to(self.device)
        # to gpu
        for k in gt_data:
            gt_data[k] = gt_data[k].to(self.device)
        
        # prepare vis data dict 
        diff_jaw_aa = diffusion_sample[...,self.config.n_exp:]
        diff_expr = diffusion_sample[...,:self.config.n_exp]
        cam = gt_data['cam']
        global_rot_aa = gt_data['global_pose']
        diff_rot_aa = torch.cat([global_rot_aa, diff_jaw_aa], dim=-1)
        
        # flame decoder
        diff_verts, diff_lmk3d = self.flame(gt_data['shape'], diff_expr, diff_rot_aa)
        
        # 2d orthogonal projection
        diff_lmk2d = batch_orth_proj(diff_lmk3d, cam)
        diff_lmk2d[:, :, 1:] = -diff_lmk2d[:, :, 1:]

        diff_trans_verts = batch_orth_proj(diff_verts, cam)
        diff_trans_verts[:, :, 1:] = -diff_trans_verts[:, :, 1:]
        
        # # render
        diff_render_images = self.render.render_shape(diff_verts, diff_trans_verts, images=gt_data['original_image'])
        
        # landmarks vis
        
        lmk2d_vis = utils_visualize.tensor_vis_landmarks(gt_data['original_image'], diff_lmk2d[...,:2], gt_data['lmk_2d'])

        for i in range(diff_lmk2d.shape[0]):
            if self.config.occlusion_type in ['non_occ', 'audio_driven']:
                vis_dict = {
                    'gt_img': gt_data['original_image'][i].cpu(),   # (3, h, w)
                    'diff_mesh': diff_render_images[i].detach().cpu(),  # (3, h, w)
                    'lmk': lmk2d_vis[i].detach().cpu()
                }
            else:
                vis_dict = {
                    'gt_img': gt_data['original_image'][i].cpu(),   # (3, h, w)
                    'occ_img': gt_data['image'][i].cpu(),   # (3, h, w)
                    'diff_mesh': diff_render_images[i].detach().cpu(),  # (3, h, w)
                    'lmk': lmk2d_vis[i].detach().cpu()
                }
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
        grid = torch.cat(list(visdict.values()), dim)
        grid_image = (grid.numpy().transpose(1,2,0).copy()*255)
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        grid_image = cv2.resize(grid_image, (self.view_w,self.view_h))
        return grid_image
            
    def track(self):
        
        # make prediction by split the whole sequences into chunks
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.flame.to(self.device)
        batch, motion_id, audio_path = self.test_data
        logger.info(f'Process [{motion_id}].')
        video_path = self.output_folder + f'/{motion_id}_{self.config.occlusion_type}'
        
        # set the output writer
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
        
        self.num_frames = batch['lmk_2d'].shape[0]
        
        # run reconstruction
        if self.config.overlap:
            diffusion_output = self.overlapping_inference(batch)
        else:
            diffusion_output = self.non_overlapping_inference(batch)
        

        # batch visualiza all frames
        for i in range(0, self.num_frames, self.visualization_batch):
            batch_sample = diffusion_output[i:i+self.visualization_batch]
            gt_data = {}
            for key in batch:
                if key != 'audio_input':
                    gt_data[key] = batch[key][i:i+self.visualization_batch]
            self.vis_motion_split(gt_data, batch_sample)

        # concat audio to output video 
        if self.to_mp4:
            self.writer.release()
            assert os.path.exists(audio_path)
            os.system(f"ffmpeg -i {video_path}.mp4 -i {audio_path} -c:v copy {video_path}_audio.mp4")
            os.system(f"rm {video_path}.mp4")

def main():
    args = predict_args()
    args.timestep_respacing = '100' # use DDIM samper
    pretrained_args = get_cfg_defaults()
    
    # load processed data of the test video
    video_processor = VideoProcessor(pretrained_args.emoca, args)

    video_processor.preprocess_video()
    test_data = video_processor.get_processed_data()

    motion_tracker = MotionTracker(args, pretrained_args.model, test_data, 'cuda')
    
    motion_tracker.track()

if __name__ == "__main__":
    main()
