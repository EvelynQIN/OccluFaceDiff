
import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F_v

from tqdm import tqdm

from utils import utils_transform, utils_visualize
from utils.metrics import get_metric_function
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import predict_args
from utils.data_util import batch_orth_proj
from configs.config import get_cfg_defaults
from prepare_video import  VideoProcessor
import os.path
from enum import Enum
from glob import glob
from pathlib import Path

import cv2
import torch.nn.functional as F
from loguru import logger
from pytorch3d.io import load_obj
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from pytorch3d.utils import opencv_from_cameras_projection
from tqdm import tqdm
from time import time
from matplotlib import cm
# pretrained
from model.deca import EMOCA
from model.FLAME import FLAME, FLAMETex
from utils.renderer import SRenderY
from skimage.io import imread
import imageio


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class MotionTracker:
    
    def __init__(self, config, model_cfg, device='cuda'):
        
        self.config = config
        self.model_cfg = model_cfg
        self.device = device
        self.sld_wind_size = config.sld_wind_size
        self.input_motion_length = config.input_motion_length
        self.target_nfeat = config.n_exp + config.n_pose
        # IO setups
        self.motion_id = os.path.split(config.video_path)[-1].split('.')[0]
                        
        # name of the tested motion sequence
        self.save_folder = self.config.save_folder
        self.output_folder = os.path.join(self.save_folder, self.config.arch, 'in-the-wild', self.config.exp_name)
        
        logger.add(os.path.join(self.output_folder, 'predict.log'))
        logger.info(f"Using device {self.device}.")
        logger.info(f"Predict motion [{self.motion_id}] for Exp [{self.config.exp_name}].")
        
        self.image_size = config.image_size
        
        self.sample_time = 0

        # visualization settings
        self.colormap = cm.get_cmap('jet')
        self.min_error = 0.
        self.max_error = 30.

        # diffusion models
        self.denoise_model, self.diffusion = self.load_diffusion_model_from_ckpt(config, model_cfg)
        
        # load relavant models
        self.emoca = EMOCA(model_cfg)
        self.emoca.to(self.device)
        self._create_flame()
        self._setup_renderer()
        
        # writer
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        if self.config.with_audio:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_path = os.path.join(self.output_folder, f'{self.motion_id}.mp4')
            self.writer = cv2.VideoWriter(self.video_path, fourcc, self.config.fps, (self.image_size*4, self.image_size))
        else:
            self.writer = imageio.get_writer(os.path.join(self.output_folder, f'{self.motion_id}.gif'), mode='I')

    def _create_flame(self):
        self.flame = FLAME(self.model_cfg).to(self.device)
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
        denoise_model, diffusion = create_model_and_diffusion(args, model_cfg, self.device)

        logger.info(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location="cpu")
        load_model_wo_clip(denoise_model, state_dict)

        denoise_model.to(self.device)  # dist_util.dev())
        denoise_model.eval()  # disable random masking
        return denoise_model, diffusion
    
    def sample_motion(self, motion_split, mem_idx):
        
        sample_fn = self.diffusion.p_sample_loop

        images = motion_split['image'].to(self.device)
        
        # get deca reconstruction result
        deca_code = self.emoca(images[mem_idx:])
        deca_code['images'] = images[mem_idx:]
        deca_code['lmk_gt'] = motion_split['lmk_2d'][mem_idx:].to(self.device)
        deca_code['img_mask'] = motion_split['img_mask'][mem_idx:]

        with torch.no_grad():
            split_length = images.shape[0]

            model_kwargs = {
                "image": images.unsqueeze(0),
                'audio_emb': motion_split['audio_emb'].unsqueeze(0).to(self.device),
                'img_mask': motion_split['img_mask'].unsqueeze(0).to(self.device),
                'lmk_mask':motion_split['lmk_mask'].unsqueeze(0).to(self.device),
                'lmk_2d': motion_split['lmk_2d'].unsqueeze(0).to(self.device)
            }

            if self.config.fix_noise:
                # fix noise seed for every frame
                noise = torch.randn(1, 1, 1).cuda()
                noise = noise.repeat(1, split_length, self.target_nfeat)
            else:
                noise = None
                
            # motion inpainting with overlapping frames
            if self.motion_memory is not None:
                model_kwargs["y"] = {}
                model_kwargs["y"]["inpainting_mask"] = torch.zeros(
                    (
                        1,
                        self.input_motion_length,
                        self.target_nfeat,
                    )
                ).cuda()
                model_kwargs["y"]["inpainting_mask"][:, :mem_idx, :] = 1
                model_kwargs["y"]["inpainted_motion"] = torch.zeros(
                    (
                        1,
                        self.input_motion_length,
                        self.target_nfeat,
                    )
                ).cuda()
                model_kwargs["y"]["inpainted_motion"][:, :mem_idx, :] = self.motion_memory[
                    :, -mem_idx:, :
                ]
            start_time = time()
            output_sample = sample_fn(
                self.denoise_model,
                (1, split_length, self.target_nfeat),
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
            output_sample = output_sample[:, mem_idx:].reshape(-1, self.target_nfeat).float()

        return output_sample, deca_code # (n, *shape)
    
    def vis_motion_split(self, deca_code, diffusion_sample):
        
        # prepare vis data dict 
        diff_jaw = diffusion_sample[...,:self.config.n_pose]
        diff_expr = diffusion_sample[...,self.config.n_pose:]
        diff_jaw_aa = utils_transform.sixd2aa(diff_jaw)
        deca_R = deca_code['pose'][...,:3]
        diff_pose = torch.cat([deca_R, diff_jaw_aa], dim=-1) # (n, 6)
        
        # flame decoder
        deca_verts, _, _ = self.flame(
            shape_params=deca_code['shape'], 
            expression_params=deca_code['exp'],
            pose_params=deca_code['pose'])
        
        diff_verts, diff_lmk2d, _ = self.flame(
            shape_params=deca_code['shape'], 
            expression_params=diff_expr,
            pose_params=diff_pose)
        
        # orthogonal projection
        # deca_lmk2d = batch_orth_proj(deca_lmk2d, deca_code['cam'])[:, :, :2]
        # deca_lmk2d[:, :, 1:] = -deca_lmk2d[:, :, 1:]
        deca_trans_verts = batch_orth_proj(deca_verts, deca_code['cam'])
        deca_trans_verts[:, :, 1:] = -deca_trans_verts[:, :, 1:]
        
        diff_lmk2d = batch_orth_proj(diff_lmk2d, deca_code['cam'])[:, :, :2]
        diff_lmk2d[:, :, 1:] = -diff_lmk2d[:, :, 1:]
        diff_trans_verts = batch_orth_proj(diff_verts, deca_code['cam'])
        diff_trans_verts[:, :, 1:] = -diff_trans_verts[:, :, 1:]
        
        # albedo = self.flametex(deca_code['tex']).detach()
        
        # # render
        deca_render_images = self.render.render_shape(deca_verts, deca_trans_verts, images=deca_code['images'])
        diff_render_images = self.render.render_shape(diff_verts, diff_trans_verts, images=deca_code['images'])
        # diff_render_images = self.render(diff_verts, diff_trans_verts, albedo, deca_code['light'], background=deca_code['images'])['images']
        
        # landmarks vis
        lmk2d_vis = utils_visualize.tensor_vis_landmarks(deca_code['images'], diff_lmk2d, deca_code['lmk_gt'])

        # frame_id = str(frame_id).zfill(5)
        for i in range(diff_lmk2d.shape[0]):
            vis_dict = {
                'gt_img': deca_code['images'][i].detach().cpu() * deca_code['img_mask'][i].unsqueeze(0),   # (3, 224, 224)
                'deca_img': deca_render_images[i].detach().cpu(),  # (3, 224, 224)
                'diff_img': diff_render_images[i].detach().cpu(),  # (3, 224, 224)
                'lmk': lmk2d_vis[i].detach().cpu()
            }
            grid_image = self.visualize(vis_dict)
            
            if self.config.with_audio:
                self.writer.write(grid_image[:,:,[2,1,0]])
            else:
                self.writer.append_data(grid_image)
        
        # cv2.imwrite(f'{savefolder}/{frame_id}.jpg', final_views)
            
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
        return grid_image
    
    def track(self):
        
        # load the input motion sequence
        video_processor = VideoProcessor(self.config)
        video_processor.preprocess_video()
        
        # make prediction by split the whole sequences into chunks due to memory limit
        self.num_frames = video_processor.num_frames
        self.motion_memory = None   # init diffusion memory for motin infilling
        start_id = 0
        while start_id + self.input_motion_length <= self.num_frames:
            print(f"Processing frame {start_id}")
            motion_split = video_processor.prepare_chunk_motion(start_id)
            if start_id == 0:
                mem_idx = 0
            else:
                mem_idx = self.input_motion_length - self.sld_wind_size
            start_id += self.sld_wind_size
            output_sample, deca_code = self.sample_motion(motion_split, mem_idx)
            self.vis_motion_split(deca_code, output_sample)
        
        if start_id < self.num_frames:
            print(f"Processing frame {self.num_frames-self.input_motion_length}")
            motion_split = video_processor.prepare_chunk_motion(self.num_frames-self.input_motion_length)
            mem_idx = self.input_motion_length - (
                self.num_frames - (start_id - self.sld_wind_size + self.input_motion_length))
            output_sample, deca_code = self.sample_motion(motion_split, mem_idx)
            self.vis_motion_split(deca_code, output_sample)
        logger.info(f'DDPM sample {self.num_frames} frames used: {self.sample_time} seconds.')
        
        # concat audio 
        if self.config.with_audio:
            self.writer.release()
            out_video_path = os.path.join(self.output_folder, f"{self.motion_id}_audio.mp4")
            os.system(f"ffmpeg -i {self.video_path} -i {video_processor.audio_path} -c:v copy {out_video_path}")
            os.system(f"rm {self.video_path}")
        
            

def main():
    args = predict_args()
    pretrained_args = get_cfg_defaults()
    # to guanrantee reproducibility
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    motion_tracker = MotionTracker(args, pretrained_args.model, 'cuda')
    
    motion_tracker.track()

if __name__ == "__main__":
    main()
