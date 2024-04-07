
import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F_v

from data_loaders.dataloader import load_data, TestDataset
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
from model.deca import ResnetEncoder, ExpressionLossNet
from model.FLAME import FLAME, FLAMETex
from utils.renderer import SRenderY
from skimage.io import imread
import imageio

import sys
sys.path.append("external/Visual_Speech_Recognition_for_Multiple_Languages")
from external.Visual_Speech_Recognition_for_Multiple_Languages.lipreading.model import Lipreading
from external.Visual_Speech_Recognition_for_Multiple_Languages.dataloader.transform import Compose, Normalize, CenterCrop, SpeedRate, Identity
from configparser import ConfigParser


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

pred_metrics = [
    "pred_jitter",
    "mvpe",
    "mvve",
    "shape_error",
    "expre_error",
    "pose_error",
    "lmk_3d_mvpe",
    "lmk_2d_mpe",
    "mvpe_face",
    "mvpe_eye_region",
    
    "mvpe_forehead",
    "mvpe_lips",
    "mvpe_neck",
    "mvpe_nose",
]
gt_metrics = [
    "gt_jitter",
]

all_metrics = pred_metrics + gt_metrics

class MotionTracker:
    
    def __init__(self, config, model_cfg, device='cuda'):
        
        self.config = config
        self.model_cfg = model_cfg
        self.device = device
        self.sld_wind_size = config.sld_wind_size
        self.input_motion_length = config.input_motion_length
        self.target_nfeat = config.n_exp + config.n_pose
        # IO setups
        self.motion_id = f'{config.subject_id}_{config.motion_id}' if config.test_mode != 'in_the_wild' \
                        else os.path.split(config.video_path)[-1].split('.')[0]
                        
        # name of the tested motion sequence
        self.save_folder = self.config.save_folder
        self.output_folder = os.path.join(self.save_folder, self.config.arch)
        self.motion_name = f'{self.motion_id}_{self.config.exp_name}'
        
        logger.add(os.path.join(self.output_folder, 'predict.log'))
        logger.info(f"Using device {self.device}.")
        logger.info(f"Predict motion [{self.motion_id}] for Exp [{self.config.exp_name}].")
        
        self.image_size = torch.tensor([[config.image_size, config.image_size]]).to(self.device)
        
        self.sample_time = 0

        # visualization settings
        self.colormap = cm.get_cmap('jet')
        self.min_error = 0.
        self.max_error = 30.

        # diffusion models
        self.denoise_model, self.diffusion = self.load_diffusion_model_from_ckpt(config, model_cfg)
        
        # load relavant models
        self._load_deca_model(model_cfg)
        self._create_flame()
        self._setup_renderer()
        
        # gif_writer
        savefolder = os.path.join(self.output_folder, self.motion_name)
        Path(savefolder).mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(os.path.join(savefolder, 'motion.gif'), mode='I')


    def _load_deca_model(self, model_cfg):
         # set up parameters
        self.n_param = model_cfg.n_shape + model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light
        self.n_cond = model_cfg.n_exp + 3  # exp + jaw pose
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam, model_cfg.n_light]
        self.param_dict = {i: model_cfg.get('n_' + i) for i in model_cfg.param_list}

        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)
            
        # resume model from ckpt path
        model_path = model_cfg.ckpt_path
        if os.path.exists(model_path):
            logger.info(f"[DECA] Pretrained model found at {model_path}.")
            checkpoint = torch.load(model_path)

            if 'state_dict' in checkpoint.keys():
                checkpoint = checkpoint['state_dict']
            else:
                checkpoint = checkpoint
             
            if 'deca' in list(checkpoint.keys())[0]:
                for key in checkpoint.keys():
                    k = key.replace("deca.","")
                    if "E_flame" in key:
                        processed_checkpoint["E_flame"][k.replace("E_flame.","")] = checkpoint[key]
                    else:
                        pass
            else:
                processed_checkpoint = checkpoint
            self.E_flame.load_state_dict(processed_checkpoint['E_flame'], strict=True) 
        else:
            raise(f'please check model path: {model_path}')

        # eval mode to freeze deca throughout the process
        self.E_flame.eval()
        self.E_flame.requires_grad_(False)

    def decompose_deca_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0

        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[..., start:end]
            start = end
            if key == 'light':
                dims_ = code_dict[key].ndim -1 # (to be able to handle batches of videos)
                code_dict[key] = code_dict[key].reshape(*code_dict[key].shape[:dims_], 9, 3)
        return code_dict

    def deca_encode(self, images):
        with torch.no_grad():
            parameters = self.E_flame(images)
        codedict = self.decompose_deca_code(parameters, self.param_dict)
        return codedict
    
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
    
    def get_image_size(self):
        return self.image_size[0][0].item(), self.image_size[0][1].item()
    
    def output_video(self, fps=30):
        utils_visualize.images_to_video(self.output_folder, fps, self.motion_name)
    
    def sample_motion(self, motion_split, mem_idx):
        
        # sample_fn = self.diffusion.p_sample_loop
        
        images = motion_split['cropped_imgs'].to(self.device)
        
        # get deca reconstruction result
        deca_code = self.deca_encode(images[mem_idx:])
        deca_code['images'] = images[mem_idx:]
        deca_code['lmk_gt'] = motion_split['lmk_2d'][mem_idx:].to(self.device)

        # with torch.no_grad():
        #     split_length = images.shape[0]

        #     model_kwargs = {
        #         "image": images.unsqueeze(0),
        #     }

        #     if self.config.fix_noise:
            #     # fix noise seed for every frame
            #     noise = torch.randn(1, 1, 1).cuda()
            #     noise = noise.repeat(1, split_length, self.target_nfeat)
            # else:
            #     noise = None
                
            # # motion inpainting with overlapping frames
            # if self.motion_memory is not None:
            #     model_kwargs["y"] = {}
            #     model_kwargs["y"]["inpainting_mask"] = torch.zeros(
            #         (
            #             1,
            #             self.input_motion_length,
            #             self.target_nfeat,
            #         )
            #     ).cuda()
            #     model_kwargs["y"]["inpainting_mask"][:, :mem_idx, :] = 1
            #     model_kwargs["y"]["inpainted_motion"] = torch.zeros(
            #         (
            #             1,
            #             self.input_motion_length,
            #             self.target_nfeat,
            #         )
            #     ).cuda()
            #     model_kwargs["y"]["inpainted_motion"][:, :mem_idx, :] = self.motion_memory[
            #         :, -mem_idx:, :
            #     ]
            # start_time = time()
            # output_sample = sample_fn(
            #     self.denoise_model,
            #     (1, split_length, self.target_nfeat),
            #     clip_denoised=False,
            #     model_kwargs=model_kwargs,
            #     skip_timesteps=0,
            #     init_image=None,
            #     progress=False,
            #     dump_steps=None,
            #     noise=noise,
            #     const_noise=False,
            # )
            # self.sample_time += time() - start_time
            # self.memory = output_sample.clone().detach()
            # output_sample = output_sample[:, mem_idx:].reshape(-1, self.target_nfeat).float()
        output_sample = None
        return output_sample, deca_code # (n, *shape)
    
    def vis_motion_split(self, deca_code, diffusion_sample):
        
        # prepare vis data dict 
        # diff_jaw = diffusion_sample[...,:self.config.n_pose]
        # diff_expr = diffusion_sample[...,self.config.n_pose:]
        # diff_jaw_aa = utils_transform.sixd2aa(diff_jaw)
        # deca_R = deca_code['pose'][...,:3]
        # diff_pose = torch.cat([deca_R, diff_jaw_aa], dim=-1) # (n, 6)
        
        # flame decoder
        deca_verts, deca_lmk2d, _ = self.flame(
            shape_params=deca_code['shape'], 
            expression_params=deca_code['exp'],
            pose_params=deca_code['pose'])
        
        # diff_verts, diff_lmk2d, _ = self.flame(
        #     shape_params=deca_code['shape'], 
        #     expression_params=diff_expr,
        #     pose_params=diff_pose)
        
        # orthogonal projection
        deca_lmk2d = batch_orth_proj(deca_lmk2d, deca_code['cam'])[:, :, :2]
        deca_lmk2d[:, :, 1:] = -deca_lmk2d[:, :, 1:]
        deca_trans_verts = batch_orth_proj(deca_verts, deca_code['cam'])
        deca_trans_verts[:, :, 1:] = -deca_trans_verts[:, :, 1:]
        
        # diff_lmk2d = batch_orth_proj(diff_lmk2d, deca_code['cam'])[:, :, :2]
        # diff_lmk2d[:, :, 1:] = -diff_lmk2d[:, :, 1:]
        # diff_trans_verts = batch_orth_proj(diff_verts, deca_code['cam'])
        # diff_trans_verts[:, :, 1:] = -diff_trans_verts[:, :, 1:]
        
        albedo = self.flametex(deca_code['tex']).detach()
        
        # # render
        deca_render_images = self.render(deca_verts, deca_trans_verts, albedo, deca_code['light'], background=deca_code['images'])['images']
        # diff_render_images = self.render(diff_verts, diff_trans_verts, albedo, deca_code['light'], background=deca_code['images'])['images']
        
        # landmarks vis
        lmk2d_vis = utils_visualize.tensor_vis_landmarks(deca_code['images'], deca_lmk2d, deca_code['lmk_gt'])

        # frame_id = str(frame_id).zfill(5)
        for i in range(deca_lmk2d.shape[0]):
            vis_dict = {
                'gt_img': deca_code['images'][i].detach().cpu(),   # (3, 224, 224)
                'deca_img': deca_render_images[i].detach().cpu(),  # (3, 224, 224)
                # 'diff_img': diff_render_images[i],  # (3, 224, 224)
                'lmk': lmk2d_vis[i].detach().cpu()
            }
            grid_image = self.visualize(vis_dict)
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
            motion_split = video_processor.prepare_chunk_motion(start_id)
            if start_id == 0:
                mem_idx = 0
            else:
                mem_idx = self.input_motion_length - self.sld_wind_size
            start_id += self.sld_wind_size
            output_sample, deca_code = self.sample_motion(motion_split, mem_idx)
            self.vis_motion_split(deca_code, output_sample)
        
        if start_id < self.num_frames:
            motion_split = video_processor.prepare_chunk_motion(self.num_frames-self.input_motion_length)
            mem_idx = self.input_motion_length - (
                self.num_frames - (start_id - self.sld_wind_size + self.input_motion_length))
            output_sample, deca_code = self.sample_motion(motion_split, mem_idx)
            self.vis_motion_split(deca_code, output_sample)
        logger.info(f'DDPM sample {self.num_frames} frames used: {self.sample_time} seconds.')
            

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
