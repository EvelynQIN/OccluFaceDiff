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
from data_loaders.dataloader_MEAD import load_test_data, TestMeadDataset
import ffmpeg
import pickle
from model.wav2vec import Wav2Vec2Model

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class MotionTracker:
    
    def __init__(self, config, model_cfg, test_data, device='cuda'):
        
        self.config = config
        self.model_cfg = model_cfg
        self.device = device
        self.sld_wind_size = config.sld_wind_size
        self.input_motion_length = config.input_motion_length
        self.target_nfeat = config.n_exp + config.n_pose
        self.test_data = test_data
        self.original_data_folder = 'dataset/mead_25fps/original_data'
        self.vis = config.vis
        # IO setups
                        
        # name of the tested motion sequence
        self.save_folder = self.config.save_folder
        self.output_folder = os.path.join(self.save_folder, self.config.arch, config.exp_name)
        
        logger.add(os.path.join(self.output_folder, 'test_mead.log'))
        logger.info(f"Using device {self.device}.")

        # vis settings
        self.to_mp4 = True # if true then to mp4, false then to gif wo audio
        self.visualization_batch = 10
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
        ]

        # from emica pseudo gt
        gt_metrics = [
            "gt_jitter",
            "gt_mouth_closure",
        ]

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
        self.denoise_model.load_state_dict(state_dict, strict=False)

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
    
    def sample_motion(self, motion_split, mem_idx):
        
        sample_fn = self.diffusion.p_sample_loop

        with torch.no_grad():
            split_length = motion_split['lmk_2d'].shape[0]
            
            model_kwargs = {}
            for key in self.diffusion_input_keys:
                model_kwargs[key] = motion_split[key].unsqueeze(0).to(self.device)
            model_kwargs['audio_input'] = model_kwargs['audio_input'].reshape(1, -1)

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

        return output_sample
    
    def vis_motion_split(self, gt_data, diffusion_sample):
        diffusion_sample = diffusion_sample.to(self.device)
        # to gpu
        for k in gt_data:
            gt_data[k] = gt_data[k].to(self.device)
        
        # prepare vis data dict 
        diff_jaw = diffusion_sample[...,:self.config.n_pose]
        diff_expr = diffusion_sample[...,self.config.n_pose:]
        diff_jaw_aa = utils_transform.sixd2aa(diff_jaw)

        gt_jaw = gt_data['target'][...,:self.config.n_pose]
        gt_exp = gt_data['target'][...,self.config.n_pose:]
        gt_jaw_aa = utils_transform.sixd2aa(gt_jaw)

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
        diff_jaw = diffusion_output[...,:self.config.n_pose]
        diff_expr = diffusion_output[...,self.config.n_pose:]
        diff_jaw_aa = utils_transform.sixd2aa(diff_jaw)
        diff_rot_aa = torch.cat([global_rot_aa, diff_jaw_aa], dim=-1)

        gt_jaw = batch['target'][...,:self.config.n_pose]
        gt_expr = batch['target'][...,self.config.n_pose:]
        gt_jaw_aa = utils_transform.sixd2aa(gt_jaw)
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
            
    def track(self):
        
        # make prediction by split the whole sequences into chunks
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        eval_all = {}
        for metric in self.all_metrics:
            eval_all[metric] = 0.0
        num_test_motions = len(self.test_data)
        for i in tqdm(range(num_test_motions)):
            self.flame.to(self.device)
            batch, motion_id = self.test_data[i]
            logger.info(f'Process [{motion_id}].')
            video_path = self.output_folder + f'/{motion_id}'
            
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
            self.motion_memory = None   # init diffusion memory for motin infilling
            start_id = 0
            diffusion_output = []          

            while start_id == 0 or start_id + self.input_motion_length <= self.num_frames:
                motion_split, flag_index = self.prepare_chunk_diffusion_input(batch, start_id, self.num_frames)
                if start_id == 0:
                    mem_idx = 0
                else:
                    mem_idx = self.input_motion_length - self.sld_wind_size
                print(f"Processing frame from No.{start_id} at mem {mem_idx}")
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
                print(f"Processing last frame No.{start_id} at mem {mem_idx}")
                diffusion_output.append(output_sample.cpu())
            logger.info(f'DDPM sample {self.num_frames} frames used: {self.sample_time} seconds.')

            diffusion_output = torch.cat(diffusion_output, dim=0)

            assert batch['lmk_2d'].shape[0] == diffusion_output.shape[0]

            if self.vis:
                # batch visualiza all frames
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

            torch.cuda.empty_cache()
        
        logger.info("==========Metrics for all test motion sequences:===========")
        for metric in eval_all:
            logger.info(f"{metric} : {eval_all[metric] / num_test_motions}")

def main():
    args = test_args()
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
        args.input_motion_length,
        args.fps,
        args.n_shape,
        args.n_exp,
        args.exp_name,
        args.load_tex,
        args.use_iris
    )

    motion_tracker = MotionTracker(args, pretrained_args.model, test_dataset, 'cuda')
    
    motion_tracker.track()

if __name__ == "__main__":
    main()
