import math
import os
import random
import pickle
import numpy as np
import trimesh
import torch

from data_loaders.dataloader_from_path import load_data, TestDataset

from model.FLAME import FLAME
from model.mica import MICA

from model.networks import PureMLP
from tqdm import tqdm

from utils import utils_transform, utils_visualize
from utils.metrics import get_metric_function
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import predict_args
from utils.famos_camera import batch_cam_to_img_project
from configs.config import get_cfg_defaults
from test import evaluate_prediction
from prepare_FaMoS import prepare_one_motion_for_test
from prepare_video import  VideoProcessor
import os.path
from enum import Enum
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import trimesh
from loguru import logger
from pytorch3d.io import load_obj
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from pytorch3d.utils import opencv_from_cameras_projection
from tqdm import tqdm
from time import time
from matplotlib import cm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

IMAGE_SIZE = 224
FOCAL_LEN = 1000.0
PRINCIPAL_POINT_OFFSET = 112.0

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

class View(Enum):
    GROUND_TRUTH = 1
    MESH_GT = 2
    MESH_PRED = 4
    MESH_OVERLAY = 8
    LANDMARKS = 16
    HEATMAP = 32

class MotionTracker:
    
    def __init__(self, config, pretrained_args, device='cuda'):
        
        self.config = config
        self.device = device
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

        # fixed camera params
        self.focal_length = torch.FloatTensor([config.focal_length, config.focal_length]).unsqueeze(0).to(self.device)
        self.principal_point = torch.FloatTensor([config.principal_point, config.principal_point]).unsqueeze(0).to(self.device)
        self.R = torch.eye(3).unsqueeze(0).to(self.device)
        self.image_size = torch.tensor([[config.image_size, config.image_size]]).to(self.device)

        # visualization settings
        self.colormap = cm.get_cmap('jet')
        self.vis_views = [
            [View.GROUND_TRUTH, View.MESH_GT, View.MESH_OVERLAY, View.LANDMARKS]   # View.GROUND_TRUTH, View.MESH_GT, View.MESH_PRED, View.LANDMARKS
        ]
        self.min_error = 0.
        self.max_error = 8.

        # diffusion models
        self.denoise_model, self.diffusion = self.load_diffusion_model_from_ckpt(config)
        
        # mica shape predictor
        self.mica = MICA(pretrained_args.mica)
        self.mica.load_model('cpu')
        self.mica.to(self.device)
        self.mica.eval()

        # load norm dict
        norm_dict_path = 'processed_data/FaMoS_norm_dict.pt'
        norm_dict = torch.load(norm_dict_path)
        self.mean = norm_dict['mean']
        self.std = norm_dict['std']    
        
        self.setup_renderer()
        self.setup_flame()

    def setup_flame(self):
        flame_template_file = 'flame_2020/head_template_mesh.obj'
        self.flame = FLAME(
            self.config.flame_model_path, 
            self.config.flame_lmk_embedding_path,
            self.config.n_shape, 
            self.config.n_exp).to(self.device)
        self.faces = load_obj(flame_template_file)[1]

        flame_vmask_path = "flame_2020/FLAME_masks.pkl"
        with open(flame_vmask_path, 'rb') as f:
            self.flame_v_mask = pickle.load(f, encoding="latin1")

        for k, v in self.flame_v_mask.items():
            self.flame_v_mask[k] = torch.from_numpy(v)
    
    def setup_renderer(self):
        
        self.config.image_size = self.get_image_size()

        raster_settings = RasterizationSettings(
            image_size=self.get_image_size(),
            faces_per_pixel=1,
            cull_backfaces=True,
            perspective_correct=True
        )

        self.lights = PointLights(
            device=self.device,
            location=((0.0, 0.0, -5.0),),
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.5, 0.5, 0.5),)
        )

        self.mesh_rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.renderer = MeshRenderer(
            rasterizer=self.mesh_rasterizer,
            shader=SoftPhongShader(device=self.device, lights=self.lights)
        )
    
    def load_test_motion(self):
        # load test motion sequence
        if self.config.test_mode == "in_the_wild":
            if self.config.video_path is None:
                logger.error('Please provide video path!')
                exit(1)
            video_processor = VideoProcessor(self.config)
            self.test_seq = video_processor.run(self.config.video_path)
        else:
            self.test_seq = prepare_one_motion_for_test(
                self.config.dataset, 
                self.config.split,
                self.config.subject_id, 
                self.config.motion_id, 
                flame_model_path=self.config.flame_model_path, 
                flame_lmk_embedding_path=self.config.flame_lmk_embedding_path,
                n_shape=self.config.n_shape, 
                n_exp=self.config.n_exp,
                fps=self.config.fps)

        self.motion_len = self.test_seq['lmk_2d'].shape[0]
        
        if 'occlusion_mask' not in self.test_seq:
            self.generate_random_occlusion_mask()
        
    def load_diffusion_model_from_ckpt(self, args):
        logger.info("Creating model and diffusion...")
        args.arch = args.arch[len("diffusion_") :]
        denoise_model, diffusion = create_model_and_diffusion(args)

        logger.info(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location="cpu")
        load_model_wo_clip(denoise_model, state_dict)

        denoise_model.to(args.device)  # dist_util.dev())
        denoise_model.eval()  # disable random masking
        return denoise_model, diffusion
    
    def get_image_size(self):
        return self.image_size[0][0].item(), self.image_size[0][1].item()
    
    def render_mesh(self, vertices, cameras, faces=None, white=True):
        """
        Args:
            vertices: flame mesh verts, [B, V, 3]
        """
        B = vertices.shape[0]
        V = vertices.shape[1]
        if faces is None:
            faces = self.faces.verts_idx.cuda()[None].repeat(B, 1, 1)
        if not white:
            verts_rgb = torch.from_numpy(np.array([80, 140, 200]) / 255.).cuda().float()[None, None, :].repeat(B, V, 1)
        else:
            verts_rgb = torch.from_numpy(np.array([1.0, 1.0, 1.0])).cuda().float()[None, None, :].repeat(B, V, 1)
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        meshes_world = Meshes(verts=[vertices[i] for i in range(B)], faces=[faces[i] for i in range(B)], textures=textures)

        blend = BlendParams(background_color=(1.0, 1.0, 1.0))
        
        fragments = self.mesh_rasterizer(meshes_world, cameras=cameras)
        rendering = self.renderer.shader(fragments, meshes_world, cameras=cameras, blend_params=blend)
        rendering = rendering.permute(0, 3, 1, 2).detach()
        return fragments, rendering[:, 0:3, :, :]
    
    def vertex_error_heatmap(self, vertices, cameras, vertex_error, faces=None):
        """
        Args:
            vertices: flame mesh verts, [B, V, 3]
            vertex_error: per vertex error [B, V]
        """
        B = vertices.shape[0]
        if faces is None:
            faces = self.faces.verts_idx.cuda()[None].repeat(B, 1, 1)
        vertex_error = vertex_error.to('cpu').numpy()
        vertex_color_code = ((vertex_error - self.min_error) / (self.max_error - self.min_error)) * 255.
        verts_rgb = self.colormap(vertex_color_code.astype(int))[:,:,:3]    # (B, V, 3)
        textures = TexturesVertex(verts_features=torch.from_numpy(verts_rgb).float().cuda())
        meshes_world = Meshes(verts=[vertices[i] for i in range(B)], faces=[faces[i] for i in range(B)], textures=textures)

        blend = BlendParams(background_color=(1.0, 1.0, 1.0))
        
        fragments = self.mesh_rasterizer(meshes_world, cameras=cameras)
        rendering = self.renderer.shader(fragments, meshes_world, cameras=cameras, blend_params=blend)
        rendering = rendering.permute(0, 3, 1, 2).detach()
        return rendering[:, 0:3, :, :]
    
    def vis_one_frame(
            self, 
            frame_id,
            vis_data, # dict
            visualizations
        ):
        # images, landmarks, landmarks_dense, _, _ = self.parse_batch(batch)

        
        image = vis_data['image']
        lmk_2d_gt = vis_data['lmk_2d_gt']
        lmk_2d_pred = vis_data['lmk_2d_pred']
        verts_pred = vis_data['verts_pred']
        trans_pred = vis_data['trans_pred']
        occlusion_mask = vis_data['occlusion_mask']
        
        verts_gt = vis_data.get('verts_gt', None)
        trans_gt = vis_data.get('trans_gt', None)
        verts_error = vis_data.get('verts_error', None)

        savefolder = os.path.join(self.output_folder, self.motion_name)
        Path(savefolder).mkdir(parents=True, exist_ok=True)

        self.renderer.rasterizer.raster_settings.image_size = self.get_image_size()
        
        if trans_gt is not None:
            cameras_gt = PerspectiveCameras(
                            device=self.device,
                            principal_point=self.principal_point, 
                            focal_length=-self.focal_length, # TODO dark megic for pytorch3d's ndc coord sys
                            R=self.R, T=trans_gt,
                            image_size=self.image_size,
                            in_ndc=False)
        

        cameras_pred = PerspectiveCameras(
                        device=self.device,
                        principal_point=self.principal_point, 
                        focal_length=-self.focal_length, # TODO dark megic for pytorch3d's ndc coord sys
                        R=self.R, T=trans_pred,
                        image_size=self.image_size,
                        in_ndc=False)
        
        
        final_views = []

        for views in visualizations:
            row = []
            for view in views:
                if view == View.GROUND_TRUTH:
                    row.append(image[0].cpu().numpy())
                if view == View.MESH_GT and verts_gt is not None:
                    raster_gt, mesh_gt = self.render_mesh(verts_gt, cameras_gt, white=False)
                    mesh_gt = mesh_gt[0].cpu().numpy()
                    row.append(mesh_gt)
                if view == View.MESH_PRED:
                    raster_pred, mesh_pred = self.render_mesh(verts_pred, cameras_pred, white=False)
                    mesh_pred = mesh_pred[0].cpu().numpy()
                    row.append(mesh_pred)
                if view == View.LANDMARKS:
                    gt_lmks = image.clone()
                    gt_lmks = utils_visualize.tensor_vis_landmarks(gt_lmks, lmk_2d_gt, 'g', occlusion_mask)
                    gt_lmks = utils_visualize.tensor_vis_landmarks(gt_lmks, lmk_2d_pred, 'r')
                    row.append(gt_lmks[0].cpu().numpy())
                if view == View.MESH_OVERLAY: 
                    back_image = image[0] # (3, h, w)
                    raster_pred, mesh_pred = self.render_mesh(verts_pred, cameras_pred, white=False)
                    mesh_mask = (raster_pred.pix_to_face[0].permute(2, 0, 1) > -1).long()   # (1, h, w)
                    blend = back_image * (1 - mesh_mask) + back_image * mesh_mask * 0.3 + mesh_pred[0] * 0.7 * mesh_mask
                    row.append(blend.cpu().numpy())
                if view == View.HEATMAP and verts_error is not None: 
                    heatmap = self.vertex_error_heatmap(verts_pred, cameras_pred, verts_error)
                    heatmap = heatmap[0].cpu().numpy()
                    row.append(heatmap)
            final_views.append(row)

            # VIDEO
            final_views = utils_visualize.merge_views(final_views)
            frame_id = str(frame_id).zfill(5)
            
            cv2.imwrite(f'{savefolder}/{frame_id}.jpg', final_views)
            
    def vis_all_frames(self):
        vis_data = self.prepare_vis_dict(with_image=True)

        frame_ids = vis_data['frame_id'].numpy().tolist()
        
        for i, fid in tqdm(enumerate(frame_ids)):
            frame_data = dict()
            for k in (vis_data.keys() - {'frame_id'}):
                frame_data[k] = vis_data[k][i].unsqueeze(0).to(self.device)
            self.vis_one_frame(fid, frame_data, visualizations=self.vis_views)
        
        self.output_video(fps=30)
    
    def output_video(self, fps=30):
        utils_visualize.images_to_video(self.output_folder, fps, self.motion_name)
    
    def inv_transform(self, target):
        
        target = target * self.std["target"] + self.mean["target"]
        
        return target
    
    def generate_random_occlusion_mask(self):
        occlusion_mask = torch.zeros(self.motion_len, 68)
        mask_type = 'non'
        
        if mask_type == 'right_corner':
            occlusion_mask[:, 9:13] = 1 # mask out right corner
            occlusion_mask[:, 54:58] = 1 
            occlusion_mask[:, 62:66] = 1 
        elif mask_type == 'mouth':
            occlusion_mask[:, 48:68] = 1 
        elif mask_type == 'top_left':
            occlusion_mask[:, 17:21] = 1 
            occlusion_mask[:, 0: 3] = 1 
            occlusion_mask[:, 36:38] = 1 
        elif mask_type == 'img_frame':
            occlusion_mask[~self.test_seq['img_mask'], :] = 1
            
        self.test_seq['occlusion_mask'] = occlusion_mask
            
    def prepare_diffusion_input(self):
        # prepare input data
        lmk_2d = self.test_seq['lmk_2d']
        lmk_3d_normed = self.test_seq['lmk_3d_normed']
        arcface_input = self.test_seq['arcface_input']
        input_motion_length = lmk_2d.shape[0]
        occlusion_mask = self.test_seq['occlusion_mask']
        
        # use mica to predict the mica shape code per frame
        mica_shapes = self.mica.predict_per_frame_shape(
            arcface_input.to(self.device)).to('cpu')[:,:self.config.n_shape]
        
        # TODO:agg mica_shape robust to anomalous frames
        mica_shape = torch.median(mica_shapes, dim=0).values
        
        # normalization
        if not self.config.no_normalization:
            
            lmk_2d = lmk_2d / self.config.image_size[0] * 2 - 1
            lmk_3d_normed = ((lmk_3d_normed.reshape(-1, 3) - self.mean['lmk_3d_normed']) / (self.std['lmk_3d_normed'] + 1e-8)).reshape(input_motion_length, -1, 3)
            mica_shape = (mica_shape - self.mean['target'][:self.config.n_shape]) / (self.std['target'][:self.config.n_shape] + 1e-8)
        
        return lmk_2d, lmk_3d_normed, mica_shape, occlusion_mask
    
    def sample_motion(self):
        
        sample_fn = self.diffusion.p_sample_loop
        
        lmk_2d, lmk_3d_normed, mica_shape, occlusion_mask = self.prepare_diffusion_input()
        
        with torch.no_grad():
            motion_length = lmk_2d.shape[0]

            model_kwargs = {
                "lmk_2d": lmk_2d.unsqueeze(0).to(self.device),
                "lmk_3d": lmk_3d_normed.unsqueeze(0).to(self.device),
                "mica_shape": mica_shape.unsqueeze(0).to(self.device),
                "occlusion_mask": occlusion_mask.unsqueeze(0).to(self.device),
            }

            if self.config.fix_noise:
                # fix noise seed for every frame
                noise = torch.randn(1, 1, 1).cuda()
                noise = noise.repeat(1, motion_length, self.config.target_nfeat)
            else:
                noise = None
            
            start_time = time()
            output_sample = sample_fn(
                self.denoise_model,
                (1, motion_length, self.config.target_nfeat),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=noise,
                const_noise=False,
            )
            time_used = time() - start_time
            logger.info(f'DDPM sample {self.motion_len} frames used: {time_used} seconds.')
            output_sample = output_sample.cpu().float().squeeze(0)
            if not self.config.no_normalization:
                output_sample = self.inv_transform(output_sample)
        
        self.diffusion_output = output_sample
    
    def parse_model_target(self, target):
        nshape = self.config.n_shape
        nexp = self.config.n_exp
        ntrans = self.config.n_trans
        shape = target[:, :nshape]
        exp = target[:, nshape:nshape+nexp]
        pose = target[:, nshape+nexp:-ntrans]
        trans = target[:, -ntrans:]
        return shape, exp, pose, trans
    
    def prepare_vis_dict(self, with_image=True):
        img_mask = torch.ones(self.motion_len).bool()
        if with_image:
            image = self.test_seq['cropped_imgs']
            img_mask = self.test_seq['img_mask']
            
        num_frames = sum(img_mask)
        
        lmk_2d_gt = self.test_seq['lmk_2d'][img_mask]
        diffusion_pred = self.diffusion_output[img_mask]
        shape_pred, expr_pred, pose_pred, trans_pred = self.parse_model_target(diffusion_pred)
        pose_aa_pred = utils_transform.sixd2aa(pose_pred.reshape(-1, 6)).reshape(num_frames, -1)
        verts_pred, lmk_3d_pred = self.flame(shape_pred, expr_pred, pose_aa_pred) 
        # 2d reprojection
        lmk_2d_pred = batch_cam_to_img_project(lmk_3d_pred, trans_pred) 
        
        vis_dict = {
            'verts_pred': verts_pred,
            'trans_pred': trans_pred,
            'lmk_2d_gt': lmk_2d_gt,
            'lmk_2d_pred': lmk_2d_pred,
            'occlusion_mask': self.test_seq['occlusion_mask'][img_mask],
            'frame_id': self.test_seq['frame_id'][img_mask],
        }
        
        # gt information
        if 'target' in self.test_seq:
            diffusion_target = self.test_seq['target'][img_mask]
            shape_gt, expr_gt, pose_gt, trans_gt = self.parse_model_target(diffusion_target)
            pose_aa_gt = utils_transform.sixd2aa(pose_gt.reshape(-1, 6)).reshape(num_frames, -1)
            verts_gt, lmk_3d_gt = self.flame(shape_gt, expr_gt, pose_aa_gt)   
        
            verts_error = torch.norm(verts_pred-verts_gt, p=2, dim=-1) * 1000.0 # converts from m to mm
            vis_dict['verts_gt'] = verts_gt
            vis_dict['trans_gt'] = trans_gt 
            vis_dict['verts_error'] = verts_error
        
        if with_image:
            vis_dict['image'] = image
        
        return vis_dict

    def eval_motion(self):
            
        log = evaluate_prediction(
            self.config,
            all_metrics,
            self.diffusion_output,
            self.flame,
            self.test_seq['target'].to('cpu'), 
            self.test_seq['lmk_2d'].reshape(self.motion_len, -1, 2).to('cpu'),
            self.config.fps,
            self.motion_id,
            self.flame_v_mask,
        )
        logger.info("Metrics for the predictions")
        for metric in pred_metrics:
            logger.info(f"{metric} : {log[metric]}")

        logger.info("Metrics for the ground truth")
        for metric in gt_metrics:
            logger.info(f"{metric} : {log[metric]}")
    
    def track(self):
        
        self.load_test_motion()
        self.sample_motion()
        if 'target' in self.test_seq:
            self.eval_motion()
        self.vis_all_frames()
            

def main():
    args = predict_args()
    pretrained_args = get_cfg_defaults()
    # to guanrantee reproducibility
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    motion_tracker = MotionTracker(args, pretrained_args, 'cuda')
    
    motion_tracker.track()

if __name__ == "__main__":
    main()
