import math
import os
import random
import pickle
import numpy as np
import trimesh
import torch

from data_loaders.dataloader import load_data, TestDataset

from model.FLAME import FLAME

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

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

IMAGE_SIZE = 224
FOCAL_LEN = 1000.0
PRINCIPAL_POINT_OFFSET = 112.0
MEAN_TRANS = torch.FloatTensor([0.004, 0.222, 1.200])   # guessing from training 

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
    
    def __init__(self, config, pretrained_args, test_seq, device='cuda'):
        self.config = config
        self.device = device
        
        # IO setups
        self.motion_id = config.motion_id # name of the tested motion sequence
        self.save_folder = self.config.save_folder
        self.output_folder = os.path.join(self.save_folder, self.config.arch, self.motion_id, self.config.exp_name)
        
        logger.add(os.path.join(self.output_folder, 'predict.log'))
        logger.info(f"Using device {self.device}.")
        logger.info(f"Predict motion {self.motion_id}.")

        # fixed camera params
        self.focal_length = torch.FloatTensor([FOCAL_LEN, FOCAL_LEN]).unsqueeze(0).to(self.device)
        self.principal_point = torch.FloatTensor([PRINCIPAL_POINT_OFFSET, PRINCIPAL_POINT_OFFSET]).unsqueeze(0).to(self.device)
        self.R = torch.eye(3).unsqueeze(0).to(self.device)
        self.image_size = torch.tensor([[IMAGE_SIZE, IMAGE_SIZE]]).to(self.device)
        

        # diffusion models
        self.cam_model, self.denoise_model, self.diffusion = self.load_diffusion_model_from_ckpt(config, pretrained_args)

        # load norm dict
        norm_dict_path = 'processed_data/FaMoS_norm_dict.pt'
        norm_dict = torch.load(norm_dict_path)
        mean_target = norm_dict['mean']['target']
        std_target = norm_dict['std']['target']
        norm_dict['mean']['target'] = torch.cat(
            [mean_target[:config.n_shape], mean_target[300:300+config.n_exp], mean_target[400:]])
        norm_dict['std']['target'] = torch.cat(
            [std_target[:config.n_shape], std_target[300:300+config.n_exp], std_target[400:]])
        self.mean = norm_dict['mean']
        self.std = norm_dict['std']

        # test data
        self.test_seq = test_seq
        self.motion_len = test_seq['target'].shape[0]
        self.generate_random_occlusion_mask()
        
        
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
    
    def load_diffusion_model_from_ckpt(self, args, pretrained_args):
        logger.info("Creating model and diffusion...")
        args.arch = args.arch[len("diffusion_") :]
        cam_model, denoise_model, diffusion = create_model_and_diffusion(args, pretrained_args)

        logger.info(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location="cpu")
        load_model_wo_clip(denoise_model, state_dict)

        cam_model.to(args.device)
        cam_model.eval()
        denoise_model.to(args.device)  # dist_util.dev())
        denoise_model.eval()  # disable random masking
        return cam_model, denoise_model, diffusion
    
    def get_image_size(self):
        return self.image_size[0][0].item(), self.image_size[0][1].item()
    
    def get_verts_error_heatmap(self, raster_output, verts_error):
        # TODO
        pass
        # l2 = tensor2im(values)
        # l2 = cv2.cvtColor(l2, cv2.COLOR_RGB2BGR)
        # l2 = cv2.normalize(l2, None, 0, 255, cv2.NORM_MINMAX)
        # heatmap = cv2.applyColorMap(l2, cv2.COLORMAP_JET)
        # heatmap = cv2.cvtColor(cv2.addWeighted(heatmap, 0.75, l2, 0.25, 0).astype(np.uint8), cv2.COLOR_BGR2RGB) / 255.
        # heatmap = torch.from_numpy(heatmap).permute(2, 0, 1)

        # return heatmap
    
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
    
    def vis_one_frame(
            self, 
            frame_id,
            vis_data, # dict
            visualizations, 
            frame_dst='video'
        ):
        # images, landmarks, landmarks_dense, _, _ = self.parse_batch(batch)

        image = vis_data['image']
        lmk_2d_gt = vis_data['lmk_2d_gt']
        lmk_2d_pred = vis_data['lmk_2d_pred']
        verts_pred = vis_data['verts_pred']
        trans_pred = vis_data['trans_pred']
        trans_gt = vis_data['trans_gt']
        occlusion_mask = vis_data['occlusion_mask']
        
        verts_gt = vis_data.get('verts_gt', None)
        verts_error = vis_data.get('verts_error', None)

        # input_image = util.to_image(batch['image'].clone()[0].cpu().numpy())

        savefolder = os.path.join(self.output_folder, frame_dst)
        Path(savefolder).mkdir(parents=True, exist_ok=True)

        cameras = PerspectiveCameras(
            device=self.device,
            principal_point=self.principal_point, 
            focal_length=-self.focal_length, # TODO dark megic for pytorch3d's ndc coord sys
            R=self.R, T=trans_pred,
            image_size=self.image_size,
            in_ndc=False)
        

        self.renderer.rasterizer.raster_settings.image_size = self.get_image_size()

        # lmk68 = self.cameras.transform_points_screen(lmk68, image_size=self.image_size)
        # shape_mask = ((ops['alpha_images'] * ops['mask_images_mesh']) > 0.).int()[0]

        final_views = []

        for views in visualizations:
            row = []
            for view in views:
                if view == View.GROUND_TRUTH:
                    row.append(image[0].cpu().numpy())
                if view == View.MESH_GT and verts_gt is not None:
                    raster_gt, mesh_gt = self.render_mesh(verts_gt, cameras, white=False)
                    mesh_gt = mesh_gt[0].cpu().numpy()
                    row.append(mesh_gt)
                if view == View.MESH_PRED:
                    raster_pred, mesh_pred = self.render_mesh(verts_pred, cameras, white=False)
                    mesh_pred = mesh_pred[0].cpu().numpy()
                    row.append(mesh_pred)
                if view == View.LANDMARKS:
                    gt_lmks = image.clone()
                    gt_lmks = utils_visualize.tensor_vis_landmarks(gt_lmks, lmk_2d_gt, 'g', occlusion_mask)
                    gt_lmks = utils_visualize.tensor_vis_landmarks(gt_lmks, lmk_2d_pred, 'r')
                    row.append(gt_lmks[0].cpu().numpy())
                # if view == View.SHAPE_OVERLAY: TODO
                #     shape = self.render_shape(vertices, white=False)[0] * shape_mask
                #     blend = images[0] * (1 - shape_mask) + images[0] * shape_mask * 0.3 + shape * 0.7 * shape_mask
                #     row.append(blend.cpu().numpy())
                if view == View.HEATMAP and verts_error is not None: # TODO
                    heatmap = self.get_heatmap(raster_pred, verts_error)
                    row.append(heatmap)
            final_views.append(row)

            # VIDEO
            final_views = utils_visualize.merge_views(final_views)
            frame_id = str(frame_id).zfill(5)
            cv2.imwrite(f'{savefolder}/{frame_id}.jpg', final_views)
            
    def vis_all_frames(self):
        vis_data = self.prepare_vis_dict(with_image=True)

        frame_ids = vis_data['frame_id'].numpy().tolist()
        vis_views = [
            [View.GROUND_TRUTH, View.MESH_GT, View.MESH_PRED, View.LANDMARKS]   # View.GROUND_TRUTH, View.MESH_GT, View.MESH_PRED, View.LANDMARKS
        ]
        
        for i, fid in tqdm(enumerate(frame_ids)):
            frame_data = dict()
            for k in (vis_data.keys() - {'frame_id'}):
                frame_data[k] = vis_data[k][i].unsqueeze(0).to(self.device)
            self.vis_one_frame(fid, frame_data, visualizations=vis_views)
        
        self.output_video(fps=25)
    
    def output_video(self, fps=25):
        utils_visualize.images_to_video(self.output_folder, fps)
    
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
        flame_params = self.test_seq['target']
        lmk_2d = self.test_seq['lmk_2d']
        lmk_3d_normed = self.test_seq['lmk_3d_normed']
        img_arr = self.test_seq['arcface_imgs'][:4] # TODO
        input_motion_length = flame_params.shape[0]
        occlusion_mask = self.test_seq['occlusion_mask']
        
        # normalization
        
        lmk_2d = lmk_2d / self.config.image_size[0] * 2 - 1
        lmk_3d_normed = ((lmk_3d_normed.reshape(-1, 3) - self.mean['lmk_3d_normed']) / (self.std['lmk_3d_normed'] + 1e-8)).reshape(input_motion_length, -1, 3)
        flame_params = (flame_params - self.mean['target']) / (self.std['target'] + 1e-8)
        
        return flame_params, lmk_2d, lmk_3d_normed, img_arr, occlusion_mask
    
    def sample_motion(self):
        sample_fn = self.diffusion.p_sample_loop
        
        flame_params, lmk_2d, lmk_3d_normed, img_arr, occlusion_mask = self.prepare_diffusion_input()
        
        with torch.no_grad():
            motion_length = flame_params.shape[0]
            flame_params = flame_params.unsqueeze(0).to(self.device)
            lmk_2d = lmk_2d.unsqueeze(0).to(self.device)
            occlusion_mask = occlusion_mask.unsqueeze(0).to(self.device)

            bs, n = lmk_2d.shape[:2]
            occlusion = (1-occlusion_mask).unsqueeze(-1)
            lmk_2d_occ = (lmk_2d * occlusion).reshape(bs, n, -1)
            trans_cam = self.cam_model(lmk_2d_occ, flame_params[:, :, :self.config.n_shape])

            target = torch.cat([flame_params, trans_cam], dim=-1)
            model_kwargs = {
                "lmk_2d": lmk_2d,
                "lmk_3d": lmk_3d_normed.unsqueeze(0).to(self.device),
                "img_arr": img_arr.unsqueeze(0).to(self.device),
                "occlusion_mask": occlusion_mask,
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
            target = target.cpu().float().squeeze(0)
            if not self.config.no_normalization:
                output_sample[:,:-3] = self.inv_transform(output_sample[:,:-3])
                target[:,:-3] = self.inv_transform(target[:,:-3])
        
        self.diffusion_output = {
            'pred': output_sample,
            'target': target
        }
        
        return output_sample, target
    
    def parse_model_target(self, target):
        nshape = 100
        nexp = 50 
        npose = 5*6 
        ntrans = 3 
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
        diffusion_target = self.diffusion_output['target'][img_mask]
        diffusion_pred = self.diffusion_output['pred'][img_mask]
        
        shape_gt, expr_gt, pose_gt, trans_gt = self.parse_model_target(diffusion_target)
        shape_pred, expr_pred, pose_pred, trans_pred = self.parse_model_target(diffusion_pred)
        
        # pose 6d to aa
        pose_aa_gt = utils_transform.sixd2aa(pose_gt.reshape(-1, 6)).reshape(num_frames, -1)
        pose_aa_pred = utils_transform.sixd2aa(pose_pred.reshape(-1, 6)).reshape(num_frames, -1)
        
        
        # flame regress
        verts_gt, lmk_3d_gt = self.flame(shape_gt, expr_gt, pose_aa_gt)   
        verts_pred, lmk_3d_pred = self.flame(shape_pred, expr_pred, pose_aa_pred)         
        
        # 2d reprojection
        lmk_2d_pred = batch_cam_to_img_project(lmk_3d_pred, trans_pred) 
        
        vis_dict = {
            'verts_gt': verts_gt,
            'verts_pred': verts_pred,
            'trans_gt': trans_gt + MEAN_TRANS, 
            'trans_pred': trans_pred + MEAN_TRANS,
            'lmk_2d_gt': lmk_2d_gt,
            'lmk_2d_pred': lmk_2d_pred,
            'occlusion_mask': self.test_seq['occlusion_mask'][img_mask],
            'frame_id': self.test_seq['frame_id'][img_mask]
        }
        
        if with_image:
            vis_dict['image'] = image
        
        return vis_dict

    def eval_motion(self):
        log = evaluate_prediction(
            self.config,
            all_metrics,
            self.diffusion_output['pred'].squeeze(0),
            self.flame,
            self.diffusion_output['target'].squeeze(0), 
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
        



def main():
    args = predict_args()
    pretrained_args = get_cfg_defaults()
    # to guanrantee reproducibility
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    test_seq = prepare_one_motion_for_test(
        args.dataset, 
        args.subject_id, 
        args.motion_id, 
        flame_model_path=args.flame_model_path, 
        flame_lmk_embedding_path=args.flame_lmk_embedding_path,
        n_shape=args.n_shape, 
        n_exp=args.n_exp)
    
    args.motion_id = f'{args.subject_id}_{args.motion_id}'
    motion_tracker = MotionTracker(args, pretrained_args, test_seq, 'cuda')
    
    motion_tracker.sample_motion()
    
    motion_tracker.eval_motion()
    
    motion_tracker.vis_all_frames()

if __name__ == "__main__":
    main()
