
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
from model.FLAME import FLAME
from configs.config import get_cfg_defaults
from utils import dataset_setting

import torch
import torchvision.transforms.functional as F_v
import torch.nn.functional as F
from pytorch3d.io import load_obj
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams
from pytorch3d.structures import Meshes
from utils.famos_camera import batch_perspective_project
# pretrained

import imageio
from data_loaders.dataloader_w3d import load_split_for_subject, TestOneMotion
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
        # IO setups
                        
        # name of the tested motion sequence
        self.save_folder = self.config.save_folder
        self.output_folder = os.path.join(self.save_folder, self.config.arch, config.subject_id, config.exp_name)
        
        logger.add(os.path.join(self.output_folder, 'predict.log'))
        logger.info(f"Using device {self.device}.")
        
        h, w = dataset_setting.image_size[self.config.dataset]
        self.image_size = torch.tensor([h, w]).to(self.device)  # [300, 400]
        self.resize_factor=0.5  # resize the final grid image
        self.view_h, self.view_w = int(h*self.resize_factor), int(w*4*self.resize_factor)
        
        self.sample_time = 0

        # visualization settings
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
        ]
        gt_metrics = [
            "gt_jitter",
        ]

        self.all_metrics = pred_metrics + gt_metrics

    
    def _create_flame(self):
        self.flame = FLAME(self.model_cfg).to(self.device)
        flame_template_file = 'flame_2020/head_template_mesh.obj'
        self.faces = load_obj(flame_template_file)[1]

        flame_vmask_path = "flame_2020/FLAME_masks.pkl"
        with open(flame_vmask_path, 'rb') as f:
            self.flame_v_mask = pickle.load(f, encoding="latin1")

        for k, v in self.flame_v_mask.items():
            self.flame_v_mask[k] = torch.from_numpy(v)
    
    def _setup_renderer(self):

        raster_settings = RasterizationSettings(
            image_size=(self.image_size[0].item(), self.image_size[1].item()),
            faces_per_pixel=1,
            cull_backfaces=True,
            perspective_correct=True
        )

        self.lights = PointLights(
            device=self.device,
            location=((0.0, 0.0, 1.0),),
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.5, 0.5, 0.5),)
        )

        self.mesh_rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.renderer = MeshRenderer(
            rasterizer=self.mesh_rasterizer,
            shader=SoftPhongShader(device=self.device, lights=self.lights)
        )
    
    def setup_cameras(self, calib, bs):
        extrin = calib['extrinsics']
        intrin = calib['intrinsics']
        extrins = torch.from_numpy(extrin).unsqueeze(0).repeat(bs,1,1).float()   # (bs, 3, 4)
        R, T = extrins[:,:3,:3].transpose(1, 2), extrins[:,:3,3]    # convert rotation into row vectors to fit pytorch3d standard

        focal_length = torch.FloatTensor([intrin[0, 0], intrin[1, 1]]).unsqueeze(0).repeat(bs,1) # (bs, 2)
        principal_point = torch.FloatTensor([intrin[0, 2], intrin[1, 2]]).unsqueeze(0).repeat(bs,1)

        image_size = self.image_size.unsqueeze(0).repeat(bs, 1) # (bs, 2)
        self.cameras = PerspectiveCameras(
            device=self.device,
            principal_point=principal_point, 
            focal_length=-focal_length, # dark megic for pytorch3d's ndc coord sys
            R=R, 
            T=T,
            image_size=image_size,
            in_ndc=False)
        
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

    def load_diffusion_model_from_ckpt(self, args, model_cfg):
        logger.info("Creating model and diffusion...")
        args.arch = args.arch[len("diffusion_") :]
        self.denoise_model, self.diffusion = create_model_and_diffusion(args, model_cfg, self.device)

        logger.info(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location="cpu")
        self.denoise_model.load_state_dict(state_dict, strict=False)

        self.denoise_model.to(self.device)  # dist_util.dev())
        self.denoise_model.eval()  # disable random masking

        # create audio encoder tuned in the state_dict
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        w2v_ckpt = {}
        for key in state_dict.keys():
            if key.startswith('audio_encoder.'):
                k = key.replace("audio_encoder.","")
                w2v_ckpt[k] = state_dict[key]
        if len(w2v_ckpt) > 0:
            self.audio_encoder.load_state_dict(w2v_ckpt, strict=True)
            logger.info(f"Load Audio Encoder Successfully from CKPT!")
        self.audio_encoder.to(self.device)
        self.audio_encoder.eval()
    
    def output_video(self, fps=30):
        utils_visualize.images_to_video(self.output_folder, fps, self.motion_name)
    
    def sample_motion(self, motion_split, mem_idx):
        
        sample_fn = self.diffusion.p_sample_loop

        with torch.no_grad():
            split_length = motion_split['image'].shape[0]
            
            model_kwargs = {}
            for key in ['image', 'audio_emb', 'img_mask', 'lmk_mask', 'lmk_2d']:
                model_kwargs[key] = motion_split[key].unsqueeze(0).to(self.device)

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

    def batch_3d_to_2d(self, calibration, lmk_3d):
    
        # all in tensor
        bs = lmk_3d.shape[0]
        device = lmk_3d.device
        camera_intrinsics = torch.from_numpy(calibration["intrinsics"]).expand(bs,-1,-1).float().to(device)
        camera_extrinsics = torch.from_numpy(calibration["extrinsics"]).expand(bs,-1,-1).float().to(device)
        radial_distortion = torch.from_numpy(calibration["radial_distortion"]).expand(bs,-1).float().to(device)
        
        lmk_2d = batch_perspective_project(lmk_3d, camera_intrinsics, camera_extrinsics, radial_distortion)

        return lmk_2d
    
    def vis_motion_split(self, gt_data, diffusion_sample, calibration, with_audio):

        # to gpu
        for k in gt_data:
            gt_data[k] = gt_data[k].to(diffusion_sample.device)
        
        # prepare vis data dict 
        diff_jaw = diffusion_sample[...,:self.config.n_pose]
        diff_expr = diffusion_sample[...,self.config.n_pose:]
        diff_jaw_aa = utils_transform.sixd2aa(diff_jaw)
        diff_rot_aa = gt_data['pose'].clone()
        diff_rot_aa[:,6:] = diff_jaw_aa
        
        # flame decoder
        gt_verts, _, _ = self.flame(
            shape_params=gt_data['shape'], 
            expression_params=gt_data['expr'],
            pose_params=gt_data['pose'])
        gt_verts += gt_data['trans']
        
        diff_verts, diff_lmk2d, _ = self.flame(
            shape_params=gt_data['shape'], 
            expression_params=diff_expr,
            pose_params=diff_rot_aa)
        diff_verts += gt_data['trans']
        diff_lmk2d += gt_data['trans']
        
        # perspective projection
        diff_lmk2d = self.batch_3d_to_2d(calibration, diff_lmk2d)
        
        # # render
        _, mesh_gt = self.render_mesh(gt_verts, self.cameras, white=False)

        raster_pred, mesh_pred = self.render_mesh(diff_verts, self.cameras, white=False)
        mesh_mask = (raster_pred.pix_to_face.permute(0, 3, 1, 2) > -1).long()   # (bs, 1, h, w)
        
        blend_pred = gt_data['image'] * (1 - mesh_mask) + gt_data['image'] * mesh_mask * 0.3 + mesh_pred * 0.7 * mesh_mask  # (bs, 3, h, w)
        
        # landmarks vis
        lmk2d_vis = utils_visualize.tensor_vis_landmarks(gt_data['image'], diff_lmk2d, gt_data['lmk_2d'], isScale=False)
        
        gt_img = gt_data['image'] * gt_data['img_mask'].unsqueeze(1)

        for i in range(diff_lmk2d.shape[0]):
            vis_dict = {
                'gt_img': gt_img[i].detach().cpu(),   # (3, h, w)
                'gt_mesh': mesh_gt[i].detach().cpu(),  # (3, h, w)
                'diff_mesh': blend_pred[i].detach().cpu(),  # (3, h, w)
                'lmk': lmk2d_vis[i].detach().cpu()
            }
            grid_image = self.visualize(vis_dict)
            if with_audio:
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
        test_motion, 
    ):      
        
        diff_jaw = diffusion_output[...,:self.config.n_pose]
        diff_expr = diffusion_output[...,self.config.n_pose:]
        diff_jaw_aa = utils_transform.sixd2aa(diff_jaw)
        diff_rot_aa = test_motion.rot_aa.clone()
        diff_rot_aa[:,6:] = diff_jaw_aa
        
        # flame decoder
        verts_gt, lmk_3d_gt, _ = self.flame(
            shape_params=test_motion.shape, 
            expression_params=test_motion.expression,
            pose_params=test_motion.rot_aa)
        verts_gt += test_motion.trans
        lmk_3d_gt += test_motion.trans
        
        # flame decoder
        verts_pred, lmk_3d_pred, _ = self.flame(
            shape_params=test_motion.shape, 
            expression_params=diff_expr,
            pose_params=diff_rot_aa)
        verts_pred += test_motion.trans 
        lmk_3d_pred += test_motion.trans

        eval_log = {}
        for metric in self.all_metrics:
            eval_log[metric] = (
                get_metric_function(metric)(
                    diff_expr, diff_jaw_aa, verts_pred, lmk_3d_pred,
                    test_motion.expression, test_motion.rot_aa[:,6:], verts_gt, lmk_3d_gt,
                    self.config.fps, self.flame_v_mask 
                )
                .numpy()
            )
        
        return eval_log
    
    def track(self):
        
        # make prediction by split the whole sequences into chunks due to memory limit
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        eval_all = {}
        for metric in self.all_metrics:
            eval_all[metric] = 0.0
        num_test_motions = len(self.test_data['img_folder'])
        for i in tqdm(range(num_test_motions)):
            self.flame.to(self.device)
            motion_split_paths = {}
            for k in self.test_data:
                motion_split_paths[k] = self.test_data[k][i]
            test_motion = TestOneMotion(
                self.config.dataset,
                motion_split_paths,
                self.input_motion_length,
                self.config.occlusion_mask_prob,
                self.config.fps,
                self.config.exp_name,
                self.audio_encoder
            )
            subject_id = test_motion.subject_id
            motion_id = test_motion.motion_id
            logger.info(f'Process [{subject_id} -- {motion_id}].')
            video_path = self.output_folder + f'/{motion_id}'
            # set the output writer
            audio_path = None
            if test_motion.with_audio:
                audio_path = test_motion.audio_path
                self.writer = cv2.VideoWriter(
                    video_path+'.mp4', fourcc, self.config.fps, 
                    (self.view_w, self.view_h))
            else:
                self.writer = imageio.get_writer(video_path + '.gif', mode='I')
            self.num_frames = test_motion.num_frames
            self.motion_memory = None   # init diffusion memory for motin infilling
            start_id = 0
            diffusion_output = []
            while start_id + self.input_motion_length <= self.num_frames:
                motion_split = test_motion.prepare_chunk_motion(start_id)
                if start_id == 0:
                    mem_idx = 0
                else:
                    mem_idx = self.input_motion_length - self.sld_wind_size
                print(f"Processing frame from No.{start_id} at mem {mem_idx}")
                output_sample = self.sample_motion(motion_split, mem_idx)
                # get gt data
                gt_split = {}
                gt_split['image'] = motion_split['original_img'][mem_idx:]
                gt_split['lmk_2d'] = test_motion.lmk_2d[start_id:start_id+self.input_motion_length][mem_idx:]
                gt_split['img_mask'] = test_motion.img_mask[start_id:start_id+self.input_motion_length][mem_idx:]
                gt_split['shape'] = test_motion.shape[start_id:start_id+self.input_motion_length][mem_idx:]
                gt_split['expr'] = test_motion.expression[start_id:start_id+self.input_motion_length][mem_idx:]
                gt_split['pose'] = test_motion.rot_aa[start_id:start_id+self.input_motion_length][mem_idx:]
                gt_split['trans'] = test_motion.trans[start_id:start_id+self.input_motion_length][mem_idx:]
                bs = gt_split['trans'].shape[0]
                self.setup_cameras(test_motion.calib, bs)
                self.vis_motion_split(gt_split, output_sample, test_motion.calib, test_motion.with_audio)
                start_id += self.sld_wind_size
                diffusion_output.append(output_sample.cpu())
            
            if start_id < self.num_frames:
                last_start_id = self.num_frames-self.input_motion_length
                motion_split = test_motion.prepare_chunk_motion(last_start_id)
                mem_idx = self.input_motion_length - (
                    self.num_frames - (start_id - self.sld_wind_size + self.input_motion_length))
                
                output_sample = self.sample_motion(motion_split, mem_idx)
                start_id = last_start_id
                print(f"Processing last frame No.{start_id} at mem {mem_idx}")
                # get gt data
                gt_split = {}
                gt_split['image'] = motion_split['original_img'][mem_idx:]
                gt_split['lmk_2d'] = test_motion.lmk_2d[start_id:start_id+self.input_motion_length][mem_idx:]
                gt_split['img_mask'] = test_motion.img_mask[start_id:start_id+self.input_motion_length][mem_idx:]
                gt_split['shape'] = test_motion.shape[start_id:start_id+self.input_motion_length][mem_idx:]
                gt_split['expr'] = test_motion.expression[start_id:start_id+self.input_motion_length][mem_idx:]
                gt_split['pose'] = test_motion.rot_aa[start_id:start_id+self.input_motion_length][mem_idx:]
                gt_split['trans'] = test_motion.trans[start_id:start_id+self.input_motion_length][mem_idx:]
                bs = gt_split['trans'].shape[0]
                self.setup_cameras(test_motion.calib, bs)
                self.vis_motion_split(gt_split, output_sample, test_motion.calib, test_motion.with_audio)
                diffusion_output.append(output_sample.cpu())
            logger.info(f'DDPM sample {self.num_frames} frames used: {self.sample_time} seconds.')
            
            # concat audio 
            if test_motion.with_audio:
                self.writer.release()
                os.system(f"ffmpeg -i {video_path}.mp4 -i {audio_path} -c:v copy {video_path}_audio.mp4")
                os.system(f"rm {video_path}.mp4")
            
            # start evaluation
            self.flame.to('cpu')
            diffusion_output = torch.cat(diffusion_output, dim=0)
            eval_log = self.evaluate_one_motion(diffusion_output, test_motion)
            
            for metric in eval_log:
                eval_all[metric] += eval_log[metric]

            torch.cuda.empty_cache()
        
        logger.info("Metrics for all test motion sequences:")
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
    
    args.dataset = 'vocaset'
    
    split_data = load_split_for_subject('vocaset', args.dataset_path, args.subject_id, args.split, args.motion_id)

    motion_tracker = MotionTracker(args, pretrained_args.model, split_data, 'cuda')
    
    motion_tracker.track()

if __name__ == "__main__":
    main()
