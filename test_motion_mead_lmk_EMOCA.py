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
from collections import defaultdict
import argparse

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
from data_loaders.dataloader_MEAD_flint import load_test_data, TestMeadDataset
import ffmpeg
import pickle
from model.deca import EMOCA

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class MotionTracker:
    
    def __init__(self, config, model_cfg, test_data, device='cuda'):
        
        self.config = config
        self.model_cfg = model_cfg
        self.device = device
        self.sld_wind_size = config.sld_wind_size
        self.test_data = test_data
        self.original_data_folder = 'dataset/mead_25fps/original_data'
        self.vis = config.vis
        self.save_rec = config.save_rec
        # IO setups
                        
        # name of the tested motion sequence
        self.save_folder = self.config.save_folder
        self.output_folder = os.path.join(self.save_folder, 'EMOCA', config.exp_name)
        
        if self.save_rec:
            self.sample_folder = os.path.join(self.output_folder, 'EMOCA_reconstruction')
            if not os.path.exists(self.sample_folder):
                os.makedirs(self.sample_folder)
        
        logger.add(os.path.join(self.output_folder, 'test_mead_wrt_gt.log'))
        logger.info(f"Using device {self.device}.")

        # vis settings
        self.to_mp4 = True # if true then to mp4, false then to gif wo audio
        self.visualization_batch = 10
        self.image_size = model_cfg.image_size
        self.resize_factor=1.0  # resize the final grid image
        self.heatmap_view = False
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

        # load emoca model
        self.load_emoca()
        
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
            "lmk2d_vis_reproj_error",
            "lmk2d_invis_reproj_error"
        ]

        # # from emica pseudo gt
        # gt_metrics = [
        #     "gt_jitter",
        #     "gt_mouth_closure",
        # ]

        self.all_metrics = pred_metrics # + gt_metrics
    
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

    def load_emoca(self):
        self.emoca = EMOCA(self.model_cfg)
        self.emoca.to(self.device)
    
    def output_video(self, fps=30):
        utils_visualize.images_to_video(self.output_folder, fps, self.motion_name)
    
    def vis_motion_split(self, gt_data, emoca_codes):
    
        # to gpu
        for k in gt_data:
            gt_data[k] = gt_data[k].to(self.device)
            emoca_codes[k] = emoca_codes[k].to(self.device)
        
        global_rot_aa = gt_data['global_pose']
        
        emoca_exp = emoca_codes['exp_emoca']
        deca_exp = emoca_codes['exp']
        deca_jaw_aa = emoca_codes['pose'][3:]
        deca_pose_aa = torch.cat([global_rot_aa, deca_jaw_aa], dim=-1)

        gt_jaw_aa = gt_data['jaw']
        gt_exp = gt_data['exp']
        cam = gt_data['cam']
        gt_rot_aa = torch.cat([global_rot_aa, gt_jaw_aa], dim=-1)
        
        # flame decoder
        emica_verts, _ = self.flame(gt_data['shape'], gt_exp, gt_rot_aa)
        emica_trans_verts = batch_orth_proj(emica_verts, cam)
        emica_trans_verts[:, :, 1:] = -emica_trans_verts[:, :, 1:]

        # flame decoder for deca    
        deca_verts, deca_lmk3d = self.flame(gt_data['shape'], deca_exp, deca_pose_aa)
        deca_trans_verts = batch_orth_proj(deca_verts, cam)
        deca_trans_verts[:, :, 1:] = -deca_trans_verts[:, :, 1:]


        # flame decoder for emoca
        emoca_verts, deca_lmk3d = self.flame(gt_data['shape'], emoca_exp, deca_pose_aa)
        emoca_trans_verts = batch_orth_proj(emoca_verts, cam)
        emoca_trans_verts[:, :, 1:] = -emoca_trans_verts[:, :, 1:]   
        
        # # render
        deca_render_images = self.render.render_shape(deca_verts, deca_trans_verts, images=gt_data['image'])
        emoca_render_images = self.render.render_shape(emoca_verts, emoca_trans_verts, images=gt_data['image'])
        emica_render_images = self.render.render_shape(emica_verts, emica_trans_verts, images=gt_data['image'])
        # if self.heatmap_view:
        #     vertex_error = torch.norm(emica_verts - diff_verts, p=2, dim=-1) * 1000. # vertex dist in mm
        #     face_error_colors = self.get_vertex_error_heat_color(vertex_error).to(self.device)
        #     heat_maps = self.render.render_shape(diff_verts, diff_trans_verts,colors=face_error_colors)
        
        # landmarks vis
        # lmk2d_vis = utils_visualize.tensor_vis_landmarks(gt_data['image'], diff_lmk2d[...,:2], gt_data['lmk_2d'])
        
        gt_img = gt_data['image'] * gt_data['img_mask'].unsqueeze(1)

        for i in range(deca_render_images.shape[0]):
            vis_dict = {
                'gt_img': gt_img[i].detach().cpu(),   # (3, h, w)
                'gt_mesh': emica_render_images[i].detach().cpu(),  # (3, h, w)
                'deca_mesh': deca_render_images[i].detach().cpu(),  # (3, h, w)
                'emoca_mesh': emoca_render_images[i].detach().cpu(),  # (3, h, w)
            }
            # if self.heatmap_view:
            #     vis_dict['heatmap'] = heat_maps[i].detach().cpu()
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
        emoca_codes,
        batch, 
    ):      
        global_rot_aa = batch['global_pose']
        
        emoca_exp = emoca_codes['exp_emoca']
        deca_exp = emoca_codes['exp']
        deca_jaw_aa = emoca_codes['pose'][...,3:]
        deca_pose_aa = torch.cat([global_rot_aa, deca_jaw_aa], dim=-1)

        gt_jaw_aa = batch['jaw']
        gt_expr = batch['exp']
        gt_rot_aa = torch.cat([global_rot_aa, gt_jaw_aa], dim=-1)
        
        # flame decoder
        verts_gt, lmk_3d_gt = self.flame(batch['shape'], gt_expr, gt_rot_aa)
        
        #===== eval for deca ========#
        verts_pred, lmk_3d_pred = self.flame(batch['shape'], deca_exp, deca_pose_aa)

        # 2d orthogonal projection
        lmk_2d_pred = batch_orth_proj(lmk_3d_pred, batch['cam'])[...,:2]
        lmk_2d_pred[:, :, 1:] = -lmk_2d_pred[:, :, 1:]

        # 2d orthogonal projection
        lmk_2d_emica = batch_orth_proj(lmk_3d_gt, batch['cam'])[...,:2]
        lmk_2d_emica[:, :, 1:] = -lmk_2d_emica[:, :, 1:]

        lmk_2d_gt = batch['lmk_2d'][:,self.flame.landmark_indices_mediapipe]
        lmk_mask_emb = batch['lmk_mask'][:, self.flame.landmark_indices_mediapipe]

        eval_log_deca = {}
        for metric in self.all_metrics:
            eval_log_deca[metric] = (
                get_metric_function(metric)(
                    deca_exp, deca_jaw_aa, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_emica,
                    gt_expr, gt_jaw_aa, verts_gt, lmk_3d_gt, lmk_2d_gt,
                    self.config.fps, self.flame_v_mask, lmk_mask_emb
                )
                .numpy()
            )
        
        #===== eval for emoca ========#
        verts_pred, lmk_3d_pred = self.flame(batch['shape'], emoca_exp, deca_pose_aa)

        # 2d orthogonal projection
        lmk_2d_pred = batch_orth_proj(lmk_3d_pred, batch['cam'])[...,:2]
        lmk_2d_pred[:, :, 1:] = -lmk_2d_pred[:, :, 1:]

        # 2d orthogonal projection
        lmk_2d_emica = batch_orth_proj(lmk_3d_gt, batch['cam'])[...,:2]
        lmk_2d_emica[:, :, 1:] = -lmk_2d_emica[:, :, 1:]

        lmk_2d_gt = batch['lmk_2d'][:,self.flame.landmark_indices_mediapipe]

        eval_log_emoca = {}
        for metric in self.all_metrics:
            eval_log_emoca[metric] = (
                get_metric_function(metric)(
                    deca_exp, deca_jaw_aa, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_emica,
                    gt_expr, gt_jaw_aa, verts_gt, lmk_3d_gt, lmk_2d_gt,
                    self.config.fps, self.flame_v_mask, lmk_mask_emb
                )
                .numpy()
            )
        
        return eval_log_deca, eval_log_emoca
            
    def track(self):
        
        # make prediction by split the whole sequences into chunks
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        eval_all_deca = defaultdict(float)
        eval_all_emoca = defaultdict(float)
        num_test_motions = len(self.test_data)
        eval_motion_num = 0
        for i in tqdm(range(num_test_motions)):
            self.flame.to(self.device)
            batch, motion_id = self.test_data[i]
            assert 'lmk_mask' in batch
            self.num_frames = batch['image'].shape[0]
            if self.num_frames < 25:
                logger.info(f'[{motion_id}] is shorter than 1 sec, skipped.')
                continue
            eval_motion_num += 1
            logger.info(f'Process [{motion_id}].')
            video_path = self.output_folder + f'/{motion_id}'
            
            # set the output writer
            if self.vis:
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
            
            
            # batch inference
            emoca_codes = None
            # if self.save_rec:
            #     # save inference results
            #     save_path = f"{self.sample_folder}/{motion_id}.npy"
            #     if os.path.exists(save_path):
            #         emoca_codes = np.load(save_path, allow_pickle=True)[()]
            #         for key in emoca_codes:
            #             emoca_codes[key] = torch.from_numpy(emoca_codes[key])

            if emoca_codes is None:
                emoca_codes = defaultdict(list)
                for start_id in range(0, self.num_frames, self.sld_wind_size):        
                    image_split = batch['image'][start_id:start_id+self.sld_wind_size].to(self.device)
                    img_mask_split = batch['img_mask'][start_id:start_id+self.sld_wind_size].to(self.device)
                    image_split =  image_split * img_mask_split.unsqueeze(1)
                    with torch.no_grad():
                        emoca_rec_split = self.emoca(image_split)
                    for k in emoca_rec_split:
                        emoca_codes[k].append(emoca_rec_split[k].cpu())

                for k in emoca_codes:
                    emoca_codes[k] = torch.cat(emoca_codes[k], dim=0)
                    # print(f"{k}: {emoca_codes[k].shape}")

            if self.vis:
                # batch visualiza all frames
                for i in range(0, self.num_frames, self.visualization_batch):
                    gt_data = {}
                    rec_data = {}
                    for key in batch:
                        gt_data[key] = batch[key][i:i+self.visualization_batch]
                        rec_data[key] = emoca_codes[key][i:i+self.visualization_batch]
                    self.vis_motion_split(gt_data, rec_data)

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
            eval_log_deca, eval_log_emoca = self.evaluate_one_motion(emoca_codes, batch)

            for metric in self.all_metrics:
                logger.info(f"{metric}_deca : {eval_log_deca[metric]}")
                eval_all_deca[metric] += eval_log_deca[metric]

                logger.info(f"{metric}_emoca : {eval_log_emoca[metric]}")
                eval_all_emoca[metric] += eval_log_emoca[metric]

            if self.save_rec:
                # save inference results
                save_path = f"{self.sample_folder}/{motion_id}.npy"
                Path(save_path).parent.mkdir(exist_ok=True, parents=True)
                for k in emoca_codes:
                    emoca_codes[k] = emoca_codes[k].numpy()
                np.save(save_path, emoca_codes)
            torch.cuda.empty_cache()

        logger.info("==========DECA Metrics for all test motion sequences:===========")
        for metric in eval_all_deca:
            logger.info(f"{metric} : {eval_all_deca[metric] / eval_motion_num}")
        
        logger.info("==========EMOCA Metrics for all test motion sequences:===========")
        for metric in eval_all_emoca:
            logger.info(f"{metric} : {eval_all_emoca[metric] / eval_motion_num}")

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="split in the dataset",
    )
    
    parser.add_argument(
        "--sld_wind_size",
        default=32,
        type=int,
        help="slide window size.",
    )
    
    parser.add_argument(
        "--save_folder",
        default="vis_result",
        type=str,
        help="folder to save visualization result.",
    )
    
    parser.add_argument(
        "--exp_name",
        type=str,
        help="name of the experiment.",
    )
    
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
        "--vis",
        action="store_true",
        help="whether to visualize the output.",
    )

    parser.add_argument(
        "--with_audio",
        action="store_true",
        help="whether the input with audio.",
    )

    parser.add_argument(
        "--save_rec",
        action="store_true",
        help="whether to store the diffusion reconstruction.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="fix to random seed.",
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="video fps.",
    )

    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="whether to assign mask path to it.",
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        default='MEAD',
        help="name of the test dataset.",
    )

    parser.add_argument('--dataset', type=str, default='mead_25fps', help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='./dataset',help='dataset name')

    args = parser.parse_args()
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
            pretrained_args.emoca.n_shape,
            pretrained_args.emoca.n_exp,
            args.exp_name,
            load_tex=args.vis,
            use_iris=False,
            load_audio_input=False,
            vis=True,
            mask_path = args.mask_path  # 
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
            use_iris=False,
            load_audio_input=False,
            vis=True,
            mask_path = args.mask_path
        )
    else:
        raise ValueError(f"{args.test_dataset} not supported!")

    motion_tracker = MotionTracker(args, pretrained_args.emoca, test_dataset, 'cuda')
    
    motion_tracker.track()

if __name__ == "__main__":
    main()
