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
from model.spectre import SPECTRE

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class MotionTracker:
    
    def __init__(self, config, model_cfg, test_data, device='cuda'):
        
        self.config = config
        self.model_cfg = model_cfg
        self.device = device
        self.test_data = test_data
        self.original_data_folder = 'dataset/mead_25fps/original_data'
        self.vis = config.vis
        self.save_rec = config.save_rec
        # IO setups
                        
        # name of the tested motion sequence
        self.save_folder = self.config.save_folder
        self.output_folder = os.path.join(self.save_folder, 'SPECTRE', config.exp_name)
        
        if self.save_rec:
            self.sample_folder = os.path.join(self.output_folder, 'SPECTRE_reconstruction')
            if not os.path.exists(self.sample_folder):
                os.makedirs(self.sample_folder)
        
        logger.add(os.path.join(self.output_folder, 'test_mead_wrt_gt.log'))
        logger.info(f"Using device {self.device}.")

        # vis settings
        self.to_mp4 = True # if true then to mp4, false then to gif wo audio
        self.visualization_batch = 32
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
        self.load_spectre()
        
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

    def load_spectre(self):
        self.spectre = SPECTRE(self.model_cfg, self.device)
        self.spectre.eval()
    
    def output_video(self, fps=30):
        utils_visualize.images_to_video(self.output_folder, fps, self.motion_name)
    
    def vis_motion_split(self, gt_data, pred_codes):
    
        # to gpu
        for k in gt_data:
            gt_data[k] = gt_data[k].to(self.device)
        for k in pred_codes:
            pred_codes[k] = pred_codes[k].to(self.device)
        
        global_rot_aa = gt_data['global_pose']
        
        pred_exp = pred_codes['exp']
        pred_jaw_aa = pred_codes['pose'][3:]
        pred_pose_aa = torch.cat([global_rot_aa, pred_jaw_aa], dim=-1)

        gt_jaw_aa = gt_data['jaw']
        gt_exp = gt_data['exp']
        cam = gt_data['cam']
        gt_rot_aa = torch.cat([global_rot_aa, gt_jaw_aa], dim=-1)
        
        # flame decoder
        emica_verts, _ = self.flame(gt_data['shape'], gt_exp, gt_rot_aa)
        emica_trans_verts = batch_orth_proj(emica_verts, cam)
        emica_trans_verts[:, :, 1:] = -emica_trans_verts[:, :, 1:]

        # flame decoder for deca    
        pred_verts, pred_lmk3d = self.flame(gt_data['shape'], pred_exp, pred_pose_aa)
        pred_trans_verts = batch_orth_proj(pred_verts, cam)
        pred_trans_verts[:, :, 1:] = -pred_trans_verts[:, :, 1:]
        pred_lmk2d = batch_orth_proj(pred_lmk3d, cam)
        pred_lmk2d[:, :, 1:] = -pred_lmk2d[:, :, 1:]


        # flame decoder for emoca
        pred_verts, _ = self.flame(gt_data['shape'], pred_exp, pred_pose_aa)
        pred_trans_verts = batch_orth_proj(pred_verts, cam)
        pred_trans_verts[:, :, 1:] = -pred_trans_verts[:, :, 1:]   

        
        # # render
        pred_render_images = self.render.render_shape(pred_verts, pred_trans_verts, images=gt_data['image'])
        emica_render_images = self.render.render_shape(emica_verts, emica_trans_verts, images=gt_data['image'])
        # if self.heatmap_view:
        #     vertex_error = torch.norm(emica_verts - diff_verts, p=2, dim=-1) * 1000. # vertex dist in mm
        #     face_error_colors = self.get_vertex_error_heat_color(vertex_error).to(self.device)
        #     heat_maps = self.render.render_shape(diff_verts, diff_trans_verts,colors=face_error_colors)
        
        # landmarks vis
        lmk2d_vis = utils_visualize.tensor_vis_landmarks(gt_data['image'], pred_lmk2d[...,:2], gt_data['lmk_2d'])
        
        gt_img = gt_data['image'] * gt_data['img_mask'].unsqueeze(1)

        for i in range(gt_img.shape[0]):
            vis_dict = {
                'gt_img': gt_img[i].detach().cpu(),   # (3, h, w)
                'gt_mesh': emica_render_images[i].detach().cpu(),  # (3, h, w)
                'emoca_mesh': pred_render_images[i].detach().cpu(),  # (3, h, w)
                'lmk2d': lmk2d_vis[i].detach().cpu()
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
        code_dict,
        batch, 
    ):      
        global_rot_aa = batch['global_pose']
        
        spectre_exp = code_dict['exp']
        spectre_jaw_aa = code_dict['pose'][...,3:]
        spectre_pose_aa = torch.cat([global_rot_aa, spectre_jaw_aa], dim=-1)

        gt_jaw_aa = batch['jaw']
        gt_expr = batch['exp']
        gt_rot_aa = torch.cat([global_rot_aa, gt_jaw_aa], dim=-1)
        
        # flame decoder
        verts_gt, lmk_3d_gt = self.flame(batch['shape'], gt_expr, gt_rot_aa)
        
        #===== eval for spectre ========#
        verts_pred, lmk_3d_pred = self.flame(batch['shape'], spectre_exp, spectre_pose_aa)

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
                    spectre_exp, spectre_jaw_aa, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_emica,
                    gt_expr, gt_jaw_aa, verts_gt, lmk_3d_gt, lmk_2d_gt,
                    self.config.fps, self.flame_v_mask 
                )
                .numpy()
            )
        
        return eval_log
            
    def spectre_inference(self, batch):
        """ SPECTRE uses a temporal convolution of size 5. 
        Thus, in order to predict the parameters for a contiguous video with need to 
        process the video in chunks of overlap 2, dropping values which were computed from the 
        temporal kernel which uses pad 'same'. For the start and end of the video we
        pad using the first and last frame of the video. 
        e.g., consider a video of size 48 frames and we want to predict it in chunks of 20 frames 
        (due to memory limitations). We first pad the video two frames at the start and end using
        the first and last frames correspondingly, making the video 52 frames length.
        
        Then we process independently the following chunks:
        [[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
        [16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35]
        [32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51]]
        
        In the first chunk, after computing the 3DMM params we drop 0,1 and 18,19, since they were computed 
        from the temporal kernel with padding (we followed the same procedure in training and computed loss 
        only from valid outputs of the temporal kernel) In the second chunk, we drop 16,17 and 34,35, and in 
        the last chunk we drop 32,33 and 50,51. As a result we get:
        [2..17], [18..33], [34..49] (end included) which correspond to all frames of the original video 
        (removing the initial padding).     
        """
         # pad
        image = batch['image'].clone()
        img_mask = batch['img_mask']
        image =  image * img_mask.unsqueeze(1)  # (n, 3, 224, 224)
        image = torch.cat([
            image[:1].repeat(2, 1, 1, 1), image, image[-1:].repeat(2, 1, 1, 1)], dim=0) # (n+4, 3, 224, 224)
        L = 50 # chunk size

        # create lists of overlapping indices
        indices = list(range(image.shape[0]))
        overlapping_indices = [indices[i: i + L] for i in range(0, len(indices), L-4)]

        if len(overlapping_indices[-1]) < 5:
            # if the last chunk has less than 5 frames, pad it with the semilast frame
            overlapping_indices[-2] = overlapping_indices[-2] + overlapping_indices[-1]
            overlapping_indices[-2] = np.unique(overlapping_indices[-2]).tolist()
            overlapping_indices = overlapping_indices[:-1]

        overlapping_indices = np.array(overlapping_indices)
        code_dict_all = defaultdict(list)
        with torch.no_grad():
            for chunk_id in range(len(overlapping_indices)):
                print('Processing frames {} to {}'.format(overlapping_indices[chunk_id][0], overlapping_indices[chunk_id][-1]))
                image_chunk = image[overlapping_indices[chunk_id]].to(self.device) #K,3,224,224
                codedict = self.spectre.encode(image_chunk)
                for key in codedict.keys():
                    """ filter out invalid indices - see explanation at the top of the function """

                    if chunk_id == 0 and chunk_id == len(overlapping_indices) - 1:
                        pass
                    elif chunk_id == 0:
                        codedict[key] = codedict[key][:-2]
                    elif chunk_id == len(overlapping_indices) - 1:
                        codedict[key] = codedict[key][2:]
                    else:
                        codedict[key] = codedict[key][2:-2]
                    
                    code_dict_all[key].append(codedict[key])

        for k in code_dict_all:
            code_dict_all[k] = torch.cat(code_dict_all[k], dim=0).detach().cpu()
            code_dict_all[k] = code_dict_all[k][2:-2]
            # print(f"{k} {code_dict_all[k].shape}")
        return code_dict_all
    
    def track(self):
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')      
        num_test_motions = len(self.test_data)
        eval_motion_num = 0
        eval_all = defaultdict(float)
        for i in tqdm(range(num_test_motions)):
            self.flame.to(self.device)
            batch, motion_id = self.test_data[i]
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
            code_dict = None
            if self.save_rec:
                # save inference results
                save_path = f"{self.sample_folder}/{motion_id}.npy"
                if os.path.exists(save_path):
                    code_dict = np.load(save_path, allow_pickle=True)[()]
                    for key in code_dict:
                        code_dict[key] = torch.from_numpy(code_dict[key])
            if code_dict is None:
                code_dict = self.spectre_inference(batch)

            if self.vis:
                # batch visualiza all frames
                for i in range(0, self.num_frames, self.visualization_batch):
                    gt_data = {}
                    rec_data = {}
                    for key in batch:
                        gt_data[key] = batch[key][i:i+self.visualization_batch]
                        rec_data[key] = code_dict[key][i:i+self.visualization_batch]
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
            for k in batch:
                batch[k] = batch[k].to('cpu')
            for k in code_dict:
                code_dict[k] = code_dict[k].to('cpu')
            eval_log = self.evaluate_one_motion(code_dict, batch)

            for metric in self.all_metrics:
                logger.info(f"{metric} : {eval_log[metric]}")
                eval_all[metric] += eval_log[metric]

            if self.save_rec:
                # save inference results
                save_path = f"{self.sample_folder}/{motion_id}.npy"
                Path(save_path).parent.mkdir(exist_ok=True, parents=True)
                for k in code_dict:
                    code_dict[k] = code_dict[k].numpy()
                np.save(save_path, code_dict)
            torch.cuda.empty_cache()

        logger.info("==========SPECTRE Metrics for all test motion sequences:===========")
        for metric in eval_all:
            logger.info(f"{metric} : {eval_all[metric] / eval_motion_num}")

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="split in the dataset",
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
