"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
# MIT License
# Copyright (c) 2021 OpenAI
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
from memory_profiler import profile
import torch
import torch.nn.functional as F
from utils import utils_transform
from utils.data_util import batch_orth_proj
from model.FLAME import FLAME, FLAMETex, FLAME_mediapipe
from model.deca import ExpressionLossNet
import numpy as np 
from utils.renderer import SRenderY
from skimage.io import imread
import torchvision.transforms.functional as F_v
import pickle
import sys
sys.path.append("external/Visual_Speech_Recognition_for_Multiple_Languages")
from external.Visual_Speech_Recognition_for_Multiple_Languages.lipreading.model import Lipreading
from external.Visual_Speech_Recognition_for_Multiple_Languages.dataloader.transform import Compose, Normalize, CenterCrop, SpeedRate, Identity
from configparser import ConfigParser
from utils.MediaPipeLandmarkLists import *

from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)

def normalize(value, mean, std):
    return (value - mean) / (std + 1e-8)

def denormalize(value, mean, std):
    return  value * std + mean

def mouth_closure_lmk_loss(pred_lmks, gt_lmks):
    
    diff_pred = (pred_lmks[:, UPPER_LIP_EM, :] - pred_lmks[:, LOWER_LIP_EM, :]).abs().sum(2)
    diff_gt = (gt_lmks[:, UPPER_LIP_EM, :] - gt_lmks[:, LOWER_LIP_EM, :]).abs().sum(2)
    closure_loss = torch.mean(torch.abs(diff_gt - diff_pred))
    return closure_loss

def eye_closure_lmk_loss(pred_lmks, gt_lmks):
    diff_pred = (pred_lmks[:, UPPER_EYELIDS_EM, :2] - pred_lmks[:, LOWER_EYELIDS_EM, :2]).abs().sum(2)
    diff_gt = (gt_lmks[:, UPPER_EYELIDS_EM, :2] - gt_lmks[:, LOWER_EYELIDS_EM, :2]).abs().sum(2)
    closure_loss = torch.mean(torch.abs(diff_gt - diff_pred))
    return closure_loss


class DiffusionModel(GaussianDiffusion):
    def __init__(
        self,
        **kwargs,
    ):
        super(DiffusionModel, self).__init__(**kwargs,)
    
    def _setup(self):
        self.train_stage = self.model_cfg.train_stage
        print(f"[Diffusion] Set up training stage {self.train_stage}")

        # at train stage 1, only optimize over mesh verts
        if self.train_stage == 1:
            self._create_flame()

            # set up loss weight
            self.loss_weight = {
                'expr_loss': 1.0,
                'pose_loss': 1.0,
                'expr_vel_loss': 0.01,
                'pose_vel_loss': 0.01,
                'lmk3d_loss': 0.2,
                'lmk2d_loss': 0,
                'mouth_closure_loss': 0.5,
                'emotion_loss': 0,
                'lipread_loss': 0
            }
            print(f"[Diffusion] Loss weights used: {self.loss_weight}")
        
        # at train stage 2, optimize over rendering loss
        elif self.train_stage == 2:
            self._create_flame()
            self._setup_renderer()
            self._load_evalnet()

            # set up loss weight
            self.loss_weight = {
                'expr_loss': 1.0,
                'pose_loss': 1.0,
                'expr_vel_loss': 0.01,
                'pose_vel_loss': 0.01,
                'lmk3d_loss': 0.1,
                'lmk2d_loss': 0.2,
                'mouth_closure_loss': 0.5,
                'emotion_loss': 0.0005,
                'lipread_loss': 0.005
            }
            print(f"[Diffusion] Loss weights used: {self.loss_weight}")
        else:
            raise ValueError(f"Traing stage[ {self.train_stage}] not valid!")

    def _create_flame(self):
        print(f"[Diffusion] Load FLAME.")
        self.flame = FLAME_mediapipe(self.model_cfg).to(self.device)
        if self.train_stage == 2 and self.model_cfg.use_texture:
            self.flametex = FLAMETex(self.model_cfg).to(self.device)
        
        # flame_vmask_path = "flame_2020/FLAME_masks.pkl"
        # with open(flame_vmask_path, 'rb') as f:
        #     flame_v_mask = pickle.load(f, encoding="latin1")

        # self.lip_v_mask = torch.from_numpy(flame_v_mask['lips']).to(self.device)

    def _setup_renderer(self):
        print(f"[Diffusion] Set up Renderer.")
        self.render = SRenderY(
            self.model_cfg.image_size, 
            obj_filename=self.model_cfg.topology_path, 
            uv_size=self.model_cfg.uv_size).to(self.device)
        # face mask for rendering details
        mask = imread(self.model_cfg.face_eye_mask_path).astype(np.float32)/255. 
        mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)
        # mask = imread(self.model_cfg.face_mask_path).astype(np.float32)/255.
        # mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        # self.uv_face_mask = F.interpolate(mask, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)
        # # TODO: displacement correction
        # fixed_dis = np.load(self.model_cfg.fixed_displacement_path)
        # self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        # mean_texture = imread(self.model_cfg.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        # self.mean_texture = F.interpolate(mean_texture, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)
        # # dense mesh template, for save detail mesh
        # self.dense_template = np.load(self.model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()
    
    def _load_evalnet(self):
 
        # ----- load resnet trained from EMOCA https://github.com/radekd91/emoca for expression loss ----- #
        self.expression_net = ExpressionLossNet().to(self.device)
        self.emotion_checkpoint = torch.load(self.model_cfg.expression_net_path)['state_dict']
        self.emotion_checkpoint['linear.0.weight'] = self.emotion_checkpoint['linear.weight']
        self.emotion_checkpoint['linear.0.bias'] = self.emotion_checkpoint['linear.bias']

        print(f"[Diffusion] Load emotion net.")
        self.expression_net.load_state_dict(self.emotion_checkpoint, strict=False)
        self.expression_net.eval()
        for param in self.expression_net.parameters():
            param.requires_grad = False

        # ----- load lipreader network for lipread loss ----- #
        config = ConfigParser()

        config.read('configs/lipread_config.ini')
        self.lip_reader = Lipreading(
            config,
            device=self.device
            
        )
        self.lip_reader.eval()
        self.lip_reader.model.eval()
        for param in self.lip_reader.parameters():
            param.requires_grad = False

        # ---- initialize values for cropping the face around the mouth for lipread loss ---- #
        # ---- this code is borrowed from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages ---- #
        self._crop_width = 96
        self._crop_height = 96
        self._window_margin = 12
        self._lip_idx = torch.from_numpy(LIP_EM).long()
        # self._start_idx = 48
        # self._stop_idx = 68
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)

        # ---- transform mouths before going into the lipread network for loss ---- #
        self.mouth_transform = Compose([
            Normalize(0.0, 1.0),
            CenterCrop(crop_size),
            Normalize(mean, std),
            Identity()]
        )

    def cut_mouth_vectorized(
        self,
        images, 
        landmarks, 
        convert_grayscale=True
        ):
                
        with torch.no_grad():

            landmarks = landmarks * 112 + 112
            # #1) smooth the landmarks with temporal convolution
            # landmarks are of shape (T, 68, 2) 
            landmarks_t = landmarks.reshape(*landmarks.shape[:2], -1)   # （bs, t, 468x2）
            # make temporal dimension last 
            landmarks_t = landmarks_t.permute(0, 2, 1)  # (bs, 468x2, t)

            # smooth with temporal convolution
            temporal_filter = torch.ones(self._window_margin, device=images.device) / self._window_margin
            # pad the the landmarks 
            landmarks_t_padded = F.pad(landmarks_t, (self._window_margin // 2, self._window_margin // 2), mode='replicate')
            # convolve each channel separately with the temporal filter
            num_channels = landmarks_t.shape[1]
            smooth_landmarks_t = F.conv1d(landmarks_t_padded, 
                temporal_filter.unsqueeze(0).unsqueeze(0).expand(num_channels,1,temporal_filter.numel()), 
                groups=num_channels, padding='valid'
            )
            smooth_landmarks_t = smooth_landmarks_t[..., 0:landmarks_t.shape[-1]]

            # reshape back to the original shape 
            smooth_landmarks_t = smooth_landmarks_t.permute(0, 2, 1).view(landmarks.shape)
            smooth_landmarks_t = smooth_landmarks_t + landmarks.mean(dim=2, keepdims=True) - smooth_landmarks_t.mean(dim=2, keepdims=True)

            # #2) get the mouth landmarks
            mouth_landmarks_t = smooth_landmarks_t[..., self._lip_idx, :]
            
            # #3) get the mean of the mouth landmarks
            mouth_landmarks_mean_t = mouth_landmarks_t.mean(dim=-2, keepdims=True)
        
            # #4) get the center of the mouth
            center_x_t = mouth_landmarks_mean_t[..., 0]
            center_y_t = mouth_landmarks_mean_t[..., 1]

            # #5) use grid_sample to crop the mouth in every image 
            # create the grid
            height = self._crop_height//2
            width = self._crop_width//2

            # torch.arange(0, mouth_crop_width, device=images.device)

            grid = torch.stack(torch.meshgrid(torch.linspace(-height, height, self._crop_height).to(images.device) / 112,
                                            torch.linspace(-width, width, self._crop_width).to(images.device) / 112 ), 
                               dim=-1)
            grid = grid[..., [1, 0]]
            grid = grid.unsqueeze(0).unsqueeze(0).repeat(*images.shape[:2], 1, 1, 1)    # (bs, n, 1, 9, 2)

            # normalize the center to [-1, 1]
            center_x_t = (center_x_t - 112) / 112
            center_y_t = (center_y_t - 112) / 112

            center_xy =  torch.cat([center_x_t, center_y_t ], dim=-1).unsqueeze(-2).unsqueeze(-2)
            if center_xy.ndim != grid.ndim:
                center_xy = center_xy.unsqueeze(-2)
            assert grid.ndim == center_xy.ndim, f"grid and center_xy have different number of dimensions: {grid.ndim} and {center_xy.ndim}"
            grid = grid + center_xy
        B, T = images.shape[:2]
        images = images.view(B*T, *images.shape[2:])
        grid = grid.view(B*T, *grid.shape[2:])

        if convert_grayscale: 
            images = F_v.rgb_to_grayscale(images)
        
            image_crops = F.grid_sample(
                images, 
                grid,  
                align_corners=True, 
                padding_mode='zeros',
                mode='bicubic'
            )
        
        # image_crops = image_crops.view(B, T, *image_crops.shape[1:])

        # if convert_grayscale:
        #     image_crops = image_crops#.squeeze(1)
        return image_crops.squeeze(1)   # (bs*t, 96, 96)
    
    def batch_lmk2d_loss(self, lmk2d_pred, lmk2d_gt):
        """
        Computes the l1 loss between the ground truth keypoints and the predicted keypoints
        Inputs:
        lmk2d_gt  : N x K x 2
        lmk2d_pred: N x K x 2
        """
        kp_gt = lmk2d_gt.view(-1, 2)
        kp_pred = lmk2d_pred.contiguous().view(-1, 2)

        diff_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)

        return diff_abs.mean()
        
    # loss computation between target and prediction where 2d images are available
    # @profile
    def masked_l2(self, target, model_output, **model_kwargs):

        bs, n, c = target.shape    

        loss_dict = {}
        
        # parse the output and target
        jaw_pred, expr_pred = model_output[...,:6], model_output[...,6:]
        jaw_gt, expr_gt = target[...,:6], target[...,6:]

        # l2 loss on flame parameters w.r.t deca's output             
        pose_loss = F.mse_loss(jaw_pred, jaw_gt)
        expr_loss = F.mse_loss(expr_pred, expr_gt)
        
        # velocity loss
        pose_vel_loss = torch.mean(
            ((jaw_pred[:,1:] - jaw_pred[:,:-1]) - (jaw_gt[:,1:] - jaw_gt[:,:-1])) ** 2
        )
        expr_vel_loss = torch.mean(
            ((expr_pred[:,1:] - expr_pred[:,:-1]) - (expr_gt[:,1:] - expr_gt[:,:-1])) ** 2
        )
        
        # # jitter loss for temporal smoothness
        # pose_jitter = torch.mean((jaw_pred[:,2:] + jaw_pred[:,:-2] - jaw_pred[:,1:-1]) ** 2)
        # expr_jitter = torch.mean((expr_pred[:,2:] + expr_pred[:,:-2] - expr_pred[:,1:-1]) ** 2)
        
        # batch the output
        shape = model_kwargs['shape'].view(bs*n, -1)
        R_aa = model_kwargs['global_pose'].view(bs*n, -1)
        expr_pred = expr_pred.reshape(bs*n, -1)
        jaw_pred_aa = utils_transform.sixd2aa(jaw_pred.reshape(-1, 6))
        pose_pred = torch.cat([R_aa, jaw_pred_aa], dim=-1)
        cam = model_kwargs['cam'].view(bs*n, -1)

        expr_gt = expr_gt.reshape(bs*n, -1)
        jaw_gt_aa = utils_transform.sixd2aa(jaw_gt.reshape(-1, 6))
        pose_gt = torch.cat([R_aa, jaw_gt_aa], dim=-1)
        
        # flame decoder
        verts_pred, lmk3d_pred = self.flame(
            shape_params=shape, 
            expression_params=expr_pred,
            pose_params=pose_pred)
        
        _, lmk3d_gt = self.flame(
            shape_params=shape, 
            expression_params=expr_gt,
            pose_params=pose_gt)
        
        # orthogonal projection
        lmk_pred = batch_orth_proj(lmk3d_pred, cam)[:, :, :2]
        lmk_pred[:, :, 1:] = -lmk_pred[:, :, 1:]

        # ---- geometric losses ---- #
        lmk_gt = model_kwargs['lmk_2d'][:,:,self.flame.landmark_indices_mediapipe].view(bs*n, *lmk_pred.shape[-2:])
        lmk2d_loss = self.batch_lmk2d_loss(lmk_pred, lmk_gt)  
        lmk3d_loss = torch.mean(torch.norm(lmk3d_pred - lmk3d_gt, p=2, dim=-1))
        # eye_closure_loss = eye_closure_lmk_loss(lmk_pred, lmk_gt)
        mouth_closure_loss = mouth_closure_lmk_loss(lmk_pred, lmk_gt)

        emotion_loss = 0
        lipread_loss = 0

        if self.train_stage == 2:

            trans_verts = batch_orth_proj(verts_pred, cam)
            trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

            # rendering
            light = model_kwargs['light'].view(bs*n, 9, 3)
            
            # if not using texture, default to gray
            if self.model_cfg.use_texture:
                tex = model_kwargs['tex'].view(bs*n, -1)
                albedo = self.flametex(tex).detach()
            else:
                albedo = torch.ones([bs*n, 3, self.model_cfg.uv_size, self.model_cfg.uv_size], device=light.device) * 0.5

            images = model_kwargs['image'].view(bs*n, *model_kwargs['image'].shape[2:])

            pred_render_img = self.render(verts_pred, trans_verts, albedo, light)['images']
            
            #  # ---- photometric loss ---- #
            # mask_face_eye = F.grid_sample(self.uv_face_eye_mask.expand(bs*n,-1,-1,-1), ops['grid'].detach(), align_corners=False)
            # photometric_loss = torch.mean(mask_face_eye * ops['alpha_images'] * (ops['images'] - images).abs())
            
        
            # # ---- emotion loss from EMOCA ---- #
            with torch.no_grad():
                emotion_features_gt = self.expression_net(images)
            emotion_features_pred = self.expression_net(pred_render_img)

            emotion_loss = F.mse_loss(emotion_features_pred, emotion_features_gt)

        
            # ---- lipread loss ---- #
            # first crop the mouths of the input and rendered faces and then calculate the distance of features 
            images = images.view(bs,n, *images.shape[1:])
            lmk_gt = lmk_gt.view(bs, n, *lmk_gt.shape[1:])
            lmk_pred = lmk_pred.view(bs, n, *lmk_pred.shape[1:])
            pred_render_img = pred_render_img.view(bs, n, *pred_render_img.shape[1:])
            mouths_gt = self.cut_mouth_vectorized(images, lmk_gt[...,:2])   # (bs*t, 96, 96)
            mouths_pred = self.cut_mouth_vectorized(pred_render_img, lmk_pred[...,:2])
            mouths_gt = self.mouth_transform(mouths_gt) # (bs*t, 88, 88)
            mouths_pred = self.mouth_transform(mouths_pred)

            # # # ---- resize back to BxNx1xHxW (grayscale input for lipread net) ---- #
            mouths_gt = mouths_gt.view(bs, n, *mouths_gt.shape[1:]) # (bs, n, 88, 88)
            mouths_pred = mouths_pred.view(bs, n, *mouths_pred.shape[1:]) # (bs, n, 88, 88)

            with torch.no_grad():
                lip_features_gt = self.lip_reader.model.encoder(
                    mouths_gt,
                    None,
                    extract_resnet_feats=True
                )
                
            lip_features_pred = self.lip_reader.model.encoder(
                mouths_pred,
                None,
                extract_resnet_feats=True
            )

            lipread_loss = F.mse_loss(lip_features_gt, lip_features_pred)
        loss = self.loss_weight['lmk2d_loss'] * lmk2d_loss + self.loss_weight['mouth_closure_loss'] * mouth_closure_loss \
            + self.loss_weight['expr_loss'] * expr_loss + self.loss_weight['pose_loss'] * pose_loss \
            + self.loss_weight['expr_vel_loss'] * expr_vel_loss + self.loss_weight['pose_vel_loss'] * pose_vel_loss \
            + self.loss_weight['lmk3d_loss'] * lmk3d_loss \
            + self.loss_weight['emotion_loss'] * emotion_loss + self.loss_weight['lipread_loss'] * lipread_loss
        loss_dict = {
            'expr_loss': expr_loss.detach().item(),
            'pose_loss': pose_loss.detach().item(),
            'expr_vel_loss': expr_vel_loss.detach().item(),
            'pose_vel_loss': pose_vel_loss.detach().item(),
            'lmk3d_loss': lmk3d_loss.detach().item(),
            'lmk2d_loss': lmk2d_loss.detach().item(),
            'mouth_closure_loss': mouth_closure_loss.detach().item(),
            # 'eye_closure_loss': eye_closure_loss.detach().item(),
            # 'photometric_loss': photometric_loss.detach().item(),
            'loss': loss
        }

        if self.train_stage == 2:
            loss_dict.update({
                'emotion_loss': emotion_loss.detach().item(),
                'lipread_loss' : lipread_loss.detach().item(),})

        return loss_dict

    def training_losses(
        self, model, x_start, t, model_kwargs=None, noise=None, dataset=None
    ):
        # print("model_kwargs = ", model_kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:  
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            assert model_output.shape == target.shape == x_start.shape

            
            terms.update(self.masked_l2(
                target,
                model_output,
                **model_kwargs
            ))

            terms["loss"] += terms.get("vb", 0.0)


        else:
            raise NotImplementedError(self.loss_type)

        return terms
