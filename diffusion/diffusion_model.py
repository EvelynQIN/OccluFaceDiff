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
from model.FLAME import FLAME, FLAMETex
from model.deca import ExpressionLossNet
import numpy as np 
from utils.renderer import SRenderY
from skimage.io import imread
import torchvision.transforms.functional as F_v
import sys
sys.path.append("external/Visual_Speech_Recognition_for_Multiple_Languages")
from external.Visual_Speech_Recognition_for_Multiple_Languages.lipreading.model import Lipreading
from external.Visual_Speech_Recognition_for_Multiple_Languages.dataloader.transform import Compose, Normalize, CenterCrop, SpeedRate, Identity
from configparser import ConfigParser

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
    upper_mouth_lmk_ids = [49, 50, 51, 52, 53, 61, 62, 63]
    lower_mouth_lmk_ids = [59, 58, 57, 56, 55, 67, 66, 65]
    diff_pred = (pred_lmks[:, upper_mouth_lmk_ids, :2] - pred_lmks[:, lower_mouth_lmk_ids, :2])
    diff_gt = (gt_lmks[:, upper_mouth_lmk_ids, :2] - gt_lmks[:, lower_mouth_lmk_ids, :2])
    closure_loss = torch.mean(torch.abs(diff_gt - diff_pred).sum(2))
    return closure_loss

def eye_closure_lmk_loss(pred_lmks, gt_lmks):
    upper_eyelid_lmk_ids = [37, 38, 43, 44]
    lower_eyelid_lmk_ids = [41, 40, 47, 46]
    diff_pred = (pred_lmks[:, upper_eyelid_lmk_ids, :2] - pred_lmks[:, lower_eyelid_lmk_ids, :2])
    diff_gt = (gt_lmks[:, upper_eyelid_lmk_ids, :2] - gt_lmks[:, lower_eyelid_lmk_ids, :2])
    closure_loss = torch.mean(torch.abs(diff_gt - diff_pred).sum(2))

    return closure_loss

class DiffusionModel(GaussianDiffusion):
    def __init__(
        self,
        **kwargs,
    ):
        super(DiffusionModel, self).__init__(**kwargs,)

    def _setup(self):
        self._create_flame()
        self._setup_renderer()
        # self._load_evalnet()

    @profile
    def _create_flame(self):
        print(f"[Diffusion] Load FLAME.")
        self.flame = FLAME(self.model_cfg).to(self.device)
        self.flametex = FLAMETex(self.model_cfg).to(self.device)

    @profile
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
    
    @profile
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
        self._start_idx = 48
        self._stop_idx = 68
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)

        # ---- transform mouths before going into the lipread network for loss ---- #
        self.mouth_transform = Compose([
            Normalize(0.0, 1.0),
            CenterCrop(crop_size),
            Normalize(mean, std),
            Identity()]
        )
    
    def cut_mouth(self, images, landmarks, convert_grayscale=True):
        """ function adapted from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages"""

        mouth_sequence = []

        landmarks = landmarks * 112 + 112
        for frame_idx,frame in enumerate(images):
            window_margin = min(self._window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = landmarks[frame_idx-window_margin:frame_idx + window_margin + 1].mean(dim=0)
            smoothed_landmarks += landmarks[frame_idx].mean(dim=0) - smoothed_landmarks.mean(dim=0)

            center_x, center_y = torch.mean(smoothed_landmarks[self._start_idx:self._stop_idx], dim=0)

            center_x = center_x.round()
            center_y = center_y.round()

            height = self._crop_height//2
            width = self._crop_width//2

            threshold = 5

            if convert_grayscale:
                img = F_v.rgb_to_grayscale(frame).squeeze()
            else:
                img = frame

            if center_y - height < 0:
                center_y = height
            if center_y - height < 0 - threshold:
                raise Exception('too much bias in height')
            if center_x - width < 0:
                center_x = width
            if center_x - width < 0 - threshold:
                raise Exception('too much bias in width')

            if center_y + height > img.shape[-2]:
                center_y = img.shape[-2] - height
            if center_y + height > img.shape[-2] + threshold:
                raise Exception('too much bias in height')
            if center_x + width > img.shape[-1]:
                center_x = img.shape[-1] - width
            if center_x + width > img.shape[-1] + threshold:
                raise Exception('too much bias in width')

            mouth = img[...,int(center_y - height): int(center_y + height),
                        int(center_x - width): int(center_x + round(width))]

            mouth_sequence.append(mouth)
            
            del img

        mouth_sequence = torch.stack(mouth_sequence,dim=0)
        return mouth_sequence

    
    def batch_lmk2d_loss(self, lmk2d_pred, lmk2d_gt):
        """
        Computes the l1 loss between the ground truth keypoints and the predicted keypoints
        Inputs:
        lmk2d_gt  : N x K x 3
        lmk2d_pred: N x K x 2
        """
        weights = torch.ones((68,)).cuda()
        
        # contour
        weights[5:7] = 2
        weights[10:12] = 2
        
        # nose points
        weights[27:36] = 1.5
        weights[30] = 3
        weights[31] = 3
        weights[35] = 3
        
        # eye points
        weights[36:48] = 2

        # set mouth to zero
        weights[48:68] = 3
        lmk2d_gt[:,:,2] = weights[None,:] * lmk2d_gt[:,:,2]
        kp_gt = lmk2d_gt.view(-1, 3)
        kp_pred = lmk2d_pred.contiguous().view(-1, 2)
        vis = kp_gt[:, 2]
        k = torch.sum(vis) * 2.0 + 1e-8

        dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)

        return torch.matmul(dif_abs, vis) * 1.0 / k

    def batch_normalized_3d_closure_loss(self, closure_pred, closure_gt):
        """
        Computes the loss between the ground truth and pred 3d closure losses
        Inputs:
        closure_pred  : bs x v
        closure_gt: bs x v
        """
        closure_loss =  1 - torch.mean(F.cosine_similarity(closure_pred, closure_gt, dim=1))
        return closure_loss
        
    # loss computation between target and prediction
    # @profile
    def masked_l2(self, target, model_output, **model_kwargs):

        bs, n, c = target.shape    

        loss_dict = {}
        
        # parse the output and target
        jaw_pred, expr_pred = model_output[...,:6], model_output[...,6:]
        jaw_deca, expr_deca = target[...,:6], target[...,6:]

        # l2 loss on flame parameters w.r.t deca's output             
        pose_loss = F.mse_loss(jaw_pred, jaw_deca)
        expr_loss = F.mse_loss(expr_pred, expr_deca)
        
        # velocity loss
        pose_vel_loss = torch.mean(
            ((jaw_pred[:,1:] - jaw_pred[:,:-1]) - (jaw_deca[:,1:] - jaw_deca[:,:-1])) ** 2
        )
        expr_vel_loss = torch.mean(
            ((expr_pred[:,1:] - expr_pred[:,:-1]) - (expr_deca[:,1:] - expr_deca[:,:-1])) ** 2
        )
        
        # # jitter loss for temporal smoothness
        # pose_jitter = torch.mean((jaw_pred[:,2:] + jaw_pred[:,:-2] - jaw_pred[:,1:-1]) ** 2)
        # expr_jitter = torch.mean((expr_pred[:,2:] + expr_pred[:,:-2] - expr_pred[:,1:-1]) ** 2)
        
        # batch the output
        shape = model_kwargs['shape'].view(bs*n, -1)
        R_aa = model_kwargs['R'].view(bs*n, -1)
        expr_pred = expr_pred.reshape(bs*n, -1)
        jaw_pred_aa = utils_transform.sixd2aa(jaw_pred.reshape(-1, 6))
        pose_pred = torch.cat([R_aa, jaw_pred_aa], dim=-1)
        cam = model_kwargs['cam'].view(bs*n, -1)
        
        # flame decoder
        verts, landmarks2d, _ = self.flame(
            shape_params=shape, 
            expression_params=expr_pred,
            pose_params=pose_pred)
        
        # orthogonal projection
        landmarks2d = batch_orth_proj(landmarks2d, cam)[:, :, :2]
        landmarks2d[:, :, 1:] = -landmarks2d[:, :, 1:]
        trans_verts = batch_orth_proj(verts, cam)
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        # # rendering
        # images = model_kwargs['image'].view(bs*n, *model_kwargs['image'].shape[2:])
        # light = model_kwargs['light'].view(bs*n, *model_kwargs['light'].shape[2:])
        
        # # if not using texture, default to gray
        # if self.model_cfg.use_texture:
        #     tex = model_kwargs['tex'].view(bs*n, -1)
        #     albedo = self.flametex(tex).detach()
        # else:
        #     albedo = torch.ones([bs*n, 3, self.model_cfg.uv_size, self.model_cfg.uv_size], device=images.device) * 0.5

        # ops = self.render(verts, trans_verts, albedo, light)
        
        # calculate rendered images (shapes + texture)
        # mask_face_eye = F.grid_sample(self.uv_face_eye_mask.expand(bs*n,-1,-1,-1), ops['grid'].detach(), align_corners=False)
        
        ## --------loss computation -------- ##
        
        # ---- geometric losses ---- #
        lmk_gt = model_kwargs['lmk_2d'].view(bs*n, *model_kwargs['lmk_2d'].shape[2:])
        lmk2d_loss = self.batch_lmk2d_loss(landmarks2d, lmk_gt)  
        eye_closure_loss = eye_closure_lmk_loss(landmarks2d, lmk_gt)
        mouth_closure_loss = mouth_closure_lmk_loss(landmarks2d, lmk_gt)
        
        #  # ---- photometric loss ---- #
        # photometric_loss = torch.mean(mask_face_eye * ops['alpha_images'] * (ops['images'] - images).abs())
            
        
        # # # ---- emotion loss from EMOCA ---- #
        # with torch.no_grad():
        #     emotion_features_pred = self.expression_net(ops['images'])
        #     emotion_features_gt = self.expression_net(images)
        # emotion_loss = F.mse_loss(emotion_features_pred, emotion_features_gt)

        
        # # ---- lipread loss ---- #
        # # first crop the mouths of the input and rendered faces and then calculate the distance of features 
        # mouths_gt = self.cut_mouth(images, lmk_gt[...,:2])
        # mouths_pred = self.cut_mouth(ops['images'], landmarks2d[...,:2])
        # mouths_gt = self.mouth_transform(mouths_gt)
        # mouths_pred = self.mouth_transform(mouths_pred)

        # # # # ---- resize back to BxKx1xHxW (grayscale input for lipread net) ---- #
        # mouths_gt = mouths_gt.view(bs, n, mouths_gt.shape[-2], mouths_gt.shape[-1])
        # mouths_pred = mouths_pred.view(bs, n, mouths_gt.shape[-2], mouths_gt.shape[-1])

        # with torch.no_grad():
        #     lip_features_gt = self.lip_reader.model.encoder(
        #         mouths_gt,
        #         None,
        #         extract_resnet_feats=True
        #     )
            
        #     lip_features_pred = self.lip_reader.model.encoder(
        #         mouths_pred,
        #         None,
        #         extract_resnet_feats=True
        #     )

        # lipread_loss = F.mse_loss(lip_features_gt, lip_features_pred)
        
        loss = 0.2 * lmk2d_loss + 0.5 * mouth_closure_loss + 0.1 * eye_closure_loss \
            + 1.0 * expr_loss + 1.0 * pose_loss + 0.01 * expr_vel_loss + 0.01 * pose_vel_loss \
            + 0.001 * 0 + 0.5 * 0 + 0.1 * 0\
            
        
        loss_dict = {
            'expr_loss': expr_loss.detach().item(),
            'pose_loss': pose_loss.detach().item(),
            # 'expr_jitter': expr_jitter.detach().item(),
            # 'pose_jitter': pose_jitter.detach().item(),
            'expr_vel_loss': expr_vel_loss.detach().item(),
            'pose_vel_loss': pose_vel_loss.detach().item(),
            'lmk2d_loss': lmk2d_loss.detach().item(),
            'mouth_closure_loss': mouth_closure_loss.detach().item(),
            'eye_closure_loss': eye_closure_loss.detach().item(),
            # 'photometric_loss': photometric_loss.detach().item(),
            # 'emotion_loss': emotion_loss.detach().item(),
            # 'lipread_loss' : lipread_loss.detach().item(),
            'loss': loss
        }

        del ops

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
