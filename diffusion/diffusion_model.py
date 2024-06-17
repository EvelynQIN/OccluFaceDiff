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
from utils.data_util import batch_orth_proj
from model.FLAME import FLAME_mediapipe
import numpy as np 
from utils.MediaPipeLandmarkLists import *
from model.motion_prior import L2lVqVae
from munch import Munch, munchify

from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)


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
    
    def _load_flint_decoder(self):
        ckpt_path = self.model_cfg.flint_ckpt_path
        f = open(self.model_cfg.flint_config_path)
        cfg = Munch.fromYAML(f)
        flint = L2lVqVae(cfg)
        flint.load_model_from_checkpoint(ckpt_path)
        self.flint_decoder = flint.motion_decoder.to(self.device)
        print(f"[FLINT Decoder] Load and Frozen.")
        self.flint_decoder.requires_grad_(False)
        self.flint_decoder.eval()
    
    def _setup(self):
        self._create_flame()
        self._load_flint_decoder()

        # set up loss weight
        self.loss_weight = {
            'expr_loss': 0.,
            'pose_loss': 0.,
            'latent_rec_loss': 1.,
            'mesh_verts_loss': 100.,
            'lmk3d_loss': 0.5,
            'lmk2d_loss': 0.5,
            'mouth_closure_loss': 0.01,
        }
        print(f"[Diffusion] Loss weights used: {self.loss_weight}")

    def _create_flame(self):
        print(f"[Diffusion] Load FLAME.")
        self.flame = FLAME_mediapipe(self.model_cfg).to(self.device)
    
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

        loss_dict = {}
        
        latent_rec_loss = F.mse_loss(target, model_output)
        # flint decode
        pred_flame = self.flint_decoder(model_output)   # (bs, n, 53)
        bs, n = pred_flame.shape[:2]
        
        # parse the output and target
        jaw_pred, expr_pred = pred_flame[...,self.model_cfg.n_exp:], pred_flame[...,:self.model_cfg.n_exp]

         # l2 loss on flame parameters w.r.t deca's output             
        pose_loss = F.mse_loss(jaw_pred, model_kwargs['jaw'])
        expr_loss = F.mse_loss(expr_pred, model_kwargs['exp'])

        # # velocity loss
        # pose_vel_loss = torch.mean(
        #     ((jaw_pred[:,1:] - jaw_pred[:,:-1]) - (jaw_gt[:,1:] - jaw_gt[:,:-1])) ** 2
        # )
        # expr_vel_loss = torch.mean(
        #     ((expr_pred[:,1:] - expr_pred[:,:-1]) - (expr_gt[:,1:] - expr_gt[:,:-1])) ** 2
        # )
        
        # # jitter loss for temporal smoothness
        # pose_jitter = torch.mean((jaw_pred[:,2:] + jaw_pred[:,:-2] - jaw_pred[:,1:-1]) ** 2)
        # expr_jitter = torch.mean((expr_pred[:,2:] + expr_pred[:,:-2] - expr_pred[:,1:-1]) ** 2)
        
        # batch the output
        shape = model_kwargs['shape'].view(bs*n, -1)
        R_aa = model_kwargs['global_pose'].view(bs*n, -1)
        expr_pred = expr_pred.reshape(bs*n, -1)
        pose_pred = torch.cat([R_aa, jaw_pred.reshape(bs*n, -1)], dim=-1)
        cam = model_kwargs['cam'].view(bs*n, -1)

        expr_gt = model_kwargs['exp'].reshape(bs*n, -1)
        pose_gt = torch.cat([R_aa, model_kwargs['jaw'].reshape(bs*n, -1)], dim=-1)
        
        # flame decoder
        verts_pred, lmk3d_pred = self.flame(
            shape_params=shape, 
            expression_params=expr_pred,
            pose_params=pose_pred)
        
        verts_gt, lmk3d_gt = self.flame(
            shape_params=shape, 
            expression_params=expr_gt,
            pose_params=pose_gt)
        
        # l2 on mesh verts
        mesh_verts_loss = torch.mean(
            torch.norm(verts_gt - verts_pred, p=2, dim=-1)
        )

        # orthogonal projection
        lmk_pred = batch_orth_proj(lmk3d_pred, cam)[:, :, :2]
        lmk_pred[:, :, 1:] = -lmk_pred[:, :, 1:]

        # ---- geometric losses ---- #
        lmk_gt = model_kwargs['lmk_2d'][:,:,self.flame.landmark_indices_mediapipe].view(bs*n, *lmk_pred.shape[-2:])
        lmk2d_loss = self.batch_lmk2d_loss(lmk_pred, lmk_gt)  
        lmk3d_loss = torch.mean(torch.norm(lmk3d_pred - lmk3d_gt, p=2, dim=-1))
        # eye_closure_loss = eye_closure_lmk_loss(lmk_pred, lmk_gt)
        mouth_closure_loss = mouth_closure_lmk_loss(lmk_pred, lmk_gt)

        loss = self.loss_weight['lmk2d_loss'] * lmk2d_loss + self.loss_weight['mouth_closure_loss'] * mouth_closure_loss \
            + self.loss_weight['expr_loss'] * expr_loss + self.loss_weight['pose_loss'] * pose_loss \
            + self.loss_weight['lmk3d_loss'] * lmk3d_loss \
            + self.loss_weight['latent_rec_loss'] * latent_rec_loss + self.loss_weight['mesh_verts_loss'] * mesh_verts_loss \
        # + self.loss_weight['expr_vel_loss'] * expr_vel_loss + self.loss_weight['pose_vel_loss'] * pose_vel_loss \
        
        loss_dict = {
            'expr_loss': expr_loss.detach().item(),
            'pose_loss': pose_loss.detach().item(),
            'latent_rec_loss': latent_rec_loss.detach().item(),
            'mesh_verts_loss': mesh_verts_loss.detach().item(),          
            'lmk3d_loss': lmk3d_loss.detach().item(),
            'lmk2d_loss': lmk2d_loss.detach().item(),
            'mouth_closure_loss': mouth_closure_loss.detach().item(),
            'loss': loss
        }

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
