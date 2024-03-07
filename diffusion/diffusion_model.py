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

import torch
import torch as th
from utils import utils_transform
from utils.famos_camera import batch_cam_to_img_project

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

def parse_model_target(target):
    nshape = 300
    nexp = 100 
    npose = 5*6 
    ntrans = 3 
    shape = target[:, 0, :nshape]
    exp = target[:, :, nshape:nshape+nexp]
    pose = target[:, :, nshape+nexp:-ntrans]
    trans = target[:, :, -ntrans:]
    return shape, exp, pose, trans

class DiffusionModel(GaussianDiffusion):
    def __init__(
        self,
        **kwargs,
    ):
        super(DiffusionModel, self).__init__(
            **kwargs,
        )
    
    # TODO: loss computation between target and prediction
    def masked_l2(self, target, model_output, **model_kwargs):

        bs, n, c = target.shape    

        # parse the output and target
        shape_gt, exp_gt, pose_gt, trans_gt = parse_model_target(target)
        shape_pred, exp_pred, pose_pred, trans_pred = parse_model_target(model_output)

        # l2 loss on flame parameters       
        shape_loss =  torch.mean(
            torch.norm((shape_gt - shape_pred), 2, -1))
        
        pose_loss = torch.mean(
            torch.norm((pose_gt - pose_pred), 2, -1))
        
        expr_loss = torch.mean(
            torch.norm((exp_gt - exp_pred), 2, -1))
        
        trans_loss = torch.mean(
            torch.norm((trans_gt - trans_pred), 2, -1))
        
        # jitter loss on rigid transformation (velocity match)
        pose_jitter = torch.mean(
            torch.norm(
                (pose_gt[:, 1:, :] - pose_gt[:, :-1, :]) - (pose_pred[:, 1:, :] - pose_pred[:, :-1, :]),
                2, 
                -1
            )
        )

        exp_jitter = torch.mean(
            torch.norm(
                (exp_gt[:, 1:, :] - exp_gt[:, :-1, :]) - (exp_pred[:, 1:, :] - exp_pred[:, :-1, :]),
                2, 
                -1
            )
        )
        
        # lmk verts loss 
        # denormalize thd flame params
        flame_params_pred = denormalize(
            model_output[:,:,:-3], model_kwargs['mean']['target'], model_kwargs['std']['target']
        ).reshape(bs*n, -1)

        flame_params_gt = denormalize(
            target[:,:,:-3], model_kwargs['mean']['target'], model_kwargs['std']['target']
        ).reshape(bs*n, -1)

        pose_aa_pred = utils_transform.sixd2aa(flame_params_pred[:, -30:].reshape(-1, 6)).reshape(bs*n, -1)
        pose_aa_gt = utils_transform.sixd2aa(flame_params_gt[:, -30:].reshape(-1, 6)).reshape(bs*n, -1)

        verts_pred, lmk3d_pred = self.flame(
            flame_params_pred[:, :300], flame_params_pred[:, 300:-30], pose_aa_pred)
        
        verts_gt, _ = self.flame(
            flame_params_gt[:, :300], flame_params_gt[:, 300:-30], pose_aa_gt)
        
        v_weights = self.flame_verts_weight.unsqueeze(1).to(target.device)  # (v, 1)
        verts_3d_dist = torch.norm(verts_pred.reshape(bs*n, -1, 3) - verts_gt.reshape(bs*n, -1, 3), 2, -1)    # (b, v)
        verts3d_loss = torch.mean(torch.matmul(verts_3d_dist, v_weights))
            
        # project 3d points to image plane 
        lmk2d_pred = batch_cam_to_img_project(
            points=lmk3d_pred,
            trans=trans_pred.reshape(-1, 3)
        ).reshape(-1, 2)

        # normalize lmk2d
        IMAGE_SIZE = 224
        lmk2d_pred_normed = lmk2d_pred / IMAGE_SIZE * 2 - 1
        
        lmk2d_loss = torch.mean(
            torch.norm(model_kwargs["lmk_2d"].reshape(-1, 2) - lmk2d_pred_normed, 2, -1))
            
        loss = 3.0 * shape_loss + 30.0 * pose_loss + 10.0 * expr_loss + 10 * trans_loss \
                + 0.01 * verts3d_loss + 0.01 * lmk2d_loss \
                + 0.0 * pose_jitter + 0.0 * exp_jitter

        loss_dict = {
            "loss": loss,
            "shape_loss": shape_loss,
            "expr_loss": expr_loss,
            "pose_loss": pose_loss,
            "trans_loss": trans_loss,
            "lmk2d_norm_loss": lmk2d_loss,
            "verts3d_loss": verts3d_loss,
            "expt_jitter": exp_jitter,
            "pose_jitter": pose_jitter,
        }

        return loss_dict

    def training_losses(
        self, model, x_start, t, model_kwargs=None, noise=None, dataset=None
    ):
        # print("model_kwargs = ", model_kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
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
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
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
