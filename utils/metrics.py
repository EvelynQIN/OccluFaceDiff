# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Metric functions with same inputs

import numpy as np
import torch
import math
from utils.MediaPipeLandmarkLists import *

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)  # 57.2958 grads
METERS_TO_MILLIMETERS = 1000.0

def pred_jitter(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask
):
    pred_jitter = (
        (
            (
                verts_pred[3:]
                - 3 * verts_pred[2:-1]
                + 3 * verts_pred[1:-2]
                - verts_pred[:-3]
            )
            * (fps**3)
        )
        .norm(dim=2)
        .mean()
    )
    return pred_jitter


def gt_jitter(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask
):
    gt_jitter = (
        (
            (
                verts_gt[3:]
                - 3 * verts_gt[2:-1]
                + 3 * verts_gt[1:-2]
                - verts_gt[:-3]
            )
            * (fps**3)
        )
        .norm(dim=2)
        .mean()
    )
    return gt_jitter 


def pose_error(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask
):
    diff = pose_aa_gt - pose_aa_pred
    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    rot_error = torch.mean(torch.absolute(diff))
    return rot_error * RADIANS_TO_DEGREES


def expre_error(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask 
):
    mean_expression_error = torch.mean(
        torch.abs(expr_pred - expr_gt)
    )
    return mean_expression_error

def lmk_3d_mvpe(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask
):
    lmk_3d_mean_vertex_position_error_max = torch.max(
        torch.norm(lmk_3d_gt - lmk_3d_pred,p=2,dim=-1),
        dim=0
    ).values

    lmk_3d_vpe_mean = torch.mean(lmk_3d_mean_vertex_position_error_max)

    return lmk_3d_vpe_mean * METERS_TO_MILLIMETERS

def mvpe(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask
):
    mean_vertex_pos_error = torch.mean(
        torch.norm(
            verts_gt.reshape(-1, 3) - verts_pred.reshape(-1, 3),
            2,
            1
        )
    )
    return mean_vertex_pos_error * METERS_TO_MILLIMETERS

def mvpe_face(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask 
):
    verts_gt_face = torch.index_select(verts_gt, 1, flame_v_mask['face'])
    verts_pred_face = torch.index_select(verts_pred, 1, flame_v_mask['face'])
    mean_vertex_pos_error = torch.mean(
        torch.norm(
            verts_gt_face.reshape(-1, 3) - verts_pred_face.reshape(-1, 3),
            2,
            1
        )
    )
    return mean_vertex_pos_error * METERS_TO_MILLIMETERS

def lve(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask
):
    verts_gt_lips = torch.index_select(verts_gt, 1, flame_v_mask['lips'])
    verts_pred_lips = torch.index_select(verts_pred, 1, flame_v_mask['lips'])

    lip_dist_max = torch.max(
        torch.norm(verts_gt_lips - verts_pred_lips, p=2, dim=-1),
        dim=0
    ).values

    mean_lip_dist = torch.mean(lip_dist_max)
    return mean_lip_dist * METERS_TO_MILLIMETERS

def mvpe_eye_region(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask   
):
    verts_gt_eyes = torch.index_select(verts_gt, 1, flame_v_mask['eye_region'])
    verts_pred_eyes = torch.index_select(verts_pred, 1, flame_v_mask['eye_region'])
    mean_vertex_pos_error = torch.mean(
        torch.norm(
            verts_gt_eyes.reshape(-1, 3) - verts_pred_eyes.reshape(-1, 3),
            2,
            1
        )
    )
    return mean_vertex_pos_error * METERS_TO_MILLIMETERS

def mvpe_forehead(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask 
):
    verts_gt_forehead = torch.index_select(verts_gt, 1, flame_v_mask['forehead'])
    verts_pred_forehead = torch.index_select(verts_pred, 1, flame_v_mask['forehead'])
    mean_vertex_pos_error = torch.mean(
        torch.norm(
            verts_gt_forehead.reshape(-1, 3) - verts_pred_forehead.reshape(-1, 3),
            2,
            1
        )
    )
    return mean_vertex_pos_error * METERS_TO_MILLIMETERS

def mvpe_neck(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask    
):
    verts_gt_neck = torch.index_select(verts_gt, 1, flame_v_mask['neck'])
    verts_pred_neck = torch.index_select(verts_pred, 1, flame_v_mask['neck'])
    mean_vertex_pos_error = torch.mean(
        torch.norm(
            verts_gt_neck.reshape(-1, 3) - verts_pred_neck.reshape(-1, 3),
            2,
            1
        )
    )
    return mean_vertex_pos_error * METERS_TO_MILLIMETERS

def mvpe_nose(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask
):
    verts_gt_nose = torch.index_select(verts_gt, 1, flame_v_mask['nose'])
    verts_pred_nose = torch.index_select(verts_pred, 1, flame_v_mask['nose'])
    mean_vertex_pos_error = torch.mean(
        torch.norm(
            verts_gt_nose.reshape(-1, 3) - verts_pred_nose.reshape(-1, 3),
            2,
            1
        )
    )
    return mean_vertex_pos_error * METERS_TO_MILLIMETERS

def mvve(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask   
):
    gt_velocity = (verts_gt[1:, ...] - verts_gt[:-1, ...]) * fps
    pred_velocity = (verts_pred[1:, ...] - verts_pred[:-1, ...]) * fps
    vel_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_velocity - pred_velocity), axis=-1))
    )
    return vel_error * METERS_TO_MILLIMETERS

def mean_full_vertex_error(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask 
):
    full_vertex_pos_error = torch.mean(
        torch.norm(
            verts_gt - verts_pred,
            2,
            2
        ),
        dim=0
    )
    return full_vertex_pos_error * METERS_TO_MILLIMETERS

def lmk2d_mouth_closure_error(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask 
):
    diff_pred = torch.norm(lmk_2d_pred[:, UPPER_LIP_EM, :] - lmk_2d_pred[:, LOWER_LIP_EM, :], p=1, dim=-1)
    diff_gt = torch.norm(lmk_2d_gt[:, UPPER_LIP_EM, :] - lmk_2d_gt[:, LOWER_LIP_EM, :], p=1, dim=-1)
    mouth_closure_error = torch.mean(torch.abs(diff_gt - diff_pred))
    return mouth_closure_error

def gt_lmk2d_mouth_closure_error(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask 
):
    pgt_pred = torch.norm(lmk_2d_pgt[:, UPPER_LIP_EM, :] - lmk_2d_pgt[:, LOWER_LIP_EM, :], p=1, dim=-1)
    diff_gt = torch.norm(lmk_2d_gt[:, UPPER_LIP_EM, :] - lmk_2d_gt[:, LOWER_LIP_EM, :], p=1, dim=-1)
    mouth_closure_error = torch.mean(torch.abs(diff_gt - pgt_pred))
    return mouth_closure_error

def lmk2d_reproj_error(
    expr_pred, pose_aa_pred, verts_pred, lmk_3d_pred, lmk_2d_pred, lmk_2d_pgt,
    expr_gt, pose_aa_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
    fps, flame_v_mask 
):
    lmk2d_error = torch.mean(
        torch.norm(lmk_2d_gt - lmk_2d_pred, p=1, dim=-1))
    return lmk2d_error

metric_funcs_dict = {
    "pred_jitter": pred_jitter,
    "gt_jitter": gt_jitter,
    "mvpe": mvpe,
    "mvve": mvve,
    "pose_error": pose_error,
    "expre_error": expre_error,
    "lmk_3d_mvpe": lmk_3d_mvpe,
    "mean_full_vertex_error": mean_full_vertex_error,
    "mvpe_face":mvpe_face,
    "mvpe_eye_region": mvpe_eye_region,
    "mvpe_forehead": mvpe_forehead,
    "lve": lve,
    "mvpe_neck": mvpe_neck,
    "mvpe_nose": mvpe_nose,
    "mouth_closure": lmk2d_mouth_closure_error,
    "gt_mouth_closure": gt_lmk2d_mouth_closure_error,
    "lmk2d_reproj_error": lmk2d_reproj_error
}


def get_metric_function(metric):
    return metric_funcs_dict[metric]
