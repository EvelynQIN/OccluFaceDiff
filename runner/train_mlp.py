# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import torch
from utils import utils_transform
from utils.utils_visualize import mesh_sequence_to_video_frames

def update_lr_multistep(
    nb_iter, total_iter, max_lr, min_lr, optimizer, lr_anneal_steps
):
    if nb_iter > lr_anneal_steps:
        current_lr = min_lr
    else:
        current_lr = max_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def train_loss(model_output, target):

    expr_loss = torch.mean(
        torch.norm(
            (model_output - target),
            2,
            1
        )
    )
       
    loss_dict = {
        "expr_loss": expr_loss.item()
    }

    return expr_loss, loss_dict

def eval_loss(model_output, target, shape, flame, render_video):
    n, c = target.shape
    target = target.reshape(-1, c)

    trans_target = torch.zeros((n, 3)).to(target.device)
    pose_aa_target = torch.zeros((n, 3*5)).to(target.device)
    
    verts_pred, lmk_pred = flame(shape, model_output, pose_aa_target, trans_target)
    verts_gt, lmk_gt = flame(shape, target, pose_aa_target, trans_target)

    expr_loss = torch.mean(
        torch.norm(
            (target - model_output),
            2,
            1
        )
    )
    
    dist_verts = torch.mean(
        torch.norm(
            verts_gt.reshape(-1, 3) - verts_pred.reshape(-1, 3),
            2,
            1
        )
    )
    
    dist_lmk = torch.mean(
        torch.norm(
            lmk_gt.reshape(-1, 3) - lmk_pred.reshape(-1, 3),
            2,
            1
        )
    )

    loss_dict = {
        "expr_loss": expr_loss.item(),
        "verts": dist_verts.item(),
        "lmk3d": dist_lmk.item()
    }

    gt_video_frames, rec_video_frames = None, None
    if render_video:
        faces = flame.faces_tensor.numpy()
        gt_video_frames = mesh_sequence_to_video_frames(verts_gt.to('cpu').numpy(), faces, lmk_gt.cpu().numpy())
        rec_video_frames = mesh_sequence_to_video_frames(verts_pred.to('cpu').numpy(), faces, lmk_pred.cpu().numpy())
    return loss_dict, gt_video_frames, rec_video_frames 
    
def train_step(
    motion_input,   # lmk_3d
    motion_target,  # expression (100,)
    model,
    optimizer,
    nb_iter,
    total_iter,
    max_lr,
    min_lr,
    device,
    lr_anneal_steps,
):

    motion_input = motion_input.to(device)
    motion_target = motion_target.to(device)

    motion_pred = model(motion_input)

    loss, loss_dict = train_loss(motion_pred, motion_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(
        nb_iter, total_iter, max_lr, min_lr, optimizer, lr_anneal_steps
    )

    return loss_dict, optimizer, current_lr

def val_step(
    motion_input,
    motion_target,
    shape,
    model,
    flame,
    device,
    input_motion_length,
    render_video
):
    num_frames, motion_nfeats = motion_target.shape
    count = 0 
    flag_index = None
    
    motion_inputs, motion_targets, shapes = [], [], []
    # sliding window for long motion sequence (non-overlapping)
    while count + input_motion_length <= num_frames:
        motion_inputs.append(motion_input[count:count+input_motion_length])
        motion_targets.append(motion_target[count:count+input_motion_length])
        shapes.append(shape[count:count+input_motion_length])
        count += input_motion_length
    
    if count < num_frames:
        motion_inputs.append(motion_input[-input_motion_length :])
        motion_targets.append(motion_target[-input_motion_length :])
        shapes.append(shape[-input_motion_length :])
        flag_index = count - num_frames + input_motion_length

    motion_inputs = torch.stack(motion_inputs, dim=0)
    motion_targets = torch.stack(motion_targets, dim=1) 
    shapes = torch.stack(shapes, dim=0) 
    
    motion_inputs = motion_inputs.to(device)
    motion_targets = motion_targets.to(device)
    shapes = shapes.to(device)

    motion_preds = model(motion_inputs)

    if flag_index is not None:
        last_batch = motion_preds[-1][flag_index:]
        prev_batches = motion_preds[:-1].reshape(-1, motion_nfeats)
        motion_preds = torch.cat([prev_batches, last_batch], dim=0)
    else:
        motion_preds = motion_preds.reshape(-1, motion_nfeats)
    
    assert motion_preds.shape[0] == motion_input.shape[0], "motion pred not aligned with num_frames"

    loss_dict, gt_video_frames, rec_video_frames = eval_loss(motion_preds.cpu().float(), motion_target, shape, flame, render_video)
    

    return loss_dict, gt_video_frames, rec_video_frames
