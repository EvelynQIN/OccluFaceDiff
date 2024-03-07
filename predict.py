import math
import os
import random

import numpy as np

import torch

from data_loaders.dataloader import load_data, TestDataset

from model.FLAME import FLAME

from model.networks import PureMLP
from tqdm import tqdm
import time

from utils import utils_transform, utils_visualize
from utils.metrics import get_metric_function
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import predict_args

def load_diffusion_model(args):
    print("Creating model and diffusion...")
    args.arch = args.arch[len("diffusion_") :]
    model, diffusion = create_model_and_diffusion(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    model.to("cuda:0")  # dist_util.dev())
    model.eval()  # disable random masking
    return model, diffusion

def get_one_face_motion(motion_path):
    motion = torch.load(motion_path)
    lmk_3d = motion['lmk68_3d']
    trans = motion['flame_params']['root_trans']
    pose = motion['flame_params']["full_pose_6d"]
    shape = motion["flame_params"]["shape"]
    expression = motion["flame_params"]["expression"]
    verts = motion["mesh_verts"]

    return lmk_3d, trans, pose, shape, expression, verts

def compute_metrics(target, motion_pred, lmk_gt, shape, verts_gt, flame):
    bs, n, c = target.shape
    target = target.reshape(-1, c)
    motion_pred = motion_pred.reshape(-1, c)
    expr = motion_pred[:, 33:]
    trans = motion_pred[:, :3]
    pose_6d = motion_pred[:, 3: 33]
    shape = shape.reshape(bs*n, -1)

    trans_loss = torch.mean(
        torch.norm(
            (target[:, :3] - trans),
            2,
            1
        )
    )
    pose_loss = torch.mean(
        torch.norm(
            (target[:, 3:33] - pose_6d),
            2,
            1
        )
    )
    expr_loss = torch.mean(
        torch.norm(
            (target[:, 33:] - expr),
            2,
            1
        )
    )

    pose_aa = utils_transform.sixd2aa(pose_6d.reshape(-1, 6)).reshape(-1, 3*5)
    vert_pred, lmk_pred = flame(shape, expr, pose_aa, trans)
    
    dist_verts = torch.mean(
        torch.norm(
            verts_gt.reshape(-1, 3) - vert_pred.reshape(-1, 3),
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
    loss = trans_loss + pose_loss + expr_loss

    loss_dict = {
        "trans_loss": trans_loss.item(),
        "pose_loss": pose_loss.item(),
        "expr_loss": expr_loss.item(),
        "training_loss": loss.item(),
        "verts": dist_verts.item(),
        "lmk3d": dist_lmk.item()
    }

    return loss_dict, vert_pred

def test_model(args):
    model, diffusion = load_diffusion_model(args)
    print(model)

    input_motion_length = args.input_motion_length
    device = "cuda:0"
    flame = FLAME(args)

    start_time = time.time()

    lmk_3d, trans, pose, shape, expression, verts = get_one_face_motion(args.motion_path)
    motion_target = torch.concat((trans, pose, expression), dim = -1)    # (n, 133)
    num_frames = lmk_3d.shape[0]
    assert num_frames >= input_motion_length, "motion length is too short"

    # construct sparse input
    count = 0     
    motion_inputs, motion_targets, shapes, verts_list, start_frame_id = [], [], [], [], []
    
    # sliding window for long motion sequence (non-overlapping)
    while count + input_motion_length <= num_frames:
        start_frame_id.append(count)
        motion_inputs.append(lmk_3d[count:count+input_motion_length])
        motion_targets.append(motion_target[count:count+input_motion_length])
        shapes.append(shape[count:count+input_motion_length])
        verts_list.append(verts[count:count+input_motion_length])
        count += input_motion_length
        
    
    if count < num_frames:
        start_frame_id.append(num_frames-input_motion_length)
        motion_inputs.append(lmk_3d[-input_motion_length :])
        motion_targets.append(motion_target[-input_motion_length :])
        shapes.append(shape[-input_motion_length :])
        verts_list.append(verts[-input_motion_length :])

    motion_inputs = torch.stack(motion_inputs, dim=0)
    motion_targets = torch.stack(motion_targets, dim=0) 
    shapes = torch.stack(shapes, dim=0) 
    verts_list = torch.stack(verts_list, dim=0)   
    
    n_slice = motion_inputs.shape[0]
    motion_inputs = motion_inputs.reshape(n_slice, input_motion_length, -1).to(device)

    if args.fix_noise:
        # fix noise seed for every frame
        noise = torch.randn(1, 1, 1).cuda()
        noise = noise.repeat(n_slice, args.input_motion_length, args.motion_nfeat)
    else:
        noise = None


    motion_pred = diffusion.p_sample_loop(
                model,
                (n_slice, args.input_motion_length, args.motion_nfeat),
                sparse=motion_inputs,
                clip_denoised=False,
                model_kwargs=None,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=noise,
                const_noise=False,
            )

    motion_pred = motion_pred.detach().cpu()
    print("shape of predict motion", motion_pred.shape)

    elapsed = time.time() - start_time
    print("Inference time for ", motion_pred.shape[0], " slices is: ", elapsed, " seconds.")
    print("Inference time for 1 frame is: ", elapsed / (motion_pred.shape[0] * motion_pred.shape[1]), " seconds.")
    
    # compute the evaluation metrics
    loss_dict, verts_pred = compute_metrics(motion_targets, motion_pred, motion_inputs.to('cpu'), shapes, verts_list, flame)
    print("eval metrics: ")
    output_str = ""
    for k, v in loss_dict.items():
        output_str += f"{k} = {np.round(v, 4)} || "
    print(output_str)

    # render the video for the full motion
    motion_name = os.path.split(args.motion_path)[-1].split(".")[0]
    video_dir = os.path.join("vis_result", args.arch)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    video_path = os.path.join(video_dir, f"{motion_name}_pred.mp4")

    # remove the overlapping motion on the last slice  
    prev_end_id = start_frame_id[-2] + input_motion_length
    if prev_end_id > start_frame_id[-1]:
        overlap_len = prev_end_id - start_frame_id[-1]
        mesh_verts = np.concatenate((verts_pred[:prev_end_id], verts_pred[prev_end_id+overlap_len:]), axis=0)
    utils_visualize.mesh_sequence_to_video(mesh_verts, flame.faces_tensor.numpy(), video_path, args.fps)

    gt_video_path = os.path.join(video_dir, f"{motion_name}_gt.mp4")
    utils_visualize.mesh_sequence_to_video(verts, flame.faces_tensor.numpy(), gt_video_path, args.fps)
    print(f"saving videos to {video_dir}")

def main():
    
    args = predict_args()
    print(args)
    test_model(args)


if __name__ == "__main__":
    main()