# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import math
import os
import random
import pickle
import numpy as np
import trimesh
import torch

from data_loaders.dataloader import load_data, TestDataset

from model.FLAME import FLAME

from model.networks import PureMLP
from tqdm import tqdm

from utils import utils_transform, utils_visualize
from utils.metrics import get_metric_function
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import sample_args

device = torch.device("cuda")

pred_metrics = [
    "pred_jitter",
    "mvpe",
    "mvve",
    "expre_error",
    "pose_error",
    "lmk_3d_mvpe",
    "mvpe_face",
    "mvpe_eye_region",
    "mvpe_forehead",
    "mvpe_lips",
    "mvpe_neck",
    "mvpe_nose",
]
gt_metrics = [
    "gt_jitter",
]

full_vertex_metrics = [
    "mean_full_vertex_error"
]

all_metrics = pred_metrics + gt_metrics + full_vertex_metrics


def non_overlapping_test(
    args,
    data,
    sample_fn,
    dataset,
    model,
    model_type="mlp",
):
    motion_target, lmk_3d_gt, shape_gt, motion_id = data
    sparse_original = lmk_3d_gt.cuda().float()

    num_frames = motion_target.shape[0]

    count = 0
    sparse_splits = []
    flag_index = None

    if args.input_motion_length <= num_frames:
        while count < num_frames:
            if count + args.input_motion_length > num_frames:
                tmp_k = num_frames - args.input_motion_length
                sub_sparse = sparse_original[tmp_k : tmp_k + args.input_motion_length]
                flag_index = count - tmp_k
            else:
                sub_sparse = sparse_original[count : count + args.input_motion_length]
            sparse_splits.append(sub_sparse)
            count += args.input_motion_length
    else:
        flag_index = args.input_motion_length - num_frames
        tmp_init = sparse_original[0].repeat(flag_index, 1).clone()
        sub_sparse = torch.concat([tmp_init, sparse_original], dim=0)
        sparse_splits = [sub_sparse]

    bs = len(sparse_splits)
    sparse_batch = torch.stack(sparse_splits)

    if args.fix_noise:
        # fix noise seed for every frame
        noise = torch.randn(1, 1, 1).cuda()
        noise = noise.repeat(bs, args.input_motion_length, args.motion_nfeat)
    else:
        noise = None

    if model_type == "diffusion":
        sample = sample_fn(
            model,
            (bs, args.input_motion_length, args.motion_nfeat),
            sparse=sparse_batch,
            clip_denoised=False,
            model_kwargs=None,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=noise,
            const_noise=False,
        )
    elif model_type == "mlp":
        sample = model(sparse_batch)

    if flag_index is not None:
        last_batch = sample[-1]
        last_batch = last_batch[flag_index:]
        sample = sample[:-1].reshape(-1, args.motion_nfeat)
        sample = torch.cat([sample, last_batch], dim=0)
    else:
        sample = sample.reshape(-1, args.motion_nfeat)

    if not args.no_normalization:
        output_sample = dataset.inv_transform(sample.cpu().float())
        motion_target = dataset.inv_transform(motion_target)
    else:
        output_sample = sample.cpu().float()

    return output_sample, motion_target, shape_gt, motion_id


def overlapping_test(
    args,
    data,
    sample_fn,
    dataset,
    model,
    model_type="diffusion",
):
    assert (
        model_type == "diffusion"
    ), "currently only diffusion model supports overlapping test!!!"

    sld_wind_size = args.sld_wind_size

    motion_target, lmk_3d_gt, shape_gt, motion_id = data
    sparse_original = lmk_3d_gt.cuda().float()

    num_frames = motion_target.shape[0]

    output_samples = []
    count = 0
    sparse_splits = []
    flag_index = None

    if num_frames < args.input_motion_length:
        flag_index = args.input_motion_length - num_frames
        tmp_init = sparse_original[0].repeat(flag_index, 1).clone()
        sub_sparse = torch.concat([tmp_init, sparse_original], dim=0)
        sparse_splits = [(sub_sparse, 0)]

    else:
        while count + args.input_motion_length <= num_frames:
            if count == 0:
                sub_sparse = sparse_original[count : count + args.input_motion_length]
                tmp_idx = 0
            else:
                sub_sparse = sparse_original[count : count + args.input_motion_length]
                tmp_idx = args.input_motion_length - sld_wind_size
            sparse_splits.append((sub_sparse, tmp_idx))
            count += sld_wind_size

        if count < num_frames:
            sub_sparse = sparse_original[-args.input_motion_length :]
            tmp_idx = args.input_motion_length - (
                num_frames - (count - sld_wind_size + args.input_motion_length)
            )
            sparse_splits.append((sub_sparse, tmp_idx))

    memory = None  # init memory

    if args.fix_noise:
        # fix noise seed for every frame
        noise = torch.randn(1, 1, 1).cuda()
        noise = noise.repeat(1, args.input_motion_length, args.motion_nfeat)
    else:
        noise = None

    for step_index in range(len(sparse_splits)):
        sparse_per_batch = sparse_splits[step_index][0].unsqueeze(0)
        memory_end_index = sparse_splits[step_index][1]
        new_batch_size = sparse_per_batch.shape[0]
        
        assert new_batch_size == 1

        if memory is not None:
            model_kwargs = {}
            model_kwargs["y"] = {}
            model_kwargs["y"]["inpainting_mask"] = torch.zeros(
                (
                    new_batch_size,
                    args.input_motion_length,
                    args.motion_nfeat,
                )
            ).cuda()
            model_kwargs["y"]["inpainting_mask"][:, :memory_end_index, :] = 1
            model_kwargs["y"]["inpainted_motion"] = torch.zeros(
                (
                    new_batch_size,
                    args.input_motion_length,
                    args.motion_nfeat,
                )
            ).cuda()
            model_kwargs["y"]["inpainted_motion"][:, :memory_end_index, :] = memory[
                :, -memory_end_index:, :
            ]
        else:
            model_kwargs = None

        sample = sample_fn(
            model,
            (new_batch_size, args.input_motion_length, args.motion_nfeat),
            sparse=sparse_per_batch,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=noise,
            const_noise=False,
        )

        memory = sample.clone().detach()

        if flag_index is not None:
            sample = sample[:, flag_index:].cpu().reshape(-1, args.motion_nfeat)
        else:
            sample = sample[:, memory_end_index:].reshape(-1, args.motion_nfeat)

        output_samples.append(sample.cpu().float())
        
    output_sample = torch.concat(output_samples, dim=0)
    
    if not args.no_normalization:
        motion_target = dataset.inv_transform(motion_target)
        output_sample = dataset.inv_transform(output_sample)

    assert output_sample.shape[0] == num_frames
    

    return output_sample, motion_target, shape_gt, motion_id


def evaluate_prediction(
    args,
    metrics,
    sample,
    flame,
    motion_target, 
    shape_gt,
    fps,
    motion_id,
    face_mask,
    flame_v_mask,
    split,
    test_type
):
    num_frames = motion_target.shape[0]
    pose_6d_pred = sample[:, :24]
    pose_aa_pred = utils_transform.sixd2aa(pose_6d_pred.reshape(-1, 6)).reshape(num_frames, -1)
    expr_pred = sample[:, 24:]
    
    # rigid transformation (zero)
    trans = torch.zeros((num_frames, 3))
    pose_root_aa = torch.zeros((num_frames, 3))
    
    pose_aa_pred_all = torch.cat([pose_root_aa, pose_aa_pred], dim=1)
    verts_pred, lmk_3d_pred = flame(shape_gt, expr_pred, pose_aa_pred_all, trans)

    pose_6d_gt = motion_target[:, :24]
    pose_aa_gt = utils_transform.sixd2aa(pose_6d_gt.reshape(-1, 6)).reshape(num_frames, -1)
    pose_aa_gt_all = torch.cat([pose_root_aa, pose_aa_gt], dim=1)
    expr_gt = motion_target[:, 24:]
    verts_gt, lmk_3d_gt = flame(shape_gt, expr_gt, pose_aa_gt_all, trans)                

    eval_log = {}
    for metric in metrics:
        eval_log[metric] = (
            get_metric_function(metric)(
                shape_gt, expr_pred, pose_aa_pred, trans, verts_pred, lmk_3d_pred,
                expr_gt, pose_aa_gt, trans, verts_gt, lmk_3d_gt,
                fps, flame_v_mask
            )
            .numpy()
        )
    
    # Create visualization
    if args.vis:
        subject_id = motion_id[:11] 
        motion_name = motion_id[12:]
        if split == "train" or (split == "val" and subject_id == "subject_001") or (split == "test" and subject_id == "subject_071"):
            video_dir = os.path.join(args.output_dir, args.arch, subject_id, test_type)
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            faces = flame.faces_tensor.numpy()
            video_path = os.path.join(video_dir, f"{motion_name}.gif")    
            
            pred_animation = utils_visualize.mesh_sequence_to_video_frames(verts_pred, faces, lmk_3d_pred)    
            gt_animation = utils_visualize.mesh_sequence_to_video_frames(verts_gt, faces, lmk_3d_gt)

            vertex_error_per_frame = torch.norm(verts_gt-verts_pred, p=2, dim=2) * 1000.0
            error_heatmaps = utils_visualize.compose_heatmap_to_video_frames(verts_gt, faces, vertex_error_per_frame)
            utils_visualize.concat_videos_to_gif([gt_animation, pred_animation, error_heatmaps], video_path, fps)
            
    torch.cuda.empty_cache()
    return eval_log


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


def load_mlp_model(args):
    model = PureMLP(
        args.latent_dim,
        args.input_motion_length,
        args.layers,
        args.sparse_dim,
        args.motion_nfeat,
    )
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to("cuda:0")
    model.eval()
    return model, None


def main():
    args = sample_args()

    # to guanrantee reproducibility
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fps = args.fps 

    flame = FLAME(args)
    print("Loading dataset...")

    split = "val"
    test_type = "overlap" if args.overlapping_test else "nonoverlap"

    # val data loader
    test_dict, mean, std = load_data_zero_posed(
        args.dataset,
        args.dataset_path,
        split,
        input_motion_length=args.input_motion_length,
    )
    
    dataset = TestDataset(
        args.dataset,
        mean,
        std,
        test_dict,
        args.no_normalization,
    )

    log = {}
    for metric in pred_metrics+gt_metrics:
        log[metric] = 0
    
    for metric in full_vertex_metrics:
        log[metric] = np.zeros(5023)
    
    flame_vmask_path = "flame_2020/FLAME_masks.pkl"
    with open(flame_vmask_path, 'rb') as f:
        flame_v_mask = pickle.load(f, encoding="latin1")
    vertex_mask = flame_v_mask["face"]
    mask = np.array([False] * 5023)
    mask[vertex_mask] = True
    face_mask = mask[flame.faces_tensor.numpy()].all(axis=1)

    for k, v in flame_v_mask.items():
        flame_v_mask[k] = torch.from_numpy(v)
    
    model_type = "diffusion"
    model, diffusion = load_diffusion_model(args)
    sample_fn = diffusion.p_sample_loop
    # elif model_type == "mlp":
    #     model, _ = load_mlp_model(args)
    #     sample_fn = None
    # else:
    #     raise ValueError(f"Unknown model type {model_type}")

    if not args.overlapping_test:
        test_func = non_overlapping_test
    else:
        print("Overlapping testing...")
        test_func = overlapping_test
            
    for sample_index in tqdm(range(len(dataset))):
        with torch.no_grad():
            # motion_id = dataset[sample_index][-1]
            # if motion_id[:11] != "subject_071":
            #     continue
            output_sample, motion_target, shape_gt, motion_id = test_func(
                args,
                dataset[sample_index],
                sample_fn,
                dataset,
                model,
                model_type=model_type
            )

            instance_log = evaluate_prediction(
                args,
                all_metrics,
                output_sample,
                flame,
                motion_target, 
                shape_gt,
                fps,
                motion_id,
                face_mask,
                flame_v_mask,
                split,
                test_type
            )
            for key in instance_log:
                log[key] += instance_log[key]

    # Print the value for all the metrics
    print("Metrics for the predictions")
    for metric in pred_metrics:
        print(f"{metric} : {log[metric] / len(dataset)}")

    print("Metrics for the ground truth")
    for metric in gt_metrics:
        print(f"{metric} : {log[metric] / len(dataset)}")
    
    # visualize the heatmap for full vertex error
    mesh_path = "flame_2020/template.ply"
    template_mesh = trimesh.load_mesh(mesh_path)
    for metric in full_vertex_metrics:
        mean_error = log[metric] / len(dataset)
        img_path = os.path.join(args.output_dir, args.arch, f"{metric}_{test_type}_{split}.png")
        utils_visualize.error_heatmap(template_mesh, mean_error, False, img_path)


if __name__ == "__main__":
    main()
