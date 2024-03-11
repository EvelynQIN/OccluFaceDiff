# TODO
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
from utils.famos_camera import batch_cam_to_img_project
from configs.config import get_cfg_defaults

device = torch.device("cuda")

IMAGE_SIZE = 224

pred_metrics = [
    "pred_jitter",
    "lmk_2d_mpe",
]

all_metrics = pred_metrics

def parse_model_target(target):
    nshape = 300
    nexp = 100 
    npose = 5*6 
    ntrans = 3 
    shape = target[:, :, :nshape]
    exp = target[:, :, nshape:nshape+nexp]
    pose = target[:, :, nshape+nexp:-ntrans]
    trans = target[:, :, -ntrans:]
    return shape, exp, pose, trans

def prepare_data_from_imgs(img_folder):
    """prepare motion input from frames of one video

    Args:
        img_dir: 
    Returns:
        motion: dict of motion input (torch)
    """

def evaluate_prediction(
    args,
    metrics,
    sample,
    flame,
    motion_target, 
    lmk_2d_gt,
    fps,
    motion_id,
    flame_v_mask,
    split
):
    num_frames = motion_target.shape[0]
    shape_gt, expr_gt, pose_gt, trans_gt = parse_model_target(motion_target)
    shape_pred, expr_pred, pose_pred, trans_pred = parse_model_target(sample)
    
    # pose 6d to aa
    pose_aa_gt = utils_transform.sixd2aa(pose_gt.reshape(-1, 6)).reshape(num_frames, -1)
    pose_aa_pred = utils_transform.sixd2aa(pose_pred.reshape(-1, 6)).reshape(num_frames, -1)
    
    
    # flame regress
    verts_gt, lmk_3d_gt = flame(shape_gt, expr_gt, pose_aa_gt)   
    verts_pred, lmk_3d_pred = flame(shape_pred, expr_pred, pose_aa_pred)         
    
    # 2d reprojection
    lmk_2d_pred = batch_cam_to_img_project(lmk_3d_pred, trans_pred) 

    eval_log = {}
    for metric in metrics:
        eval_log[metric] = (
            get_metric_function(metric)(
                shape_gt[:,0,:], expr_pred, pose_aa_pred, trans_pred, verts_pred, lmk_3d_pred, lmk_2d_pred,
                shape_pred[:,0,:], expr_gt, pose_aa_gt, trans_gt, verts_gt, lmk_3d_gt, lmk_2d_gt,
                fps, flame_v_mask 
            )
            .numpy()
        )
    
    # Create visualization
    if args.vis:
        subject_id = motion_id[:11] 
        motion_name = motion_id[12:]
        if (split == "val" and subject_id == "subject_001") or (split == "test" and subject_id == "subject_071"):
            video_dir = os.path.join(args.output_dir, args.arch, subject_id)
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            faces = flame.faces_tensor.numpy()
            video_path = os.path.join(video_dir, f"{motion_name}.gif")    
            
            pred_animation = utils_visualize.mesh_sequence_to_video_frames(verts_pred, faces, lmk_3d_pred)    
            gt_animation = utils_visualize.mesh_sequence_to_video_frames(verts_gt, faces, lmk_3d_gt)

            vertex_error_per_frame = torch.norm(verts_gt-verts_pred, p=2, dim=2) * 1000.0
            error_heatmaps = utils_visualize.compose_heatmap_to_video_frames(verts_gt, faces, vertex_error_per_frame)
            utils_visualize.concat_videos_to_gif([gt_animation, pred_animation, error_heatmaps], video_path, fps)
        
    return eval_log


def load_diffusion_model_from_ckpt(args, pretrained_args):
    print("Creating model and diffusion...")
    args.arch = args.arch[len("diffusion_") :]
    cam_model, denoise_model, diffusion = create_model_and_diffusion(args, pretrained_args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(denoise_model, state_dict)

    cam_model.to(args.device)
    cam_model.eval()
    denoise_model.to(args.device)  # dist_util.dev())
    denoise_model.eval()  # disable random masking
    return cam_model, denoise_model, diffusion


def main():
    args = sample_args()
    pretrained_args = get_cfg_defaults()
    # to guanrantee reproducibility
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fps = args.fps 

    flame = FLAME(args.flame_model_path, args.flame_lmk_embedding_path)
    print("Loading dataset...")

    split = args.split
    device = 'cuda:0'
    # load data from given split
    print("creating val data loader...")
    motion_paths, norm_dict = load_data(
        args,
        args.dataset,
        args.dataset_path,
        "val",
    )
    
    dataset = TestDataset(
        args.dataset,
        norm_dict,
        motion_paths,
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

    for k, v in flame_v_mask.items():
        flame_v_mask[k] = torch.from_numpy(v)
    
    cam_model, denoise_model, diffusion = load_diffusion_model_from_ckpt(args, pretrained_args)
    sample_fn = diffusion.p_sample_loop
            
    for sample_index in tqdm(range(len(dataset))):
        with torch.no_grad():
            flame_params, lmk_2d, lmk_3d_normed, img_arr, motion_id = dataset[sample_index]
            motion_length = flame_params.shape[0]
            flame_params = flame_params.unsqueeze(0).to(device)
            lmk_2d = lmk_2d.unsqueeze(0).to(device)
            trans_cam = cam_model(lmk_2d)
            target = torch.cat([flame_params, trans_cam], dim=-1)
            model_kwargs = {
                "lmk_2d": lmk_2d,
                "lmk_3d": lmk_3d_normed.unsqueeze(0).to(device),
                "img_arr": img_arr.unsqueeze(0).to(device),
            }
            if args.fix_noise:
                # fix noise seed for every frame
                noise = torch.randn(1, 1, 1).cuda()
                noise = noise.repeat(1, motion_length, args.target_nfeat)
            else:
                noise = None
                
            output_sample = sample_fn(
                denoise_model,
                (1, motion_length, args.target_nfeat),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=noise,
                const_noise=False,
            )
            
            if not args.no_normalization:
                output_sample = dataset.inv_transform(output_sample.cpu().float())
                target = dataset.inv_transform(target.cpu().float())
            else:
                output_sample = output_sample.cpu().float()
            
            lmk_2d_gt = (lmk_2d + 1) * IMAGE_SIZE / 2

            instance_log = evaluate_prediction(
                args,
                all_metrics,
                output_sample.squeeze(0),
                flame,
                target.squeeze(0), 
                lmk_2d_gt.reshape(motion_length, -1, 2).to('cpu'),
                fps,
                motion_id,
                flame_v_mask,
                split
            )
            for key in instance_log:
                log[key] += instance_log[key]
            
            torch.cuda.empty_cache()

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
        img_path = os.path.join(args.output_dir, args.arch, f"{metric}_{split}.png")
        utils_visualize.error_heatmap(template_mesh, mean_error, False, img_path)

if __name__ == "__main__":
    main()
