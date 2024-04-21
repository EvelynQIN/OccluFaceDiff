import argparse
import os
import numpy as np
import torch
from tqdm import tqdm 
import glob
from utils import utils_transform
import pickle
from model.FLAME import FLAME

from utils.famos_camera import batch_perspective_project, load_mpi_camera
from utils.data_util import batch_crop_lmks, crop_np, batch_normalize_lmk_3d
import cv2 
from skimage.transform import estimate_transform, warp, resize, rescale
from configs.config import get_cfg_defaults

def batch_3d_to_2d(calibration, lmk_3d):
    
    # all in tensor
    bs = lmk_3d.shape[0]
    camera_intrinsics = torch.from_numpy(calibration["intrinsics"]).expand(bs,-1,-1).float()
    camera_extrinsics = torch.from_numpy(calibration["extrinsics"]).expand(bs,-1,-1).float()
    radial_distortion = torch.from_numpy(calibration["radial_distortion"]).expand(bs,-1).float()
    
    lmk_2d = batch_perspective_project(lmk_3d, camera_intrinsics, camera_extrinsics, radial_distortion)

    return lmk_2d

def create_training_data(args, dataset, device='cuda'):
    print("processing dataset ", dataset)
    root_dir = './dataset'
    
    # init flame
    flame = FLAME(args)

    # # init facial segmentator
    # config_file = 'face_segmentation/deeplabv3plus_r101_512x512_face-occlusion.py'
    # checkpoint_file = 'face_segmentation/deeplabv3plus_r101_512x512_face-occlusion-93ec6695.pth'
    # face_segment = init_model(config_file, checkpoint_file, device=device)
    
    flame_params_folder = os.path.join(root_dir, dataset, "flame_params")
    calib_folder = os.path.join(root_dir, dataset, "calib")
    image_folder = os.path.join(root_dir, dataset, "image")
    out_folder = os.path.join(root_dir, dataset, "processed")
    camera_name = "26_C" # name of the selected camera view

    for sbj in tqdm(os.scandir(image_folder)):
        print(f"Process {sbj.name}")

        out_folder_sbj = os.path.join(out_folder, sbj.name)
        if not os.path.exists(out_folder_sbj):
            os.makedirs(out_folder_sbj)

        for motion in os.scandir(sbj.path):
            out_fname = os.path.join(out_folder_sbj, f"{motion.name}.pt")
            if os.path.exists(out_fname):
                continue
            flame_params_path = os.path.join(flame_params_folder, sbj.name, f"{motion.name}.npy")
            if not os.path.exists(flame_params_path):
                continue
            flame_params = np.load(flame_params_path, allow_pickle=True)[()]
   
            num_frames = len(flame_params["flame_verts"])
            print(f"processing {motion.name} with {num_frames} frames")
            if num_frames < 10:
                print(f"{sbj.name} {motion.name} is nulll")
                continue
            calib_fname = os.path.join(calib_folder, sbj.name, motion.name, f"{camera_name}.tka")

            # get the gt flame params and projected 2d lmks
            shape = torch.Tensor(flame_params["flame_shape"])[:,:100]
            expression = torch.Tensor(flame_params["flame_expr"])[:,:50]
            rot_aa = torch.Tensor(flame_params["flame_pose"])[:,:3*3] # full poses exluding eye poses (root, neck, jaw, left_eyeball, right_eyeball)
            trans = torch.Tensor(flame_params['flame_trans']).unsqueeze(1)
            frame_id = torch.Tensor(flame_params['frame_id']).long()

            # get 2d landmarks from gt mesh
            _, lmk_3d, _ = flame(shape, expression, rot_aa) # (nframes, V, 3)
            lmk_3d += trans
            calibration = load_mpi_camera(calib_fname, resize_factor=4.0)
            lmk_2d = batch_3d_to_2d(calibration, lmk_3d)

            # get 6d pose representation of jaw pose
            rot_jaw_6d = utils_transform.aa2sixd(rot_aa[...,6:9]) # (nframes, 6)
            target = torch.cat([rot_jaw_6d, expression], dim=1) 
            
            output = {
                "lmk_2d": lmk_2d, 
                "target": target, 
                "shape": shape,
                "frame_id": frame_id
            }
            for k in output:
                assert output[k] is not None
            
            torch.save(output, out_fname)
    
if __name__ == "__main__":
    model_cfg = get_cfg_defaults().model
    dataset = 'FaMoS'
    device = 'cpu'
    create_training_data(model_cfg, dataset, device)