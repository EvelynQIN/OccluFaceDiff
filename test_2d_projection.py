import argparse
import os
import numpy as np
import torch
from tqdm import tqdm 
import glob
from utils import utils_transform
import pickle
from model.FLAME import FLAME
import cv2 
from utils.famos_camera import load_mpi_camera

import open3d as o3d


def show_2d_lmks_on_img(root_dir, subject, motion, frame_id, flame):
    dataset = 'FaMoS'
    flame_params_folder = os.path.join(root_dir, dataset, "flame_params")
    camera_calibration_folder = os.path.join(root_dir, dataset, "calibrations")
    img_folder = os.path.join(root_dir, dataset, "downsampled_images_4")
    
    camera_name = "26_C" # name of the selected camera view
    calib_fname = os.path.join(camera_calibration_folder, subject, motion, f"{camera_name}.tka")
    flame_fname = os.path.join(flame_params_folder, subject, f"{motion}.npy")
    flame_motion = np.load(flame_fname, allow_pickle=True)[()]
    img_path =  os.path.join(img_folder, subject, motion, frame_id, f'{motion}.{frame_id}.{camera_name}.png')
    idx = np.where(flame_motion["frame_id"] == int(frame_id))[0][0]
    print("idx = ", idx)
    
    shape = flame_motion["flame_shape"][idx]
    expression = flame_motion["flame_expr"][idx]
    rot_aa = flame_motion["flame_pose"][idx] 
    trans = flame_motion['flame_trans'][idx]
    
    R_f = utils_transform.aa2matrot(torch.Tensor(rot_aa[:3].reshape(-1, 3))).numpy().squeeze()    
    t_f = trans.reshape(3, 1)  
    
    # camera params
    calibration = load_mpi_camera(calib_fname, resize_factor=4)
    camera_intrinsics = calibration["intrinsics"]
    camera_extrinsics = calibration["extrinsics"]
    radial_distortion = calibration["radial_distortion"]
    R_C, t_C = camera_extrinsics[:, :3], camera_extrinsics[:, 3]
    
    # flame mesh in the world coord
    verts_world, lmk_3d_world = flame(
        torch.Tensor(shape).reshape(1, -1), 
        torch.Tensor(expression).reshape(1, -1), 
        torch.Tensor(rot_aa).reshape(1, -1), 
        torch.Tensor(trans).reshape(1, -1))
    lmk_3d_world = lmk_3d_world[0].numpy()
    verts_world = verts_world[0].numpy()
    
    # compute the lmk_3d world to cam 
    ones = np.ones((verts_world.shape[0], 1))
    verts_homo = np.concatenate((verts_world, ones), axis=-1)
    lmk_3d_homo = np.concatenate((lmk_3d_world, np.ones((lmk_3d_world.shape[0], 1))), axis=-1)
    verts_cam_gt = camera_extrinsics.dot(verts_homo.T).T  
    lmk3d_cam_gt = camera_extrinsics.dot(lmk_3d_homo.T).T 
    
    # 2d preojection test 
    print(camera_intrinsics)
    lmk3d_img = camera_intrinsics.dot(lmk3d_cam_gt.T).T
    lmk3d_img[:, 0] /= lmk3d_img[:, 2]
    lmk3d_img[:, 1] /= lmk3d_img[:, 2]
    image = cv2.imread(img_path)
    for i, (x, y, z) in enumerate(lmk3d_img):
        loc = (int(x), int(y))
        cv2.circle(image, loc, 1, (255, 255, 255), -1)
        cv2.putText(image, str(i+1), loc, 0, 1, (1, 0, 0), thickness=1)
    to_path = f"lmk_detection_test.png"
    cv2.imwrite(to_path, image)
    print(f"save lmk detection image to {to_path}")
    
    # change the relative rot and trans of flame in cam coords
    global_rot_mat = np.dot(R_C, R_f)
    global_rot_aa = utils_transform.matrot2aa(torch.Tensor(global_rot_mat).unsqueeze(0)).numpy()
    rot_aa[:3] = global_rot_aa
    
    root_pos = flame.get_root_location(
        torch.Tensor(shape).reshape(1, -1), 
        torch.Tensor(expression).reshape(1, -1))[0].numpy().reshape(3, 1)
    
    global_trans = np.dot(R_C, (root_pos + t_f)) + t_C.reshape(-1, 1) - root_pos
    
    rot_aa[:3] = global_rot_aa
    verts_cam_t0, lmk_3d_cam_flame = flame(
        torch.Tensor(shape).reshape(1, -1), 
        torch.Tensor(expression).reshape(1, -1), 
        torch.Tensor(rot_aa).reshape(1, -1), 
        torch.zeros(1, 3))
    verts_cam_t0 = verts_cam_t0[0].numpy() # (nframes, V, 3)
    # print(verts_cam_gt - verts_cam_t0)    # [0.02238051 0.28489265 1.16285809]
    
    # visualize the 2 meshes
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    verts_gt_cam = o3d.utility.Vector3dVector(verts_cam_gt)
    triangles = o3d.utility.Vector3iVector(flame.faces_tensor)
    mesh_cam_gt = o3d.geometry.TriangleMesh(vertices=verts_gt_cam, triangles=triangles)
    mesh_cam_gt.compute_vertex_normals()
    mesh_cam_gt.paint_uniform_color([1, 0.706, 0])
    
    verts_t0_cam = o3d.utility.Vector3dVector(verts_cam_t0)
    mesh_t0 = o3d.geometry.TriangleMesh(vertices=verts_t0_cam, triangles=triangles)
    mesh_t0.compute_vertex_normals()
    mesh_t0.paint_uniform_color([0.635, 0.961, 0.263])
    

    visualization_list = [mesh_cam_gt, mesh_t0, coord]
    o3d.visualization.draw_geometries(visualization_list)
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--root_dir", type=str, default="./dataset", help="=dir where you put your dataset"
    )
    parser.add_argument(
        "--flame_model_path", default='flame_2020/generic_model.pkl', type=str, help="the path to the flame model"
    )
    parser.add_argument(
        "--flame_lmk_embedding_path", default='flame_2020/dense_lmk_embedding.npy', type=str, help="the path to the flame landmark embeddings"
    )
    args = parser.parse_args()
    
    flame = FLAME(args.flame_model_path, args.flame_lmk_embedding_path)
    show_2d_lmks_on_img(
        root_dir=args.root_dir, 
        subject="FaMoS_subject_001", 
        motion="anger", 
        frame_id="000013", 
        flame=flame)
