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
import face_alignment
import matplotlib.pyplot as plt
import collections
from utils.famos_camera import perspective_project, load_mpi_camera

def detect_lmk_on_img(img_path):
    
    image = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    # face detection model
    face_detector_kwargs = {
        'back_model': False
    }
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.THREE_D, 
        flip_input=False,
        face_detector='blazeface',    # support detectors ['dlib', 'blazeface', 'cfd]
        face_detector_kwargs = face_detector_kwargs,
        dtype=torch.bfloat16, device='cuda',
    )    

    # Facial landmarks
    lmks, lmk_score, bbox = fa.get_landmarks_from_image(rgb_img, return_bboxes=True, return_landmark_score=True) # tuple of (lmks, lmk_score, bbox)
    lmks_3d = lmks[0]
    return lmks_3d

def normalize_lmk_3d(lmk_3d):
    root_idx = 30
    pivot_idx = 27
    root_node = lmk_3d[root_idx]
    nose_len = np.linalg.norm(lmk_3d[pivot_idx]-lmk_3d[root_idx], 2, -1)
    
    lmk_3d_normed = (lmk_3d - root_node) / nose_len
    print(np.linalg.norm(lmk_3d_normed[pivot_idx]-lmk_3d_normed[root_idx], 2, -1))
    return lmk_3d_normed


def test(root_dir, subject, motion, frame_id, flame):
    dataset = 'FaMoS'
    flame_params_folder = os.path.join(root_dir, dataset, "flame_params")
    img_folder = os.path.join(root_dir, dataset, "downsampled_images_4")
    
    camera_name = "26_C" # name of the selected camera view
    flame_fname = os.path.join(flame_params_folder, subject, f"{motion}.npy")
    img_path = os.path.join(img_folder, subject, motion, frame_id, f'{motion}.{frame_id}.{camera_name}.png')
    flame_motion = np.load(flame_fname, allow_pickle=True)[()]
    
    idx = np.where(flame_motion["frame_id"] == int(frame_id))[0][0]
    print("idx = ", idx)
    
    shape = flame_motion["flame_shape"][idx]
    expression = flame_motion["flame_expr"][idx]
    rot_aa = flame_motion["flame_pose"][idx, :-2*3] # full pose exluding eye pose (global, neck, jaw) 3 * 3
    trans = flame_motion['flame_trans'][idx]
    
    # camera params
    camera_calibration_folder = os.path.join(root_dir, dataset, "calibrations")
    calib_fname = os.path.join(camera_calibration_folder, subject, motion, f"{camera_name}.tka")
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
    lmk_3d_cam = R_C.dot(lmk_3d_world.T).T + t_C
    lmk3d_cam_normed = normalize_lmk_3d(lmk_3d_cam)

    # lmk detection
    lmk3d_img = detect_lmk_on_img(img_path)
    lmk3d_img_normed = normalize_lmk_3d(lmk3d_img)

    print(lmk3d_cam_normed - lmk3d_img_normed)

    # visualize
    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                }
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.scatter(lmk3d_cam_normed[:, 0],
                    lmk3d_cam_normed[:, 1],
                    lmk3d_cam_normed[:, 2],
                    c='cyan',
                    alpha=1.0,
                    edgecolor='b')

    for pred_type in pred_types.values():
        ax.plot3D(lmk3d_cam_normed[pred_type.slice, 0],
                lmk3d_cam_normed[pred_type.slice, 1],
                lmk3d_cam_normed[pred_type.slice, 2], color='blue')
    ax.view_init(elev=90., azim=45.)
    ax.set_xlim(ax.get_xlim()[::-1])

    # vis detected 3d lmk
    surf = ax.scatter(lmk3d_img_normed[:, 0],
                    lmk3d_img_normed[:, 1],
                    lmk3d_img_normed[:, 2],
                    c='red',
                    alpha=1.0,
                    edgecolor='r')

    for pred_type in pred_types.values():
        ax.plot3D(lmk3d_img_normed[pred_type.slice, 0],
                lmk3d_img_normed[pred_type.slice, 1],
                lmk3d_img_normed[pred_type.slice, 2], color='red')
    
    plt.show()
    plt.savefig("lmk3d_vis.png")
    
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
    test(
        root_dir=args.root_dir, 
        subject="FaMoS_subject_001", 
        motion="anger", 
        frame_id="000013", 
        flame=flame)

