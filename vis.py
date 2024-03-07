from model.FLAME import FLAME
import numpy as np 
from utils import utils_visualize, utils_transform
import torch
import argparse


def get_face_motion_unposed(motion, flame):
    # zero the global translation and pose (rigid transformation)
    shape = torch.Tensor(motion["flame_shape"])
    expression = torch.Tensor(motion["flame_expr"])
    rot_aa = torch.Tensor(motion["flame_pose"])
    trans = torch.Tensor(motion["flame_trans"])
    
    n_frames = expression.shape[0]

    verts, lmk_3d = flame(shape, expression, rot_aa, trans)

    return verts, lmk_3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--flame_model_path", default='flame_2020/generic_model.pkl', type=str, help="the path to the flame model"
    )
    parser.add_argument(
        "--flame_lmk_embedding_path", default='flame_2020/dense_lmk_embedding.npy', type=str, help="the path to the flame landmark embeddings"
    )

    args = parser.parse_args()
    
    motion_path = "dataset/FaMoS/flame_params/FaMoS_subject_095/bareteeth.npy"
    data = np.load(motion_path, allow_pickle=True)[()]
    flame = FLAME(args)
    verts, lmk_3d = get_face_motion_unposed(data, flame)
    video_path = "test.mp4"
    fps=60
    faces = flame.faces_tensor.numpy()
    utils_visualize.mesh_sequence_to_video(verts.numpy(), faces, video_path, fps, lmk_3d.numpy())