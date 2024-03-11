from model.FLAME import FLAME
import numpy as np 
from utils import utils_visualize, utils_transform
import torch
import argparse
from vedo import trimesh2vedo, show
import trimesh

def get_face_motion(motion, flame):
    # zero the global translation and pose (rigid transformation)
    flame_params = motion['target']
    shape = flame_params[:,:300]
    expression = flame_params[:,300:400]
    rot_6d = flame_params[:,400:]
    n = flame_params.shape[0]
    rot_aa = utils_transform.sixd2aa(rot_6d.reshape(-1, 6)).reshape(n, -1)
    verts, lmk_3d = flame(shape, expression, rot_aa)

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
    
    motion_path = "processed_data/FaMoS/test/subject_071_bareteeth.pt"
    motion = torch.load(motion_path)
    flame = FLAME(args.flame_model_path, args.flame_lmk_embedding_path)
    verts, lmk_3d = get_face_motion(motion, flame)
    verts[:, :, 1:] = -verts[:, :, 1:] 
    video_path = "test.mp4"
    fps=60
    faces = flame.faces_tensor.numpy()
    utils_visualize.mesh_sequence_to_video(verts.numpy(), faces, video_path, fps)
    size=(480,640)
    mesh = trimesh.Trimesh(vertices=verts[0], faces=faces)
    vmesh = trimesh2vedo(mesh)
    plt = show(vmesh, offscreen=True, size=size, bg="black", 
               camera={'pos':(0.0, 0.25, 0.8), 'view_angle': 60}
    )
    img_path = 'test.png'
    plt.screenshot(img_path)
    plt.close()