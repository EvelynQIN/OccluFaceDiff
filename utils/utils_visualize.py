# MIT License
# Copyright (c) 2022 ETH Sensing, Interaction & Perception Lab
#
# This code is based on https://github.com/eth-siplab/AvatarPoser
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os

import cv2
import numpy as np
import trimesh
import torch
from tqdm import tqdm
import glob 
import ffmpeg
import gc
import pyrender
from vedo import trimesh2vedo, show
from PIL import Image

# os.environ["PYOPENGL_PLATFORM"] = "egl"

def mesh_sequence_to_video_frames(mesh_verts, faces, lmk_3d=None):
    """Render one mesh sequence into  video frames

    Args:
        mesh_verts: N X 5023 X 3
        faces: faces in FLAME topology
    Returns:
        video_frames: 4D numpy array of shape [T, C, H, W]
    """
    
    h, w = 640, 480

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0) 
    camera_pose = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0.22],
                            [0, 0, 1, 0.6],
                            [0.0, 0.0, 0.0, 1.0]])

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    
    material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor= (144, 117, 87, 255),
                metallicFactor=0.8,
                roughnessFactor=0.8,
                wireframe=True,
                emissiveFactor= 0.3,
    )

    
    r = pyrender.OffscreenRenderer(viewport_height=h, viewport_width=w)
    
    nframes = mesh_verts.shape[0]
    video_frames = np.zeros((nframes, h, w, 3), dtype=np.uint8)

    if lmk_3d is not None:
        n_lmks = lmk_3d.shape[1]
    
    for i in range(nframes):
        verts = mesh_verts[i]
        obj_mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=False
            )
        
        py_mesh = pyrender.Mesh.from_trimesh(obj_mesh, material=material)

        scene = pyrender.Scene(bg_color=[0, 0, 0, 255],
                               ambient_light=[0.2, 0.2, 0.2])
        node = pyrender.Node(
            mesh=py_mesh,
            translation=[0, 0, 0]
        )
        scene.add_node(node)

        if lmk_3d is not None:
            # render the 3d keypoints
            sm = trimesh.creation.uv_sphere(radius=0.002)
            sm.visual.vertex_colors = [1.0, 0.0, 0.0]
            tfs = np.tile(np.eye(4), (n_lmks, 1, 1))
            tfs[:,:3,3] = lmk_3d[i]
            py_lmk_3d = pyrender.Mesh.from_trimesh(sm, poses = tfs)
            scene.add(py_lmk_3d)            
        
        scene.add(cam, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        color, _ = r.render(scene)
        video_frames[i] = color
    return video_frames

def mesh_sequence_to_video(mesh_verts, faces, video_path, fps, lmk_3d=None):
    """Render one mesh sequence into one video and save it

    Args:
        mesh_verts: N X 5023 X 3
        faces: faces in FLAME topology
        video_path: path to the mp4 file to save the result
        lmk_3d: N X K X 3
    """
    
    h, w = 640, 480
    cam = pyrender.PerspectiveCamera(np.pi / 3) 
    
    # [[0.3669,  0.6816,  0.6331, 0],
    # [-0.6816, -0.2663,  0.6816, -0.222],
    # [0.6331, -0.6816,  0.3669, -1],
    # [0.0, 0.0, 0.0, 1.0]]

    camera_pose = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0.22],
                            [0, 0, 1, 0.8],
                            [0.0, 0.0, 0.0, 1.0]])

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    
    material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor= (144, 117, 87, 255),
                metallicFactor=0.8,
                roughnessFactor=0.8,
                wireframe=True,
                emissiveFactor= 0.3,
    )

    r = pyrender.OffscreenRenderer(viewport_height=h, viewport_width=w)
    
    nframes = mesh_verts.shape[0]
    if lmk_3d is not None:
        n_lmks = lmk_3d.shape[1]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    
    for i in range(nframes):
        verts = mesh_verts[i]
        obj_mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=False
            )
        
        py_mesh = pyrender.Mesh.from_trimesh(obj_mesh, material=material)

        scene = pyrender.Scene(bg_color=[0, 0, 0, 255],
                               ambient_light=[0.2, 0.2, 0.2])
        node = pyrender.Node(
            mesh=py_mesh,
            translation=[0, 0, 0]
        )
        
        if lmk_3d is not None:
            # render the 3d keypoints
            sm = trimesh.creation.uv_sphere(radius=0.002)
            sm.visual.vertex_colors = [1.0, 0.0, 0.0]
            tfs = np.tile(np.eye(4), (n_lmks, 1, 1))
            tfs[:,:3,3] = lmk_3d[i]
            py_lmk_3d = pyrender.Mesh.from_trimesh(sm, poses = tfs)
            scene.add(py_lmk_3d)            
        
        scene.add_node(node)
        
        scene.add(cam, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        color, _ = r.render(scene)
        video.write(color)
    video.release()
    

def error_heatmap(template_mesh, metrics, output_array=True, img_path=None, size=(480,640), color_range=None, camera=None):
    """generate heatmap w.r.t. the metrics onto the template face mask

    Args:
        template_mesh: a trimesh obejct in FLAME toplogy
        metrics: per vertex metrics, (5023,)
        img_path: the path of the image to output the heatmap (.png)
        color_range: tuple of (vmin, vmax), range of the color map
        size: tuple of (width, height)
    """
        
    vmesh = trimesh2vedo(template_mesh)
    if color_range is not None:
        vmesh.cmap("jet", metrics, vmin=color_range[0], vmax=color_range[1])
    else:
        vmesh.cmap("jet", metrics)
    vmesh.add_scalarbar(title="mm")
    
    if camera is not None:
        plt = show(vmesh, offscreen=True, size=size, bg="black", camera=camera)
    else:
        plt = show(vmesh, offscreen=True, size=size, bg="black")
    if output_array or img_path is None:
        heatmap_img = plt.screenshot(asarray=output_array)
        plt.close()
        return heatmap_img
    else:
        plt.screenshot(img_path)
        plt.close()

def compose_heatmap_to_video_frames(verts_gt, faces, vertex_error, size = (640, 480), camera=None):
    num_frames = verts_gt.shape[0]
    h, w = size
    heatmap_frames = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
    vmin = torch.min(vertex_error)
    vmax = torch.max(vertex_error)
    
    for i in range(num_frames):
        verts = verts_gt[i]
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=False
        )
        heatmap_frames[i] = error_heatmap(mesh, vertex_error[i], output_array=True, size=(w, h), color_range=(vmin,vmax), camera=camera)
    return heatmap_frames

def concat_videos_to_gif(video_list, output_path, fps):
    """concat list of frames into one gif

    Args:
        video_list (_type_): list if video frames, each element should be in (T, H, W, C)
        output_path (_type_): _description_
        fps (_type_): _description_
    """
    frames = np.concatenate(video_list, axis=2) # concat the frame horizonally
    frames_list = [Image.fromarray(f) for f in frames]
    frame_one = frames_list[0]
    frame_one.save(
        output_path, 
        format="GIF", 
        append_images=frames_list[1:],
        save_all=True, 
        duration=300/fps, 
        optimize=False,
        loop=0,)
    # t, h, w, c = frames.shape
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    # for frame in frames:
    #     im_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     video.write(im_bgr)
    # video.release()