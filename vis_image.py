import h5py
import os
import random
import numpy as np
import torch
import torchvision.transforms.functional as F_v
from tqdm import tqdm
import face_alignment

from utils import utils_transform, utils_visualize
from utils.data_util import batch_orth_proj
from configs.config import get_cfg_defaults
import os.path
from glob import glob
from pathlib import Path
import cv2
import torch.nn.functional as F
from loguru import logger
from pytorch3d.io import load_obj
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from pytorch3d.utils import opencv_from_cameras_projection
from tqdm import tqdm
from time import time
from matplotlib import cm
# pretrained
from model.FLAME import FLAME, FLAMETex, FLAME_mediapipe
from utils.renderer import SRenderY
from skimage.io import imread
import imageio
import ffmpeg
from munch import Munch, munchify
from model.motion_prior import L2lVqVae
from configs.config import get_cfg_defaults
import h5py


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class RenderMeshSequence:
    
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.device = 'cpu'
        self._setup_renderer()
        self._create_flame()
        self.image_size = 224
        
    def _create_flame(self):
        self.flame = FLAME_mediapipe(self.model_cfg).to(self.device)
        flame_template_file = 'flame_2020/head_template_mesh.obj'
        self.faces = load_obj(flame_template_file)[1]
    
    def _setup_renderer(self):
        self.render = SRenderY(
            self.model_cfg.image_size, 
            obj_filename=self.model_cfg.topology_path, 
            uv_size=self.model_cfg.uv_size).to(self.device)
        # face mask for rendering details
        mask = imread(self.model_cfg.face_eye_mask_path).astype(np.float32)/255. 
        mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)
        mask = imread(self.model_cfg.face_mask_path).astype(np.float32)/255.
        mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_mask = F.interpolate(mask, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)
        mean_texture = imread(self.model_cfg.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)

    def vis_image(self, code_split):
        
        pose = torch.cat([code_split['global_pose'], code_split['jaw']], dim=-1)
        verts, lmk_3d_mediapipe = self.flame(
            code_split['shape'], code_split['exp'], pose)
        lmk_3d_fan = self.flame.seletec_3d68(verts)

        # orthogonal projection
        trans_verts = batch_orth_proj(verts, code_split['cam'])
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        # orthogonal projection of landmarks
        # lmk_2d_mp = batch_orth_proj(lmk_3d_mediapipe, code_split['cam'])
        # lmk_2d_mp[:, :, 1:] = -lmk_2d_mp[:, :, 1:]
        lmk_2d_fan = batch_orth_proj(lmk_3d_fan, code_split['cam'])
        lmk_2d_fan[:, :, 1:] = -lmk_2d_fan[:, :, 1:]

        render_images = self.render.render_shape(verts, trans_verts)
        
        # landmarks vis
        lmk2d_vis_mesh = utils_visualize.tensor_vis_landmarks(render_images, lmk_2d_fan, color='r')
        lmk2d_vis_gt = utils_visualize.tensor_vis_landmarks(code_split['image'], code_split['lmk_2d_fan'], color='r')

        # frame_id = str(frame_id).zfill(5)
        vis_dict = {
            'lmk2d_vis_gt': lmk2d_vis_gt[0].detach().cpu(),
            'lmk2d_vis_mesh': lmk2d_vis_mesh[0].detach().cpu()  # (3, 224, 224)    
        }
        grid_image = self.visualize(vis_dict)
        cv2.imwrite('test_fan_lmk_embeddings.png', grid_image[:,:,[2,1,0]])
        
        # cv2.imwrite(f'{savefolder}/{frame_id}.jpg', final_views)
            
    def visualize(self, visdict, dim=2):
        '''
        image range should be [0,1]
        dim: 2 for horizontal. 1 for vertical
        '''
        assert dim == 1 or dim==2
        # grids = {}
        # for key in visdict:
        #     _,h,w = visdict[key].shape
            # if dim == 2:
            #     new_h = size; new_w = int(w*size/h)
            # elif dim == 1:
            #     new_h = int(h*size/w); new_w = size
            # grids[key] = F.interpolate(visdict[key].unsqueeze(0), [new_h, new_w]).detach().cpu().squeeze(0)
        grid = torch.cat(list(visdict.values()), dim)
        grid_image = (grid.numpy().transpose(1,2,0).copy()*255)
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        return grid_image

def main():
    pretrained_args = get_cfg_defaults()
    test_video_id = 'M003/front/angry/level_1/001'
    
    code_dict = {}
    rec_folder = 'dataset/mead_25fps/processed/reconstructions/EMICA-MEAD_flame2020'
    with h5py.File(os.path.join(rec_folder, test_video_id, 'shape_pose_cam.hdf5'), "r") as f:
        # cam : (1, n, 3)
        # exp : (1, n, 100)
        # global_pose : (1, n, 3)
        # jaw : (1, n, 3)
        # shape : (1, n, 300)
        for k in f.keys():
            code_dict[k] = torch.from_numpy(f[k][0]).float()
    with h5py.File(os.path.join(rec_folder, test_video_id, 'appearance.hdf5'), "r") as f:
    # light : (1, n, 27)
    # tex : (1, n, 50)
        for k in f.keys():
            code_dict[k] = torch.from_numpy(f[k][0]).float()
    code_dict['light'] = code_dict['light'].reshape(-1, 9, 3)

    # gt landmarks
    cropped_landmark_folder = os.path.join('dataset/mead_25fps/processed','cropped_landmarks')
    lmk_path = os.path.join(cropped_landmark_folder, test_video_id, 'landmarks_mediapipe.hdf5')
    with h5py.File(lmk_path, "r") as f:
        lmk_2d = torch.from_numpy(f['lmk_2d'][:1]).float()
    code_dict['lmk_2d_mp'] = lmk_2d
    
    
    image_folder = 'dataset/mead_25fps/processed/images'
    img_path = os.path.join(image_folder, test_video_id, 'cropped_frames.hdf5')
    with h5py.File(img_path, "r") as f:
        image = torch.from_numpy(f['images'][:1])
    code_dict['image'] = image.float()


    # FAN detector
    def detect_lmk_on_img(img_rgb):
        
        # face detection model
        face_detector_kwargs = {
            'back_model': False
        }
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.THREE_D, 
            flip_input=False,
            face_detector='blazeface',    # support detectors ['dlib', 'blazeface', 'cfd]
            face_detector_kwargs = face_detector_kwargs,
            device='cuda',
        )    

        # Facial landmarks
        lmks, lmk_score, bbox = fa.get_landmarks_from_image(img_rgb, return_bboxes=True, return_landmark_score=True) # tuple of (lmks, lmk_score, bbox)
        lmks_3d = lmks[0]
        return lmks_3d
    
    img_rgb = image.clone() # (1, 3, 224, 224)
    img_rgb = (img_rgb.permute(0, 2, 3, 1).numpy() * 255.).astype(np.uint8) # (1, 224, 224, 3)
    
    print(img_rgb.shape)
    lmk_2d_fan = detect_lmk_on_img(img_rgb[0])[:,:2]
    print(lmk_2d_fan.shape)
    lmk_2d_fan = torch.from_numpy(lmk_2d_fan).unsqueeze(0)
    lmk_2d_fan = lmk_2d_fan / 112 - 1
    print(lmk_2d_fan.shape)
    code_dict['lmk_2d_fan'] = lmk_2d_fan

    for key in code_dict:
        code_dict[key] = code_dict[key][:1]
    motion_tracker = RenderMeshSequence(pretrained_args.model)

    motion_tracker.vis_image(code_dict)

if __name__ == "__main__":
    main()