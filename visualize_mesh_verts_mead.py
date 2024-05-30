import h5py
import os
import random
import numpy as np
import torch
import torchvision.transforms.functional as F_v
from tqdm import tqdm

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
        self.with_audio = True
        self.device = 'cuda'
        self.sld_wind_size = 32
        self._setup_renderer()
        self._create_flame()
        self.fps = 25
        self.n_views =1 
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


    def vis_motion_split(self, mesh_split, code_split):
        n = mesh_split.shape[0]

        # shape template
        null_exp = torch.zeros((n, 100)).to(self.device)
        null_jaw = torch.zeros((n, 3)).to(self.device)
        pose = torch.cat([code_split['global_pose'], null_jaw], dim=-1)
        shape_template, _ = self.flame(code_split['shape'], null_exp, pose)
        verts = shape_template + mesh_split

        # orthogonal projection
        trans_verts = batch_orth_proj(verts, code_split['cam'])
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        render_images = self.render.render_shape(verts, trans_verts)
        
        # landmarks vis
        # lmk2d_vis = utils_visualize.tensor_vis_landmarks(motion_split['images'], lmk2d_rec, motion_split['lmk_2d'])

        # frame_id = str(frame_id).zfill(5)
        for i in range(render_images.shape[0]):
            vis_dict = {
                # 'gt_img': gt_img[i].detach().cpu(),   # (3, 224, 224)
                'mesh_pred': render_images[i].detach().cpu(),  # (3, 224, 224)
                # 'flint': flint_render_images[i].detach().cpu(),  # (3, 224, 224)
                # 'lmk2d': lmk2d_vis[i].detach().cpu()
            }
            grid_image = self.visualize(vis_dict)
            if self.with_audio:
                self.writer.write(grid_image[:,:,[2,1,0]])
            else:
                self.writer.append_data(grid_image)
        
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
    
    def run_vis(self, code_dict, batch, test_video_id, audio_path=None, v_name=None):
        # make prediction by split the whole sequences into chunks due to memory limit
        self.num_frames = batch.shape[0]
        # set the output writer
        if self.with_audio:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(v_name+'.mp4', fourcc, self.fps, (self.image_size*self.n_views, self.image_size))
        else:
            self.writer = imageio.get_writer(v_name + '.gif', mode='I')
        
        for start_id in range(0, self.num_frames, self.sld_wind_size):
            print(f"Processing frame starting from {start_id}")
            
            motion_split = batch[start_id:start_id+self.sld_wind_size].to(self.device)

            code_split = {}
            for key in ['shape', 'light', 'tex', 'cam', 'global_pose']:
                code_split[key] = code_dict[key][start_id:start_id+self.sld_wind_size].to(self.device)

            self.vis_motion_split(motion_split, code_split)
        
        # concat audio 
        if self.with_audio:
            self.writer.release()
            os.system(f"ffmpeg -i {v_name}.mp4 -i {audio_path} -c:v copy -c:a copy {v_name}_audio.mp4")
            os.system(f'rm {v_name}.mp4')
        
        torch.cuda.empty_cache()

def main():
    pretrained_args = get_cfg_defaults()
    test_video_id = 'M003/front/angry/level_1/001'
    model_type = 'VOCA'
    sbj, view, emotion, level, sent = test_video_id.split('/')
    audio_path = os.path.join('dataset/mead_25fps/original_data', sbj, 'audio', emotion, level, f'{sent}.m4a')
    
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


    motion_tracker = RenderMeshSequence(pretrained_args.model)
    mesh_path = os.path.join(f'/local/home/yaqqin/repos/face_diffuser/vis_result/{model_type}/reconstruction', f'{test_video_id}.npy')
    verts_disp = np.load(mesh_path, allow_pickle=True)
    verts_disp = torch.from_numpy(verts_disp).float()
    
    num_frames = code_dict['cam'].shape[0]
    if num_frames < verts_disp.shape[0]:
        verts_disp = verts_disp[:num_frames]
    elif num_frames > verts_disp.shape[0]:
        verts_disp = torch.cat([verts_disp, verts_disp[-1:].repeat(num_frames-verts_disp.shape[0], 1, 1)], dim=0)
    
    vid = test_video_id.split('/')
    v_name = '_'.join(vid) + f'_{model_type}'
    motion_tracker.run_vis(code_dict, verts_disp, test_video_id, audio_path, v_name)

if __name__ == "__main__":
    main()