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

class MotionTracker:
    
    def __init__(self, 
                 model_cfg, 
                 sld_wind_size=10,
                 device='cuda',
                 with_audio=False,
                 use_tex=False,
                 fps=25):
        
        self.model_cfg = model_cfg
        self.device = device
        self.sld_wind_size = sld_wind_size
        self.use_tex=use_tex
        self.fps=fps
        
        self.image_size = 224

        self.with_audio = with_audio

        self._create_flame()
        self._setup_renderer()

        # set up flint 
        ckpt_path = 'pretrained/MotionPrior/models/FLINTv2/checkpoints/model-epoch=0758-val/loss_total=0.113977119327.ckpt'
        f = open('pretrained/MotionPrior/models/FLINTv2/cfg.yaml')
        cfg = Munch.fromYAML(f)
        self.flint = L2lVqVae(cfg)
        self.flint.load_model_from_checkpoint(ckpt_path)
        self.flint.freeze_model()
        self.flint.to(self.device)
    
    def _create_flame(self):
        self.model_cfg.n_exp = 100
        self.flame = FLAME_mediapipe(self.model_cfg).to(self.device)
        self.flametex = FLAMETex(self.model_cfg).to(self.device)    
    
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
        # # TODO: displacement correction
        # fixed_dis = np.load(self.model_cfg.fixed_displacement_path)
        # self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        mean_texture = imread(self.model_cfg.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)
        # # dense mesh template, for save detail mesh
        # self.dense_template = np.load(self.model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()
    
    def vis_motion_split(self, motion_split, emica_split):
        
        # prepare vis data dict 
        shape = emica_split['shape'][:,:300]
        exp = emica_split['exp'][:,:100]
        cam = emica_split['cam']
        pose_gt = torch.cat([emica_split['global_pose'], emica_split['jaw']], dim=-1)

        # run flint
        inputs = torch.cat([exp, emica_split['jaw']], dim=-1).unsqueeze(0)  # (bs, 32, 53)
        print(inputs.shape)
        z = self.flint.motion_encoder(inputs)
        print("z: ", z.shape)

        rec = self.flint.motion_decoder(z)[0]
        print("rec: ", rec.shape)
        rec_exp = rec[...,:100]
        rec_jaw = rec[...,100:]
        rec_pose = torch.cat([emica_split['global_pose'], rec_jaw], dim=-1)

        # flame decoder
        emica_verts_gt, lmk_3d_gt = self.flame(shape, exp, pose_gt)
        # orthogonal projection
        emica_trans_verts_gt = batch_orth_proj(emica_verts_gt, cam)
        emica_trans_verts_gt[:, :, 1:] = -emica_trans_verts_gt[:, :, 1:]

        emica_verts_rec, lmk_3d_rec = self.flame(shape, rec_exp, rec_pose)
        # orthogonal projection
        emica_trans_verts_rec = batch_orth_proj(emica_verts_rec, cam)
        emica_trans_verts_rec[:, :, 1:] = -emica_trans_verts_rec[:, :, 1:]
        
        lmk2d_rec = batch_orth_proj(lmk_3d_rec, cam)[:, :, :2]
        lmk2d_rec[:, :, 1:] = -lmk2d_rec[:, :, 1:]

        # # render
        if self.use_tex:
            albedo = self.flametex(emica_split['tex']).detach()
            light = emica_split['light']
            emica_render_images = self.render(emica_verts_gt, emica_trans_verts_gt, albedo, light, background=motion_split['images'])['images']
            flint_render_images = self.render(emica_verts_rec, emica_trans_verts_rec, albedo, light, background=motion_split['images'])['images']

        else:
            emica_render_images = self.render.render_shape(emica_verts_gt, emica_trans_verts_gt, images=motion_split['images'])
            flint_render_images = self.render.render_shape(emica_verts_rec, emica_trans_verts_rec, images=motion_split['images'])
        
        # landmarks vis
        lmk2d_vis = utils_visualize.tensor_vis_landmarks(motion_split['images'], lmk2d_rec, motion_split['lmk_2d'])
        

        gt_img = motion_split['images'] * (motion_split['img_masks'].unsqueeze(1))

        # frame_id = str(frame_id).zfill(5)
        for i in range(lmk2d_rec.shape[0]):
            vis_dict = {
                'gt_img': gt_img[i].detach().cpu(),   # (3, 224, 224)
                'emica': emica_render_images[i].detach().cpu(),  # (3, 224, 224)
                'flint': flint_render_images[i].detach().cpu(),  # (3, 224, 224)
                'lmk2d': lmk2d_vis[i].detach().cpu()
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
    
    def run_vis(self, batch, emica_codes, test_video_id):
        # make prediction by split the whole sequences into chunks due to memory limit
        self.num_frames = batch['lmk_2d'].shape[0]
        vid = test_video_id.split('/')
        v_name = '_'.join(vid) + '_test_flint'
        # set the output writer
        if self.with_audio:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(v_name+'_v2.mp4', fourcc, self.fps, (self.image_size*4, self.image_size))
        else:
            self.writer = imageio.get_writer(v_name + '_v2.gif', mode='I')
        
        for start_id in range(0, self.num_frames, self.sld_wind_size):
            print(f"Processing frame starting from {start_id}")
            if self.num_frames - start_id < self.sld_wind_size:
                break
            motion_split = {}
            for key in batch:
                motion_split[key] = batch[key][start_id:start_id+self.sld_wind_size].to(self.device)
            
            emica_split = {}
            for key in emica_codes:
                emica_split[key] = emica_codes[key][start_id:start_id+self.sld_wind_size].to(self.device)
            self.vis_motion_split(motion_split, emica_split)
        
        # concat audio 
        if self.with_audio:
            self.writer.release()
            # input_video = ffmpeg.input(video_path+'.mp4')
            # input_audio = ffmpeg.input(audio_path)
            # ffmpeg.concat(input_video, input_audio, v=1, a=1).output(video_path+'_audio.mp4', vcodec='rawvideo').run()
            audio_path = 'dataset/mead_25fps/original_data/M003/audio/contempt/level_1/001.m4a'
            os.system(f"ffmpeg -i {v_name}.mp4 -i {audio_path} -c:v copy -c:a copy {v_name}_audio.mp4")
            os.system(f"rm {v_name}.mp4")
        
        torch.cuda.empty_cache()

def main():
    pretrained_args = get_cfg_defaults()
    test_video_id = 'M003/front/contempt/level_1/001'
    img_folder = 'dataset/mead_25fps/processed/images'
    filename = os.path.join(img_folder, test_video_id, 'cropped_frames.hdf5')
    data_dict = {}

    with h5py.File(filename, "r") as f:
        for k in f.keys():
            print(f"shape of {k} : {f[k].shape}")
            data_dict[k] = torch.from_numpy(f[k][:]).float()
    """
    shape of images : (88, 3, 224, 224)
    shape of img_masks : (88, 224, 224)
    shape of lmk_2d : (88, 478, 2)
    shape of valid_frames_idx : (88,)
    """

    rec_folder = 'dataset/mead_25fps/processed/reconstructions/EMICA-MEAD_flame2020'
    rec_path = os.path.join(rec_folder, test_video_id, 'shape_pose_cam.hdf5')
    emica_code = {}
    with h5py.File(rec_path, "r") as f:
        for k in f.keys():
            emica_code[k] = torch.from_numpy(f[k][0]).float()
            print(f"shape of {k} : {emica_code[k].shape}")

    tex_path = os.path.join(rec_folder, test_video_id, 'appearance.hdf5')
    with h5py.File(tex_path, "r") as f:
        for k in f.keys():
            emica_code[k] = torch.from_numpy(f[k][0]).float()
            print(f"shape of {k} : {emica_code[k].shape}")
    emica_code['light'] = emica_code['light'].reshape(-1, 9, 3)
    
    motion_tracker = MotionTracker(
        pretrained_args.model,
        sld_wind_size=32, 
        device='cuda', 
        with_audio=False,
        use_tex=False)
    
    motion_tracker.run_vis(data_dict, emica_code, test_video_id)

if __name__ == "__main__":
    main()