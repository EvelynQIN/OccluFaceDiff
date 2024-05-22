import torch
import torch.nn.functional as F
from utils import utils_transform
from utils.data_util import batch_orth_proj
from model.FLAME import FLAME, FLAMETex, FLAME_mediapipe
from model.deca import ExpressionLossNet
import numpy as np 
from utils.renderer import SRenderY
from skimage.io import imread
import torchvision.transforms.functional as F_v
import pickle
import sys
sys.path.append("external/Visual_Speech_Recognition_for_Multiple_Languages")
from external.Visual_Speech_Recognition_for_Multiple_Languages.lipreading.model import Lipreading
from external.Visual_Speech_Recognition_for_Multiple_Languages.dataloader.transform import Compose, Normalize, CenterCrop, SpeedRate, Identity
from configparser import ConfigParser
from utils.MediaPipeLandmarkLists import *
from model.motion_prior import L2lVqVae
from munch import Munch, munchify

class TransformerLoss:
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self._set_up()
    
    def _load_flint_decoder(self):
        ckpt_path = self.model_cfg.flint_ckpt_path
        f = open(self.model_cfg.flint_config_path)
        cfg = Munch.fromYAML(f)
        flint = L2lVqVae(cfg)
        flint.load_model_from_checkpoint(ckpt_path)
        self.flint_decoder = flint.motion_decoder.to(self.device)
        print(f"[FLINT Decoder] Load and Frozen.")
        self.flint_decoder.requires_grad_(False)
        self.flint_decoder.eval()

    def _setup(self):
        self.train_stage = self.model_cfg.train_stage
        print(f"[Diffusion] Set up training stage {self.train_stage}")

        # at train stage 1, only optimize over mesh verts
        if self.train_stage == 1:
            self._create_flame()
            self._load_flint_decoder()

            # set up loss weight
            self.loss_weight = {
                'expr_loss': 0.,
                'pose_loss': 0.,
                'latent_rec_loss': 1.0,
                'mesh_verts_loss': 100.,
                'lmk3d_loss': 0.,
                'lmk2d_loss': 0.05,
                'mouth_closure_loss': 0.01,
                'emotion_loss': 0,
                'lipread_loss': 0
            }
            print(f"[Diffusion] Loss weights used: {self.loss_weight}")

        # at train stage 2, optimize over rendering loss
        elif self.train_stage == 2:
            self._create_flame()
            self._load_flint_decoder()
            self._setup_renderer()
            self._load_evalnet()

            # set up loss weight
            self.loss_weight = {
                'expr_loss': 0.,
                'pose_loss': 0.,
                'latent_rec_loss': 0.,
                'mesh_verts_loss': 10.,
                'lmk3d_loss': 0.,
                'lmk2d_loss': 1.,
                'mouth_closure_loss': 0.01,
                'emotion_loss': 0.1,
                'lipread_loss': 0.5
            }
            print(f"[Diffusion] Loss weights used: {self.loss_weight}")
        else:
            raise ValueError(f"Traing stage[ {self.train_stage}] not valid!")
    
    def _create_flame(self):
        print(f"[Diffusion] Load FLAME.")
        self.flame = FLAME_mediapipe(self.model_cfg).to(self.device)
        if self.train_stage == 2 and self.model_cfg.use_texture:
            self.flametex = FLAMETex(self.model_cfg).to(self.device)

    def _setup_renderer(self):
        print(f"[Diffusion] Set up Renderer.")
        self.render = SRenderY(
            self.model_cfg.image_size, 
            obj_filename=self.model_cfg.topology_path, 
            uv_size=self.model_cfg.uv_size).to(self.device)
        # face mask for rendering details
        mask = imread(self.model_cfg.face_eye_mask_path).astype(np.float32)/255. 
        mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [self.model_cfg.uv_size, self.model_cfg.uv_size]).to(self.device)
    
    def _load_evalnet(self):
 
        # ----- load resnet trained from EMOCA https://github.com/radekd91/emoca for expression loss ----- #
        self.expression_net = ExpressionLossNet().to(self.device)
        self.emotion_checkpoint = torch.load(self.model_cfg.expression_net_path)['state_dict']
        self.emotion_checkpoint['linear.0.weight'] = self.emotion_checkpoint['linear.weight']
        self.emotion_checkpoint['linear.0.bias'] = self.emotion_checkpoint['linear.bias']

        print(f"[Diffusion] Load emotion net.")
        self.expression_net.load_state_dict(self.emotion_checkpoint, strict=False)
        self.expression_net.eval()
        for param in self.expression_net.parameters():
            param.requires_grad = False

        # ----- load lipreader network for lipread loss ----- #
        config = ConfigParser()

        config.read('configs/lipread_config.ini')
        self.lip_reader = Lipreading(
            config,
            device=self.device
            
        )
        self.lip_reader.eval()
        self.lip_reader.model.eval()
        for param in self.lip_reader.parameters():
            param.requires_grad = False

        # ---- initialize values for cropping the face around the mouth for lipread loss ---- #
        # ---- this code is borrowed from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages ---- #
        self._crop_width = 96
        self._crop_height = 96
        self._window_margin = 12
        self._lip_idx = torch.from_numpy(LIP_EM).long()
        # self._start_idx = 48
        # self._stop_idx = 68
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)

        # ---- transform mouths before going into the lipread network for loss ---- #
        self.mouth_transform = Compose([
            Normalize(0.0, 1.0),
            CenterCrop(crop_size),
            Normalize(mean, std),
            Identity()]
        )

    def cut_mouth_vectorized(
        self,
        images, 
        landmarks, 
        convert_grayscale=True
        ):
                
        with torch.no_grad():

            landmarks = landmarks * 112 + 112
            # #1) smooth the landmarks with temporal convolution
            # landmarks are of shape (T, 68, 2) 
            landmarks_t = landmarks.reshape(*landmarks.shape[:2], -1)   # （bs, t, 468x2）
            # make temporal dimension last 
            landmarks_t = landmarks_t.permute(0, 2, 1)  # (bs, 468x2, t)

            # smooth with temporal convolution
            temporal_filter = torch.ones(self._window_margin, device=images.device) / self._window_margin
            # pad the the landmarks 
            landmarks_t_padded = F.pad(landmarks_t, (self._window_margin // 2, self._window_margin // 2), mode='replicate')
            # convolve each channel separately with the temporal filter
            num_channels = landmarks_t.shape[1]
            smooth_landmarks_t = F.conv1d(landmarks_t_padded, 
                temporal_filter.unsqueeze(0).unsqueeze(0).expand(num_channels,1,temporal_filter.numel()), 
                groups=num_channels, padding='valid'
            )
            smooth_landmarks_t = smooth_landmarks_t[..., 0:landmarks_t.shape[-1]]

            # reshape back to the original shape 
            smooth_landmarks_t = smooth_landmarks_t.permute(0, 2, 1).view(landmarks.shape)
            smooth_landmarks_t = smooth_landmarks_t + landmarks.mean(dim=2, keepdims=True) - smooth_landmarks_t.mean(dim=2, keepdims=True)

            # #2) get the mouth landmarks
            mouth_landmarks_t = smooth_landmarks_t[..., self._lip_idx, :]
            
            # #3) get the mean of the mouth landmarks
            mouth_landmarks_mean_t = mouth_landmarks_t.mean(dim=-2, keepdims=True)
        
            # #4) get the center of the mouth
            center_x_t = mouth_landmarks_mean_t[..., 0]
            center_y_t = mouth_landmarks_mean_t[..., 1]

            # #5) use grid_sample to crop the mouth in every image 
            # create the grid
            height = self._crop_height//2
            width = self._crop_width//2

            # torch.arange(0, mouth_crop_width, device=images.device)

            grid = torch.stack(torch.meshgrid(torch.linspace(-height, height, self._crop_height).to(images.device) / 112,
                                            torch.linspace(-width, width, self._crop_width).to(images.device) / 112 ), 
                               dim=-1)
            grid = grid[..., [1, 0]]
            grid = grid.unsqueeze(0).unsqueeze(0).repeat(*images.shape[:2], 1, 1, 1)    # (bs, n, 1, 9, 2)

            # normalize the center to [-1, 1]
            center_x_t = (center_x_t - 112) / 112
            center_y_t = (center_y_t - 112) / 112

            center_xy =  torch.cat([center_x_t, center_y_t ], dim=-1).unsqueeze(-2).unsqueeze(-2)
            if center_xy.ndim != grid.ndim:
                center_xy = center_xy.unsqueeze(-2)
            assert grid.ndim == center_xy.ndim, f"grid and center_xy have different number of dimensions: {grid.ndim} and {center_xy.ndim}"
            grid = grid + center_xy
        B, T = images.shape[:2]
        images = images.view(B*T, *images.shape[2:])
        grid = grid.view(B*T, *grid.shape[2:])

        if convert_grayscale: 
            images = F_v.rgb_to_grayscale(images)
        
            image_crops = F.grid_sample(
                images, 
                grid,  
                align_corners=True, 
                padding_mode='zeros',
                mode='bicubic'
            )
        
        # image_crops = image_crops.view(B, T, *image_crops.shape[1:])

        # if convert_grayscale:
        #     image_crops = image_crops#.squeeze(1)
        return image_crops.squeeze(1)   # (bs*t, 96, 96)
    
    def batch_lmk2d_loss(self, lmk2d_pred, lmk2d_gt):
        """
        Computes the l1 loss between the ground truth keypoints and the predicted keypoints
        Inputs:
        lmk2d_gt  : N x K x 2
        lmk2d_pred: N x K x 2
        """
        kp_gt = lmk2d_gt.view(-1, 2)
        kp_pred = lmk2d_pred.contiguous().view(-1, 2)

        diff_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)

        return diff_abs.mean()
        
    # loss computation between target and prediction where 2d images are available
    # @profile
    def compute_loss(self, target, model_output, **model_kwargs):  

        loss_dict = {}
        
        latent_rec_loss = F.mse_loss(target, model_output)
        # flint decode
        pred_flame = self.flint_decoder(model_output)   # (bs, n, 53)
        bs, n = pred_flame.shape[:2]
        
        # parse the output and target
        jaw_pred, expr_pred = pred_flame[...,self.model_cfg.n_exp:], pred_flame[...,:self.model_cfg.n_exp]

         # l2 loss on flame parameters w.r.t deca's output             
        pose_loss = F.mse_loss(jaw_pred, model_kwargs['jaw'])
        expr_loss = F.mse_loss(expr_pred, model_kwargs['exp'])
        
        # batch the output
        shape = model_kwargs['shape'].view(bs*n, -1)
        R_aa = model_kwargs['global_pose'].view(bs*n, -1)
        expr_pred = expr_pred.reshape(bs*n, -1)
        pose_pred = torch.cat([R_aa, jaw_pred.reshape(bs*n, -1)], dim=-1)
        cam = model_kwargs['cam'].view(bs*n, -1)

        expr_gt = model_kwargs['exp'].reshape(bs*n, -1)
        pose_gt = torch.cat([R_aa, model_kwargs['jaw'].reshape(bs*n, -1)], dim=-1)
        
        # flame decoder
        verts_pred, lmk3d_pred = self.flame(
            shape_params=shape, 
            expression_params=expr_pred,
            pose_params=pose_pred)
        
        verts_gt, lmk3d_gt = self.flame(
            shape_params=shape, 
            expression_params=expr_gt,
            pose_params=pose_gt)
        
        # l2 on mesh verts
        mesh_verts_loss = torch.mean(
            torch.norm(verts_gt - verts_pred, p=2, dim=-1)
        )

        # orthogonal projection
        lmk_pred = batch_orth_proj(lmk3d_pred, cam)[:, :, :2]
        lmk_pred[:, :, 1:] = -lmk_pred[:, :, 1:]

        # ---- geometric losses ---- #
        lmk_gt = model_kwargs['lmk_2d'][:,:,self.flame.landmark_indices_mediapipe].view(bs*n, *lmk_pred.shape[-2:])
        lmk2d_loss = self.batch_lmk2d_loss(lmk_pred, lmk_gt)  
        lmk3d_loss = torch.mean(torch.norm(lmk3d_pred - lmk3d_gt, p=2, dim=-1))
        # eye_closure_loss = eye_closure_lmk_loss(lmk_pred, lmk_gt)
        mouth_closure_loss = mouth_closure_lmk_loss(lmk_pred, lmk_gt)

        emotion_loss = 0
        lipread_loss = 0

        if self.train_stage == 2:

            trans_verts = batch_orth_proj(verts_pred, cam)
            trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

            # rendering
            light = model_kwargs['light'].view(bs*n, 9, 3)
            
            # if not using texture, default to gray
            if self.model_cfg.use_texture:
                tex = model_kwargs['tex'].view(bs*n, -1)
                albedo = self.flametex(tex).detach()
            else:
                albedo = torch.ones([bs*n, 3, self.model_cfg.uv_size, self.model_cfg.uv_size], device=light.device) * 0.5

            images = model_kwargs['image'].view(bs*n, *model_kwargs['image'].shape[2:])

            pred_render_img = self.render(verts_pred, trans_verts, albedo, light)['images']
            
            #  # ---- photometric loss ---- #
            # mask_face_eye = F.grid_sample(self.uv_face_eye_mask.expand(bs*n,-1,-1,-1), ops['grid'].detach(), align_corners=False)
            # photometric_loss = torch.mean(mask_face_eye * ops['alpha_images'] * (ops['images'] - images).abs())
            
        
            # # ---- emotion loss from EMOCA ---- #
            with torch.no_grad():
                emotion_features_gt = self.expression_net(images)
            emotion_features_pred = self.expression_net(pred_render_img)

            emotion_loss = F.mse_loss(emotion_features_pred, emotion_features_gt)

        
            # ---- lipread loss ---- #
            # first crop the mouths of the input and rendered faces and then calculate the distance of features 
            images = images.view(bs,n, *images.shape[1:])
            lmk_gt = lmk_gt.view(bs, n, *lmk_gt.shape[1:])
            lmk_pred = lmk_pred.view(bs, n, *lmk_pred.shape[1:])
            pred_render_img = pred_render_img.view(bs, n, *pred_render_img.shape[1:])
            mouths_gt = self.cut_mouth_vectorized(images, lmk_gt[...,:2])   # (bs*t, 96, 96)
            mouths_pred = self.cut_mouth_vectorized(pred_render_img, lmk_pred[...,:2])
            mouths_gt = self.mouth_transform(mouths_gt) # (bs*t, 88, 88)
            mouths_pred = self.mouth_transform(mouths_pred)

            # # # ---- resize back to BxNx1xHxW (grayscale input for lipread net) ---- #
            mouths_gt = mouths_gt.view(bs, n, *mouths_gt.shape[1:]) # (bs, n, 88, 88)
            mouths_pred = mouths_pred.view(bs, n, *mouths_pred.shape[1:]) # (bs, n, 88, 88)

            with torch.no_grad():
                lip_features_gt = self.lip_reader.model.encoder(
                    mouths_gt,
                    None,
                    extract_resnet_feats=True
                )
                
            lip_features_pred = self.lip_reader.model.encoder(
                mouths_pred,
                None,
                extract_resnet_feats=True
            )

            lipread_loss = F.mse_loss(lip_features_gt, lip_features_pred)
        loss = self.loss_weight['lmk2d_loss'] * lmk2d_loss + self.loss_weight['mouth_closure_loss'] * mouth_closure_loss \
            + self.loss_weight['expr_loss'] * expr_loss + self.loss_weight['pose_loss'] * pose_loss \
            + self.loss_weight['lmk3d_loss'] * lmk3d_loss \
            + self.loss_weight['latent_rec_loss'] * latent_rec_loss + self.loss_weight['mesh_verts_loss'] * mesh_verts_loss \
            + self.loss_weight['emotion_loss'] * emotion_loss + self.loss_weight['lipread_loss'] * lipread_loss
        # + self.loss_weight['expr_vel_loss'] * expr_vel_loss + self.loss_weight['pose_vel_loss'] * pose_vel_loss \
        
        loss_dict = {
            'expr_loss': expr_loss.detach().item(),
            'pose_loss': pose_loss.detach().item(),
            'latent_rec_loss': latent_rec_loss.detach().item(),
            'mesh_verts_loss': mesh_verts_loss.detach().item(),          
            'lmk3d_loss': lmk3d_loss.detach().item(),
            'lmk2d_loss': lmk2d_loss.detach().item(),
            'mouth_closure_loss': mouth_closure_loss.detach().item(),
            # 'eye_closure_loss': eye_closure_loss.detach().item(),
            # 'photometric_loss': photometric_loss.detach().item(),
            'loss': loss.detach().item(),
        }

        if self.train_stage == 2:
            loss_dict.update({
                'emotion_loss': emotion_loss.detach().item(),
                'lipread_loss' : lipread_loss.detach().item(),})

        return loss, loss_dict
