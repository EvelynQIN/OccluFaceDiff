"""Copied from https://github.com/filby89/spectre
"""
# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.renderer import SRenderY
from model.deca import ResnetEncoder
from model.FLAME import FLAME, FLAMETex
from utils.data_util import batch_orth_proj
from skimage.io import imread
torch.backends.cudnn.benchmark = True
import numpy as np
from model import resnet
from utils.utils_visualize import tensor_vis_landmarks

class PerceptualEncoder(nn.Module):
    def __init__(self, outsize, cfg):
        super(PerceptualEncoder, self).__init__()
        if cfg.spectre_backbone == "mobilenetv2":
            self.encoder = torch.hub.load('pytorch/vision:v0.8.1', 'mobilenet_v2', pretrained=True)
            feature_size = 1280
        elif cfg.spectre_backbone == "resnet50":
            self.encoder = resnet.load_ResNet50Model() #out: 2048
            feature_size = 2048

        ### regressor
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels=feature_size, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            nn.Linear(256, 53),
        )

        self.backbone = cfg.spectre_backbone

    def forward(self, inputs):
        is_video_batch = inputs.ndim == 5

        if self.backbone == 'resnet50':
            features = self.encoder(inputs).squeeze(-1).squeeze(-1)
        else:
            inputs_ = inputs
            if is_video_batch:
                B, T, C, H, W = inputs.shape
                inputs_ = inputs.view(B * T, C, H, W)
            features = self.encoder.features(inputs_)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
            if is_video_batch:
                features = features.view(B, T, -1)

        features = features
        if is_video_batch:
            features = features.permute(0, 2, 1)
        else:
            features = features.permute(1,0).unsqueeze(0)

        features = self.temporal(features)

        if is_video_batch:
            features = features.permute(0, 2, 1)
        else:
            features = features.squeeze(0).permute(1,0)

        parameters = self.layers(features)

        parameters[...,50] = F.relu(parameters[...,50]) # jaw x is highly improbably negative and can introduce artifacts

        return parameters[...,:50], parameters[...,50:]

class SPECTRE(nn.Module):
    def __init__(self, config=None, device='cuda'):
        super(SPECTRE, self).__init__()
        self.cfg = config
        self.device = device
        self.image_size = self.cfg.image_size
        self._create_model(self.cfg)

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param = model_cfg.n_shape + model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light
        self.n_cond = model_cfg.n_exp + 3  # exp + jaw pose
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam,
                         model_cfg.n_light]
        self.param_dict = {i: model_cfg.get('n_' + i) for i in model_cfg.param_list}

        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)

        self.E_expression = PerceptualEncoder(model_cfg.n_exp, model_cfg).to(self.device)

        # resume model
        model_path = model_cfg.pretrained_spectre_path
        if os.path.exists(model_path):
            # print(f'trained model found. load {model_path}')
            checkpoint = torch.load(model_path)

            if 'state_dict' in checkpoint.keys():
                self.checkpoint = checkpoint['state_dict']
            else:
                self.checkpoint = checkpoint

            processed_checkpoint = {}
            processed_checkpoint["E_flame"] = {}
            processed_checkpoint["E_expression"] = {}
            if 'deca' in list(self.checkpoint.keys())[0]:
                for key in self.checkpoint.keys():
                    # print(key)
                    k = key.replace("deca.","")
                    if "E_flame" in key:
                        processed_checkpoint["E_flame"][k.replace("E_flame.","")] = self.checkpoint[key]#.replace("E_flame","")
                    elif "E_expression" in key:
                        processed_checkpoint["E_expression"][k.replace("E_expression.","")] = self.checkpoint[key]#.replace("E_flame","")
                    else:
                        pass

            else:
                processed_checkpoint = self.checkpoint


            self.E_flame.load_state_dict(processed_checkpoint['E_flame'], strict=True)
            try:
                m,u = self.E_expression.load_state_dict(processed_checkpoint['E_expression'], strict=True)
                # print('Missing keys', m)
                # print('Unexpected keys', u)
                # pass
            except Exception as e:
                print(f'Missing keys {e} in expression encoder weights. If starting training from scratch this is normal.')
        else:
            raise(f'please check model path: {model_path}')

        # eval mode
        self.E_flame.eval()

        self.E_expression.eval()

        self.E_flame.requires_grad_(False)


    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0

        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[..., start:end]
            start = end
            if key == 'light':
                dims_ = code_dict[key].ndim -1 # (to be able to handle batches of videos)
                code_dict[key] = code_dict[key].reshape(*code_dict[key].shape[:dims_], 9, 3)
        return code_dict

    def encode(self, images):
        with torch.no_grad():
            parameters = self.E_flame(images)

        codedict = self.decompose_code(parameters, self.param_dict)
        deca_exp = codedict['exp'].clone()
        deca_jaw = codedict['pose'][...,3:].clone()

        # codedict['images'] = images

        codedict['exp'], jaw = self.E_expression(images)
        codedict['pose'][..., 3:] = jaw


        # follow the official implementation to add predicted residuals to the initial deca reconstruction
        codedict['exp'] = codedict['exp'] + deca_exp
        codedict['pose'][..., 3:] = codedict['pose'][..., 3:] + deca_jaw

        return codedict

    def train(self):
        self.E_expression.train()

        self.E_flame.eval()


    def eval(self):
        self.E_expression.eval()
        self.E_flame.eval()


    def model_dict(self):
        return {
            'E_flame': self.E_flame.state_dict(),
            'E_expression': self.E_expression.state_dict(),
        }
