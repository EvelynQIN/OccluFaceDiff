### Taken from https://github.com/Zielon/MICA/blob/master/models/arcface.py
# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import os
import sys

sys.path.append("./micalib")

import torch
import torch.nn as nn
import torch.nn.functional as F

from micalib.arcface import Arcface
from micalib.generator import Generator


class MICA(nn.Module):
    def __init__(self, config=None, tag='MICA'):
        super(MICA, self).__init__()
        self.cfg = config
        self.tag = tag
        self.initialize()

    def initialize(self):
        self.create_model(self.cfg)

    def create_model(self, model_cfg):
        mapping_layers = model_cfg.mapping_layers
        pretrained_path = None
        if not model_cfg.use_pretrained:
            pretrained_path = model_cfg.arcface_pretrained_model
        
        # get the identity code for each image
        self.arcface = Arcface(pretrained_path=pretrained_path)

        # regress shape params from identity code
        self.flameModel = Generator(512, 300, model_cfg.n_shape, mapping_layers)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # freeze all the params in mica
        self.freezer([self.arcface, self.flameModel])

    def freezer(self, layers):
        for layer in layers:
            for block in layer.parameters():
                block.requires_grad = False
        print(f'[{self.tag}] All the parameters of MICA are frozen.')
    
    def load_model(self, device):
        if os.path.exists(self.cfg.pretrained_model_path) and self.cfg.use_pretrained:
            model_path = self.cfg.pretrained_model_path
        if os.path.exists(model_path):
            print(f'[{self.tag}] Trained model found. Path: {model_path}')
            checkpoint = torch.load(model_path, map_location=device)
            if 'arcface' in checkpoint:
                print(f'[{self.tag}] Trained model found for arcface.')
                self.arcface.load_state_dict(checkpoint['arcface'])
            if 'flameModel' in checkpoint:
                print(f'[{self.tag}] Trained model found for generator.')
                self.flameModel.load_state_dict(checkpoint['flameModel'], strict=False) # skip flame related keys
        else:
            print(f'[{self.tag}] Checkpoint not available starting from scratch!')

    def model_dict(self):
        return {
            'flameModel': self.flameModel.state_dict(),
            'arcface': self.arcface.state_dict()
        }

    def parameters_to_optimize(self):
        return [
            {'params': self.flameModel.parameters(), 'lr': self.cfg.lr},
            {'params': self.arcface.parameters(), 'lr': self.cfg.arcface_lr},
        ]

    def encode(self, arcface_imgs):
        """
        Args:
            arcface_imgs: (bs, 4, 3, 112, 112)
        Returns:
            identity_code: (bs, 512)
        """
        bs, n = arcface_imgs.shape[:2]
        arcface_imgs = arcface_imgs.reshape(-1, 3, 112, 112)

        identity_codes = F.normalize(self.arcface(arcface_imgs)).reshape(bs, n, -1) # (bs, n, 512)
        identity_codes = identity_codes.transpose(1, 2) # (bs, 512, n)
        identity_code = self.avg_pool(identity_codes).squeeze(2) # (bs, 512)
        return identity_code

    def decode(self, identity_code):

        pred_shape_code = self.flameModel(identity_code) # (bs, 300)

        return pred_shape_code
    
    def forward(self, arcface_imgs):

        identity_code = self.encode(arcface_imgs)

        pred_shape_code = self.decode(identity_code)

        return pred_shape_code  # (bs, 300)