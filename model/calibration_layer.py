import numpy as np
import torch
import torch.nn as nn 
import os

class Cam_Calibration(nn.Module):
    def __init__(
        self,
        input_feature_dim, # input feature dim 68 x 2
        output_feature_dim, # number of cam params (one set per frame)
        latent_dim,
        ckpt_path,
    ):
        super().__init__()

        self.input_feature_dim = input_feature_dim 
        self.output_feature_dim = output_feature_dim 
        self.latent_dim = latent_dim 
        self.ckpt_path = ckpt_path
        self.tag = 'CAM'
        
        self.net = nn.Sequential(
            nn.Linear(self.input_feature_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim // 2, self.output_feature_dim)
        )

        self.load_model()

        self.freezer([self.net])

    def load_model(self):
        if os.path.exists(self.ckpt_path):
            print(f'[{self.tag}] Trained model found. Path: {self.ckpt_path}')
            ckpt = torch.load(self.ckpt_path)
            self.load_state_dict(ckpt)
        else:
            print(f'[{self.tag}] Checkpoint not available starting from scratch!')
    
    def freezer(self, layers):
        for layer in layers:
            for block in layer.parameters():
                block.requires_grad = False

    def forward(self, lmk2d):
        """
        Args:
            lmk2d: [batch_size, 68x2]
        Return:
            cam_params: [batch_size, output_feature_dim]
        """
        output = self.net(lmk2d)
        
        return output