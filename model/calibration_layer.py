import numpy as np
import torch
import torch.nn as nn 
import os

class Cam_Calibration(nn.Module):
    def __init__(
        self,
        lmk2d_dim, # input feature dim 68 x 2
        n_target,
        output_feature_dim, # number of cam params (one set per frame)
        latent_dim,
        ckpt_path=None,
    ):
        super().__init__()

        # condition dim
        self.lmk2d_dim = lmk2d_dim
        self.n_target = n_target

        # output dim
        self.output_feature_dim = output_feature_dim 
        self.latent_dim = latent_dim 
        self.ckpt_path = ckpt_path
        
        self.tag = 'CAM'
        
        self.flame_process = nn.Sequential(
            nn.Linear(self.n_target, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.LeakyReLU(inplace=True)
        )
        
        self.lmk_process = nn.Sequential(
            nn.Linear(self.lmk2d_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.LeakyReLU(inplace=True)
        )

        self.net = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.latent_dim, self.output_feature_dim)
        )
        if self.ckpt_path is not None:
            self.load_model()
            self.freezer([self.flame_process, self.lmk_process, self.net])

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
        print(f'[{self.tag}] All params are frozen.')

    def forward(self, lmk_2d, target):
        """
        Args:
            lmk2d: [batch_size, 68x2]
            target: [batch_size, 180]
        Return:
            cam_params: [batch_size, output_feature_dim]
        """
        lmk2d_emb = self.lmk_process(lmk_2d)
        target_emb = self.flame_process(target)
        input_emb = torch.cat([lmk2d_emb, target_emb], dim=-1)
        output = self.net(input_emb)
        
        return output