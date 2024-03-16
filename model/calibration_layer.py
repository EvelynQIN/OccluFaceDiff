import numpy as np
import torch
import torch.nn as nn 
import os

class Cam_Calibration(nn.Module):
    def __init__(
        self,
        lmk2d_dim, # input feature dim 68 x 2
        n_shape,
        output_feature_dim, # number of cam params (one set per frame)
        latent_dim,
        ckpt_path=None,
    ):
        super().__init__()

        # condition dim
        self.lmk2d_dim = lmk2d_dim
        self.n_shape = n_shape

        # output dim
        self.output_feature_dim = output_feature_dim 
        self.latent_dim = latent_dim 
        self.ckpt_path = ckpt_path
        
        self.tag = 'CAM'
        
        self.shape_process = nn.Linear(self.n_shape, self.latent_dim)
        
        self.lmk_process = nn.Linear(self.lmk2d_dim, self.latent_dim)

        self.net = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.output_feature_dim)
        )
        if self.ckpt_path is not None:
            self.load_model()
            self.freezer([self.shape_process, self.lmk_process, self.net])

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

    def forward(self, lmk2d, shape_100):
        """
        Args:
            lmk2d: [batch_size, 68x2]
            shape_100: [batch_size, 100]
        Return:
            cam_params: [batch_size, output_feature_dim]
        """
        lmk2d_emb = self.lmk_process(lmk2d)
        shape_100_emb = self.shape_process(shape_100)
        input_emb = torch.cat([lmk2d_emb, shape_100_emb], dim=-1)
        output = self.net(input_emb)
        
        return output