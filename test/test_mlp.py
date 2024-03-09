
import os

import cv2
import numpy as np
import trimesh
from tqdm import tqdm
import glob 
import ffmpeg
import gc
import pyrender
from vedo import trimesh2vedo, show
from PIL import Image
import torch
import pickle
from utils import utils_transform
from model.meta_model import MultiBranchMLP_CategoryEmb

if __name__ == "__main__":
    # kwargs = {
    #     "arch": None,
    #     "nfeats": 124,
    #     "latent_dim": 1024,
    #     "sparse_dim": 176*3,
    #     "dropout": 0.1,
    #     "cond_mask_prob": 0,
    #     "dataset": "FaMoS",
    #     "ff_size": 1024, 
    #     "num_enc_layers": 2, 
    #     "num_dec_layers": 2,
    #     "num_heads":4, 
    # }
    # device = "cuda"
    # model = Face_Transformer(**kwargs).to(device)
    # bs = 1
    
    # x = torch.rand(bs, 150, kwargs["nfeats"]).to(device)
    # timesteps = torch.randint(low=1, high=1000, size = (bs,)).to(device)
    # sparse = torch.rand(bs, 150, kwargs["sparse_dim"]).to(device)
    # output = model(x, timesteps, sparse)
    # print(output.shape)


    kwargs = {
        "arch": "diffusion_MLP",
        "nfeats": 124,
        "cond_latent_dim": 256,
        "pose_latent_dim": 64,
        "expr_latent_dim": 256,
        "sparse_dim": 176*3,
        "pose_num_layers": 4,
        "expr_num_layers": 5,
        "dropout": 0.1,
        "cond_mask_prob": 0,
        "dataset": "FaMoS",
        "input_motion_length": 150,
    }
    device = "cuda"
    model = MultiBranchMLP_CategoryEmb(**kwargs).to(device)
    bs = 1
    
    x = torch.rand(bs, 150, kwargs["nfeats"]).to(device)
    timesteps = torch.randint(low=1, high=1000, size = (bs,)).to(device)
    sparse = torch.rand(bs, 150, kwargs["sparse_dim"]).to(device)
    output = model(x, timesteps, sparse)
    print(output.shape)
        