import argparse
import os

from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Settings for pretrained models
# ---------------------------------------------------------------------------- #

cfg = CN()

# ---------------------------------------------------------------------------- #
# Defaults for MICA
# ---------------------------------------------------------------------------- #
cfg.mica = CN()
cfg.mica.name = 'mica'
cfg.mica.n_shape = 300
cfg.mica.layers = 8
cfg.mica.hidden_layers_size = 256
cfg.mica.mapping_layers = 3
cfg.mica.use_pretrained = True
cfg.mica.pretrained_model_path = './pretrained/mica.tar'

cfg.mica.lr = 0.0001
cfg.mica.arcface_lr = 0.001
# ---------------------------------------------------------------------------- #
# Defaults for Cam_Calib 
# ---------------------------------------------------------------------------- #
cfg.cam = CN()
cfg.cam.model_name = "camT_prediction"
cfg.cam.lmk2d_dim = 136 # input feature dim 68 x 2
cfg.cam.n_target = 180
cfg.cam.output_nfeat = 3 # number of cam params (one set per frame)
cfg.cam.latent_dim = 128
cfg.cam.focal_length = 1000.0
cfg.cam.principal_point = 112.0
cfg.cam.trans_offset = [0.004, 0.222, 1.200]
cfg.cam.ckpt_path = "pretrained/cam_calib_shape.pth"

def get_cfg_defaults():
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path', required=True)

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg