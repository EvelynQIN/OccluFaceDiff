import argparse
import os

from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Settings for pretrained models
# ---------------------------------------------------------------------------- #

cfg = CN()
cfg.save_dir = "./checkpoints" 
cfg.num_epoch = 200
cfg.dataset = "mead_25fps" 
cfg.dataset_path = "./dataset"
cfg.weight_decay = 0.001
cfg.batch_size = 1
cfg.gradient_accumulation_steps = 256
cfg.lr = 0.0003
cfg.cosine_scheduler = False
cfg.warmup_steps = 100
cfg.device = 0 # -1 if using cpu for test
cfg.num_workers = 4
cfg.wandb_log = True
cfg.save_interval = 2
cfg.log_interval = 1 
cfg.resume_checkpoint = None    #'checkpoints/Emotion_Classifier_Transformer_128d_4l/model_18.pt'
cfg.seed = 0

# model
cfg.arch = "Emotion_Classifier_Transformer_128d_4l_wotest" 
cfg.latent_dim = 128
cfg.heads = 4
cfg.ff_size = 256
cfg.layers = 4
cfg.dropout = 0.1
cfg.n_shape = 100
cfg.n_exp = 50
cfg.input_dim = 153
cfg.num_classes = 22
cfg.max_pool = True

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