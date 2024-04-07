# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import json
import os
import random

import numpy as np

import torch

from data_loaders.dataloader import get_dataloader, load_data, TrainDataset
from model.FLAME import FLAME
from model.networks import PureMLP
# from runner.train_mlp import train_step, val_step
from runner.training_loop import TrainLoop

from utils import dist_util

from utils.model_util import create_model_and_diffusion
from utils.parser_util import train_mlp_args
from utils.config import Config
import wandb
from tqdm import tqdm
from configs.config import get_cfg_defaults


def train_diffusion_model(args, model_cfg, train_dataloader, val_dataloader):
    print("creating model and diffusion...")

    dist_util.setup_dist(args.device)
    denoise_model, diffusion = create_model_and_diffusion(args, model_cfg, dist_util.dev()) # the denoising MLP & spaced diffusion
    denoise_model.to(dist_util.dev())
    print(
        "Total trainable params: %.2fM"
        % (sum(p.numel() for p in denoise_model.parameters() if p.requires_grad) / 1000000.0)
    )

    print("Training...")
    TrainLoop(args, model_cfg, denoise_model, diffusion, train_dataloader, val_dataloader).run_loop()
    print("Done.")

def set_deterministic(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    args = train_mlp_args() 
    kwargs = dict(args._get_kwargs())
    cfg_path = args.config_path
    args = Config(default_cfg_path=cfg_path, **kwargs)

    pretrained_args = get_cfg_defaults()

    set_deterministic(args.seed)

    model_type = args.arch.split('_')[0]
    args.save_dir = os.path.join(args.save_dir, args.arch[len(model_type)+1:])
    
    # init wandb log
    if args.wandb_log:
        wandb.init(
            project="face_animation_from_image",
            name=args.arch,
            config=args,
            settings=wandb.Settings(start_method="fork"),
            dir="./wandb"
        )
    
    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError("save_dir [{}] already exists.".format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(dict(args), fw, indent=4, sort_keys=True) 
    
    train_image_path, train_processed_path = load_data(
        args.dataset,
        args.dataset_path,
        "train",
    )
    train_dataset = TrainDataset(
        args.dataset,
        train_image_path,
        train_processed_path,
        args.input_motion_length,
        args.train_dataset_repeat_times,
        args.no_normalization,
        args.occlusion_mask_prob,
        args.mixed_occlusion_prob,
        args.fps
    )
    
    train_loader = get_dataloader(
        train_dataset, "train", batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # val data loader
    print("creating val data loader...")
    val_image_path, val_processed_path = load_data(
        args.dataset,
        args.dataset_path,
        "test",
    )
    
    val_dataset = TrainDataset(
        args.dataset,
        val_image_path, 
        val_processed_path,
        args.input_motion_length,
        5,
        args.no_normalization,
        args.occlusion_mask_prob,
        args.mixed_occlusion_prob,
        args.fps
    )
    
    val_loader = get_dataloader(
        val_dataset, "val", batch_size=args.batch_size, num_workers=args.num_workers
    )

    train_diffusion_model(args, pretrained_args.model, train_loader, val_loader)

if __name__ == "__main__":
    main()
