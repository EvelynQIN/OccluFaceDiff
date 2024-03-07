# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import json
import os
import random

import numpy as np

import torch

from data_loaders.dataloader import get_dataloader, load_data, TrainDataset
from model.FLAME import FLAME
from runner.training_loop import TrainLoop

from utils import dist_util

from utils.model_util import create_model_and_diffusionTransformer
from utils.parser_util import train_trans_args
from utils.config import Config
import wandb
from tqdm import tqdm


def train_diffusion_model(args, train_dataloader, val_dataloader, mean, std):
    print("creating model and diffusion...")

    num_gpus = torch.cuda.device_count()
    args.num_workers = args.num_workers * num_gpus

    model, diffusion = create_model_and_diffusionTransformer(args) # the denoising model & spaced diffusion

    if num_gpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        dist_util.setup_dist()
        model = torch.nn.DataParallel(model).cuda()
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.module.parameters()) / 1000000.0)
        )
    else:
        dist_util.setup_dist(args.device)
        model.to(dist_util.dev())
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )

    print("Training...")
    TrainLoop(args, model, diffusion, train_dataloader, val_dataloader, mean, std).run_loop()
    print("Done.")

def main():
    args = train_trans_args() 
    kwargs = dict(args._get_kwargs())
    
    cfg_path = args.config_path
    args = Config(default_cfg_path=cfg_path, **kwargs)

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_type = args.arch.split('_')[0]
    args.save_dir = os.path.join(args.save_dir, args.arch[len(model_type)+1:])
    
    
     # init wandb log
    if args.wandb_log:
        wandb.init(
            project="face_motion_2dlmk",
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
    
    print("creating training data loader...")    
    data_dict, mean, std = load_data(
        args.dataset,
        args.dataset_path,
        "train",
        input_motion_length=None
    )
    dataset = TrainDataset(
        args.dataset,
        mean,
        std,
        data_dict,
        None,
        args.train_dataset_repeat_times,
        args.no_normalization,
    )
    trainloader = get_dataloader(
        dataset, "train", batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # val data loader
    print("creating val data loader...")
    val_dict, _, _ = load_data(
        args.dataset,
        args.dataset_path,
        "val",
        input_motion_length=None
    )
    
    val_dataset = TrainDataset(
        args.dataset,
        mean,
        std,
        val_dict,
        None,
        1,
        args.no_normalization,
    )
    
    val_loader = get_dataloader(
        val_dataset, "val", batch_size=args.batch_size, num_workers=args.num_workers
    )

    train_diffusion_model(args, trainloader, val_loader, mean, std)

if __name__ == "__main__":
    main()
