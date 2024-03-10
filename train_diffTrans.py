import json
import os
import random

import numpy as np

import torch

from data_loaders.dataloader import get_dataloader, load_data, TrainDataset, TestDataset
from model.FLAME import FLAME
from model.networks import PureMLP
# from runner.train_mlp import train_step, val_step
from runner.training_loop import TrainLoop

from utils import dist_util

from utils.model_util import create_model_and_diffusion
from utils.parser_util import train_trans_args
from utils.config import Config
import wandb
from tqdm import tqdm
from configs.config import get_cfg_defaults

def train_diffusion_model(args, pretrained_args, train_dataloader, val_dataloader, norm_dict):
    print("creating model and diffusion...")

    num_gpus = torch.cuda.device_count()
    args.num_workers = args.num_workers * num_gpus

    cam_model, denoise_model, diffusion = create_model_and_diffusion(args, pretrained_args) # the denoising MLP & spaced diffusion

    if num_gpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        dist_util.setup_dist()
        cam_model = torch.nn.DataParallel(cam_model).cuda()
        denoise_model = torch.nn.DataParallel(denoise_model).cuda()
        print(
            "Total trainable params: %.2fM"
            % (sum(p.numel() for p in denoise_model.module.parameters() if p.requires_grad) / 1000000.0)
        )
    else:
        dist_util.setup_dist(args.device)
        cam_model.to(dist_util.dev())
        denoise_model.to(dist_util.dev())
        print(
            "Total trainable params: %.2fM"
            % (sum(p.numel() for p in denoise_model.parameters() if p.requires_grad) / 1000000.0)
        )

    print("Training...")
    TrainLoop(args, cam_model, denoise_model, diffusion, train_dataloader, val_dataloader, norm_dict).run_loop()
    print("Done.")

def set_deterministic(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    args = train_trans_args() 
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
            project="face_motion_animation_from_image",
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
    train_motion_paths, norm_dict = load_data(
        args,
        args.dataset,
        args.dataset_path,
        "train",
    )
    train_dataset = TrainDataset(
        args.dataset,
        norm_dict,
        train_motion_paths,
        None,
        args.train_dataset_repeat_times,
        args.no_normalization,
    )
    train_loader = get_dataloader(
        train_dataset, "train", batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # val data loader
    print("creating val data loader...")
    val_motion_paths, _ = load_data(
        args,
        args.dataset,
        args.dataset_path,
        "val",
    )
    
    val_dataset = TrainDataset(
        args.dataset,
        norm_dict,
        val_motion_paths,
        None,
        1,
        args.no_normalization,
    )
    
    val_loader = get_dataloader(
        val_dataset, "val", batch_size=args.batch_size, num_workers=args.num_workers
    )

    train_diffusion_model(args, pretrained_args, train_loader, val_loader, norm_dict)

if __name__ == "__main__":
    main()
