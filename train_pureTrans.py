import json
import os
import random

import numpy as np

import torch

from data_loaders.dataloader_MEAD_flint import get_dataloader, load_data, TrainMeadDataset

# from runner.train_mlp import train_step, val_step
from runner.training_loop_flint import TrainLoop
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils.scheduler import WarmupCosineSchedule
from utils import dist_util

from utils.model_util import create_model_and_diffusion
from utils.parser_util import train_pureTrans_args
from utils.config import Config
from utils.occlusion import MediaPipeFaceOccluder
import wandb
from tqdm import tqdm
from configs.config import get_cfg_defaults

from model.non_diffusion_model import FaceTransformerFLINT
from model.motion_prior import L2lVqVae
from munch import Munch, munchify
from runner.train_pure_transformer import TransformerLoss

def parse_resume_epoch_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def load_flint(model_cfg, device):
    ckpt_path = model_cfg.flint_ckpt_path
    f = open(model_cfg.flint_config_path)
    cfg = Munch.fromYAML(f)
    flint = L2lVqVae(cfg)
    flint.load_model_from_checkpoint(ckpt_path)
    flint.to(device)
    print(f"[FLINT] Loaded and Frozen.")
    flint.requires_grad_(False)
    flint.eval()
    return flint

def train_pureTrans_model(args, model_cfg, train_loader, valid_loader):
    print("creating Pure Transformer model...")
    num_gpus = torch.cuda.device_count()
    args.num_workers = args.num_workers * num_gpus

    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    resume_epoch = 0
    resume_step = 0
    num_steps = args.num_epochs * steps_per_epoch
    train_stage = model_cfg.train_stage
    
    model = FaceTransformerFLINT(
        args.arch,
        args.latent_dim, 
        args.ff_size, 
        args.num_enc_layers, 
        args.num_heads, 
        args.dropout
    )

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
    
    # resume training from ckpt
    if args.resume_checkpoint:
        resume_epoch = parse_resume_epoch_from_filename(args.resume_checkpoint) + 1
        print(f"loading model from checkpoint: {args.resume_checkpoint}...")
        model.load_state_dict(
            dist_util.load_state_dict(
                args.resume_checkpoint,
                map_location=dist_util.dev(),
            ),
            strict=True
        )

    device = torch.device("cpu")
    if torch.cuda.is_available() and dist_util.dev() != "cpu":
        device = torch.device(dist_util.dev())
    
    # load flint
    flint = load_flint(model_cfg, device)

    # apply random occlusion 
    occluder = MediaPipeFaceOccluder()

    # initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.cosine_scheduler:
        scheduler =  WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_steps)
    
    nb_iter = 0
    batch_size = args.batch_size
    lr = args.lr
    loss_keys = None 

    train_loss = TransformerLoss(model_cfg)

    for epoch in tqdm(range(resume_epoch, resume_epoch + args.num_epoch)):
        model.train()
        print(f"Starting training epoch {epoch}")

        # set occlusion
        # focus on visual signals without mouth & all occ type
        if epoch % args.freeze_audio_encoder_interval < (args.freeze_audio_encoder_interval // 2):
            model.freeze_wav2vec()
            occluder.occlusion_regions_prob = {
                'all': 0.3,
                'eye': 0.3,
                'left': 0.3,
                'right': 0.3,
                'left_eye': 0.3,
                'right_eye': 0.3,
                'mouth': 0.2,
                'random': 0.4,
                'contour': 0.4
            }
            occluder.mask_all_prob = 0.15
            occluder.mask_frame_prob = 0.15
            # focus on combining audio signals with only mouth & all occ type
        else:
            model.unfreeze_wav2vec()
            occluder.occlusion_regions_prob = {
                'all': 0.3,
                'eye': 0.,
                'left': 0.15,
                'right': 0.15,
                'left_eye': 0.,
                'right_eye': 0.,
                'mouth': 0.8,
                'random': 0.,
                'contour': 0.3
            }
            occluder.mask_all_prob = 0.3
            occluder.mask_frame_prob = 0.2

        local_step = 0
        for batch in tqdm(train_loader):
            # generate random occlusion mask
            batch['lmk_mask'] = occluder.get_lmk_occlusion_mask(batch['lmk_2d'][0]).unsqueeze(0).repeat(batch_size, 1, 1)   # (bs, t, v)

            for k in batch:
                batch[k] = batch[k].to(device)
            motion_target = torch.cat([batch['exp'], batch['jaw']], dim=-1)

            target = flint.motion_encoder(motion_target)  # (bs, n//8, 128)

            grad_update = True if local_step % args.gradient_accumulation_steps == 0 else False
            model_output = model(batch)

            loss, loss_dict = train_loss.compute_loss(target, model_output, batch)
            
            if loss_keys is None:
                loss_keys = loss_dict.keys()
            loss = loss_dict['loss'] / args.gradient_accumulation_steps
            loss.backward()

            if grad_update:
                optimizer.step()
                scheduler.step()
                lr = optimizer.param_groups[-1]['lr']
                optimizer.zero_grad()
                if args.wandb_log:
                    loss_dict["lr"] = lr
                    loss_dict["epoch"] = epoch
                    wandb_log_dict = {}
                    for k, v in loss_dict.items():
                        wandb_log_dict["train/"+k] = v
                    wandb.log(wandb_log_dict)
            local_step += 1
        
        if epoch % args.log_interval == 0:
            model.eval()
            print("start eval ...")
            val_loss = dict()
            for key in loss_keys:
                val_loss[key] = 0.0
            eval_steps = 0.0
            with torch.no_grad():
                for batch in tqdm(valid_loader):
                    eval_steps += 1
                    batch['lmk_mask'] = occluder.get_lmk_occlusion_mask(batch['lmk_2d'][0]).unsqueeze(0).repeat(self.batch_size, 1, 1)   # (bs, t, v)
                    for k in batch:
                        batch[k] = batch[k].to(device)
                    motion_target = torch.cat([batch['exp'], batch['jaw']], dim=-1)
                    target = flint.motion_encoder(motion_target)  # (bs, n//8, 128)
                    model_output = model(batch)
                    _, loss_dict = train_loss.compute_loss(target, model_output, batch)
                    for k in val_loss:
                        val_loss[k] +=  loss_dict[k]
                for k in val_loss:
                    val_loss[k] /= eval_steps
                if args.wandb_log:   
                    loss_dict["epoch"] = epoch
                    wandb_log_dict = {}
                    for k, v in val_loss.items():
                        wandb_log_dict["val/"+k] = v
                    wandb.log(wandb_log_dict)
                        
        if epoch % args.save_interval == 0:
            with open(
                os.path.join(args.save_dir, f"model_{(epoch)}.pt"),
                "wb",
            ) as f:
                torch.save(model.state_dict(), f)

def set_deterministic(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    args = train_pureTrans_args()
    kwargs = dict(args._get_kwargs())
    cfg_path = args.config_path
    args = Config(default_cfg_path=cfg_path, **kwargs)

    default_cfg = get_cfg_defaults()
    
    set_deterministic(args.seed)

    model_type = args.arch.split('_')[0]
    args.save_dir = os.path.join(args.save_dir, args.arch[len(model_type)+1:])
    
    # init wandb log
    if args.wandb_log:
        wandb.init(
            project="face_animation_from_MEAD_non_diffusion",
            name=args.arch,
            config=args,
            settings=wandb.Settings(start_method="fork"),
            dir="./"
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
    train_processed_path = load_data(
        args.dataset,
        args.dataset_path,
        "train",
        args.input_motion_length
    )

    print(f"number of train sequences: {len(train_processed_path)}")
    train_dataset = TrainMeadDataset(
        args.dataset,
        args.dataset_path,
        train_processed_path,
        args.input_motion_length,
        args.train_dataset_repeat_times,
        args.no_normalization,
        args.fps,
        args.n_shape,
        args.n_exp,
        args.load_tex,
        args.use_iris
    )

    train_loader = get_dataloader(
        train_dataset, "train", batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # val data loader
    print("creating val data loader...")
    val_processed_path = load_data(
        args.dataset,
        args.dataset_path,
        "test",
        args.input_motion_length
    )
    print(f"number of test sequences: {len(val_processed_path)}")
    val_dataset = TrainMeadDataset(
        args.dataset,
        args.dataset_path,
        val_processed_path,
        args.input_motion_length,
        10,
        args.no_normalization,
        args.fps,
        args.n_shape,
        args.n_exp,
        args.load_tex,
        args.use_iris
    )
    
    val_loader = get_dataloader(
        val_dataset, "val", batch_size=args.batch_size, num_workers=args.num_workers
    )

    train_pureTrans_model(args, default_cfg.model, train_loader, val_loader)

if __name__ == "__main__":
    main()