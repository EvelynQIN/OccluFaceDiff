import json
import os
import random
import pickle

import numpy as np

import torch
import torch.nn.functional as F

from data_loaders.dataloader_MEAD_classifer import get_dataloader, TrainMeadDataset

# from runner.train_mlp import train_step, val_step
from runner.training_loop_flint import TrainLoop
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
from utils.scheduler import WarmupCosineSchedule
from utils import dist_util

from utils.model_util import create_model_and_diffusion
from utils.parser_util import train_pureTrans_args
from utils.config import Config
import wandb
from tqdm import tqdm
from configs.classifier_config import get_cfg_defaults

from model.video_emotion_classifier import VideoEmotionClassifier

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

def emotion_id_2_emotion_name():
    
    emotions = ['neutral', 'happy', 'sad', 'surprised', 'fear', 'disgusted', 'angry', 'contempt']
    levels = ['level_1', 'level_2', 'level_3']
    class_id = 0
    emotion_id_2_emotion_name_mapping = {}
    for emotion in emotions:
        for level in levels:
            if emotion == 'neutral' and level != 'level_1':
                continue
            emotion_id_2_emotion_name_mapping[class_id] = emotion
            class_id += 1
    return emotion_id_2_emotion_name_mapping

def train_model(args, train_loader, valid_loader):
    print("creating emotion classifier model...")
    num_gpus = torch.cuda.device_count()
    args.num_workers = args.num_workers * num_gpus

    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    resume_epoch = 0
    resume_step = 0
    num_steps = args.num_epoch * steps_per_epoch
    emo_id_2_emo_name = emotion_id_2_emotion_name()

    model = VideoEmotionClassifier(
        args.input_dim,
        args.num_classes,
        args.latent_dim,
        args.heads,
        args.layers,
        args.ff_size,
        args.max_pool,
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

    # initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.cosine_scheduler:
        scheduler =  WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps)
    else:
        scheduler = MultiStepLR(optimizer, milestones=[15,30,45], gamma=0.3)
    
    batch_size = args.batch_size
    lr = args.lr

    for epoch in tqdm(range(resume_epoch, resume_epoch + args.num_epoch)):
        model.train()
        optimizer.zero_grad()
        local_step = 0
        for x, label in tqdm(train_loader):

            x = x.to(device)
            label = label.to(device)

            grad_update = True if local_step % args.gradient_accumulation_steps == 0 else False
            model_output = model(x)

            loss = F.nll_loss(model_output, label)
            
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if grad_update:
                optimizer.step()
                lr = optimizer.param_groups[-1]['lr']
                optimizer.zero_grad()
                if args.wandb_log:
                    wandb_log_dict = {}
                    wandb_log_dict['lr'] = lr
                    wandb_log_dict['epoch'] = epoch
                    wandb_log_dict['loss'] = loss.detach().item()
                    wandb.log(wandb_log_dict)
            local_step += 1
        scheduler.step()

        if epoch % args.log_interval == 0:
            model.eval()
            print("start eval ...")
            total_loss = 0.0
            eval_steps = 0.0
            acc = 0.0
            acc_emo = 0.0
            with torch.no_grad():
                for x, label in tqdm(valid_loader):
                    eval_steps += 1
                    x = x.to(device)
                    label = label.to(device)
                    model_output = model(x)
                    
                    loss = F.nll_loss(model_output, label)
                    total_loss += loss.detach().item()

                    pred_label = torch.argmax(model_output, dim=1).detach().cpu()
                    label = label.cpu()
                    acc += torch.sum(pred_label == label).detach().item()

                    pred_emotion = emo_id_2_emo_name[pred_label.item()]
                    gt_emotion = emo_id_2_emo_name[label.item()]
                    acc_emo += 1 if pred_emotion == gt_emotion else 0

                total_loss /= eval_steps
                acc /= eval_steps
                acc_emo /= eval_steps
                if args.wandb_log:   
                    wandb_log_dict = {}
                    wandb_log_dict['val/epoch'] = epoch
                    wandb_log_dict["val/loss"] = total_loss
                    wandb_log_dict["val/acc"] = acc
                    wandb_log_dict["val/acc_emo"] = acc_emo
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
    args = get_cfg_defaults()
    
    set_deterministic(args.seed)

    args.save_dir = os.path.join(args.save_dir, args.arch)
    
    # init wandb log
    if args.wandb_log:
        wandb.init(
            project="MEAD_emotion_classifier",
            name=args.arch,
            config=args,
            settings=wandb.Settings(start_method="fork"),
            dir="./"
        )
    
    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("creating training data loader...")    
    
    with open('dataset/mead_25fps/processed/video_list_woimg.pkl', 'rb') as f:
        train_video_list = pickle.load(f)
    
    with open('dataset/mead_25fps/processed/video_list_test.pkl', 'rb') as f:
        test_video_list = pickle.load(f)
    
    train_video_list_wotest = [vid for vid in train_video_list if vid not in test_video_list]

    print(f"number of train sequences: {len(train_video_list_wotest)}")
    train_dataset = TrainMeadDataset(
        args.dataset,
        args.dataset_path,
        train_video_list_wotest,
        args.n_shape,
        args.n_exp
    )

    train_loader = get_dataloader(
        train_dataset, "train", batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # val data loader
    print("creating val data loader...")
    print(f"number of test sequences: {len(test_video_list)}")
    val_dataset = TrainMeadDataset(
        args.dataset,
        args.dataset_path,
        test_video_list,
        args.n_shape,
        args.n_exp,
    )
    
    val_loader = get_dataloader(
        val_dataset, "val", batch_size=args.batch_size, num_workers=args.num_workers
    )

    train_model(args, train_loader, val_loader)

if __name__ == "__main__":
    main()