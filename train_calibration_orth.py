import json
import os
import random
import numpy as np
import torch
import wandb
from tqdm import tqdm

from data_loaders.dataloader_calibration import get_dataloader, load_data, LmkDataset
from model.calibration_layer import Cam_Calibration
from utils import utils_transform
from utils.famos_camera import batch_orth_proj
from utils import dist_util
import yaml 
import cv2
IMAGE_SIZE = 224 

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def lmk_reproj_loss(model_output, lmk3d, lmk2d_gt, verbose=False):
    # fixed focal length and principal point offset
    device = model_output.device
    bs = model_output.shape[0]
    lmk2d_reproject = batch_orth_proj(lmk3d, model_output) # (bs, 68, 2)
    lmk2d_reproject[:, :, 1]  = -lmk2d_reproject[:, :, 1]
    
    # define loss weights
    weights = torch.ones((68,)).cuda()

    # face contour 
    weights[5:7] = 2
    weights[10:12] = 2
    # eye points
    weights[36:48] = 1.5
    weights[36] = 3
    weights[39] = 3
    weights[42] = 3
    weights[45] = 3
    # nose points
    weights[30] = 3
    weights[31] = 3
    weights[35] = 3
    # inner mouth
    weights[60:68] = 1.5
    weights[48:60] = 1.5
    weights[48] = 3
    weights[54] = 3

    k = torch.sum(weights) * 2.0

    # denormalize 
    lmk2d_gt_denormed = (lmk2d_gt + 1) * IMAGE_SIZE / 2.0

    # normalize 
    lmk2d_reproject_normed = lmk2d_reproject / IMAGE_SIZE * 2.0 - 1

    reprojection_loss = torch.mean(
        torch.abs(
            (lmk2d_gt_denormed.reshape(-1, 2) - lmk2d_reproject.reshape(-1, 2))).sum(1)
    )

    lmk_diff_normed = torch.abs(
        (lmk2d_gt.reshape(bs,-1,2) - lmk2d_reproject_normed.reshape(bs,-1,2))).sum(-1)
    
    reprojection_loss_normed_weighted = torch.mean(
        torch.matmul(lmk_diff_normed, weights.unsqueeze(-1)) * 1.0 / k)
    
    # eye width loss
    lmk2d_gt = lmk2d_gt.reshape(bs, -1, 2)
    left_eye_width_loss = torch.mean(
        torch.abs((lmk2d_reproject_normed[:,36]-lmk2d_reproject_normed[:,39]) - 
                  (lmk2d_gt[:,36]-lmk2d_gt[:,39])).sum(-1)) 

    right_eye_width_loss = torch.mean(
        torch.abs((lmk2d_reproject_normed[:,42]-lmk2d_reproject_normed[:,45]) - 
                  (lmk2d_gt[:,42]-lmk2d_gt[:,45])).sum(-1))
    
    nose_len_loss = torch.mean(
        torch.abs((lmk2d_reproject_normed[:,27]-lmk2d_reproject_normed[:,33]) - 
                  (lmk2d_gt[:,27]-lmk2d_gt[:,33])).sum(-1))
    mouth_closure_loss = torch.mean(
        torch.abs((lmk2d_reproject_normed[:,51]-lmk2d_reproject_normed[:,57]) - 
                  (lmk2d_gt[:,51]-lmk2d_gt[:,57])).sum(-1))
    
    output_mean = None
    if verbose:
        output_mean = torch.mean(model_output, dim=0)
    
    # reg_trans = 100.0 * torch.mean(model_output ** 2) * 1.0 / 2
    
    loss_dict = {
        "loss": reprojection_loss,  # denormed reprojection loss (avg)
        "loss_normed": reprojection_loss_normed_weighted + left_eye_width_loss + right_eye_width_loss + nose_len_loss + mouth_closure_loss
    }

    return loss_dict, output_mean

def plot_kpts(kpts_gt, kpts_pred):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    red = (255, 0, 0)
    green = (0, 255, 0)
    
    end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    for i in range(kpts_gt.shape[0]):
        st = kpts_gt[i, :2]
        image = cv2.circle(image,(int(st[0]), int(st[1])), 1, red, 2)
        if i in end_list:
            continue
        ed = kpts_gt[i + 1, :2]
        image = cv2.line(image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), red, 1)
    
    for i in range(kpts_pred.shape[0]):
        st = kpts_pred[i, :2]
        image = cv2.circle(image,(int(st[0]), int(st[1])), 1, green, 2)
        if i in end_list:
            continue
        ed = kpts_pred[i + 1, :2]
        image = cv2.line(image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), green, 1)

    return image

def vis_kpts(lmk2ds, lmk3ds, model):
    model_output = model(lmk2ds).detach().cpu()
    lmk2ds = lmk2ds.detach().cpu().numpy()
    lmk3ds = lmk3ds.detach().cpu()
    # fixed focal length and principal point offset
    bs = lmk3ds.shape[0]
    lmk2d_reproject = batch_orth_proj(lmk3ds, model_output).numpy()
    lmk2d_reproject[:, :, 1]  = -lmk2d_reproject[:, :, 1]

    lmk2d_gt_denormed = (lmk2ds + 1) * IMAGE_SIZE / 2.0
    imgs = []
    for i in range(bs):
        image_arr = plot_kpts(lmk2d_gt_denormed[i].reshape(-1, 2), lmk2d_reproject[i].reshape(-1, 2))
        image = wandb.Image(image_arr)
        imgs.append(image)
    return imgs
    

def train_calibration_model(args, train_loader, valid_loader):
    print("creating MLP model...")
    num_gpus = torch.cuda.device_count()
    args.num_workers = args.num_workers * num_gpus
    
    model = Cam_Calibration(
        args.input_feature_dim, # input feature dim 68 x 2
        args.output_feature_dim, # number of cam params (one set per frame)
        args.latent_dim,
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

    # initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    nb_iter = 0

    vis_id = [0, 10, 25, 50, 57]    # frame id for visualization (in validation)

    # train + val for each epoch
    for epoch in tqdm(range(args.num_epoch)):
        model.train()
        x, y, z = [], [], []
        for lmk_2d, lmk_3d in tqdm(train_loader):
            device = dist_util.dev()
            lmk_2d = lmk_2d.to(device)
            lmk_3d = lmk_3d.to(device)

            cam_pred = model(lmk_2d)

            loss_dict, trans_mean = lmk_reproj_loss(cam_pred, lmk_3d, lmk_2d, verbose=True)
            optimizer.zero_grad()
            loss_dict["loss_normed"].backward()
            optimizer.step()
            tx, ty, tz = trans_mean.detach().cpu().numpy()
            x.append(tx)
            y.append(ty)
            z.append(tz)

            loss_dict = {
                "train/epoch": epoch,
                "train/reprojection_loss": loss_dict["loss"].item(),
                "train/train_loss": loss_dict["loss_normed"].item()
            }
            wandb.log(loss_dict)
            nb_iter += 1
        
        print(f"Train for epoch {epoch}:")
        print(f'trans_x: mean = {np.mean(x)} || std = {np.std(x)}')
        print(f'trans_y: mean = {np.mean(y)} || std = {np.std(y)}')
        print(f'trans_z: mean = {np.mean(z)} || std = {np.std(z)}')
        
        model.eval()
        total_loss = 0.0
        eval_steps = 0.0
        x, y, z = [], [], []
        with torch.no_grad():
            for lmk_2d, lmk_3d in tqdm(valid_loader):
                device = dist_util.dev()
                lmk_2d = lmk_2d.to(device)
                lmk_3d = lmk_3d.to(device)

                cam_pred = model(lmk_2d)

                loss_dict, trans_mean = lmk_reproj_loss(cam_pred, lmk_3d, lmk_2d, verbose=True)
                eval_steps += 1
                total_loss += float(loss_dict["loss"].item())
                tx, ty, tz = trans_mean.detach().cpu().numpy()
                x.append(tx)
                y.append(ty)
                z.append(tz)
            
            loss_dict = {
                    "val/epoch": epoch,
                    "val/reprojection_loss": total_loss / eval_steps
            }
            print(f"Val for epoch {epoch}:")
            print(f'trans_x: mean = {np.mean(x)} || std = {np.std(x)}')
            print(f'trans_y: mean = {np.mean(y)} || std = {np.std(y)}')
            print(f'trans_z: mean = {np.mean(z)} || std = {np.std(z)}')
            
            if epoch % 5 == 0:
                lmks_2d_vis = lmk_2d[vis_id]
                lmk_3d_vis = lmk_3d[vis_id]
                imgs = vis_kpts(lmks_2d_vis, lmk_3d_vis, model)
                loss_dict['val/imgs'] = imgs
                
            wandb.log(loss_dict)
    
        if epoch % args.save_interval == 0:
            with open(
                os.path.join(args.save_dir, "model-epoch-" + str(epoch) + "-step-" + str(nb_iter) + ".pth"),
                "wb",
            ) as f:
                torch.save(model.state_dict(), f)


def main():
    
    cfg_path = 'configs/train_calibration.yaml'
    with open(cfg_path, 'r') as infile:
        cfg = yaml.safe_load(infile)
    args = Struct(**cfg)

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # init wandb log
    wandb.init(
        project="pred_cam",
        name=args.model_name,
        config=args,
        settings=wandb.Settings(start_method="fork"),
        dir="./wandb"
    )
    
    args.save_dir = os.path.join(args.save_dir, args.model_name)
    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(cfg, fw, indent=4, sort_keys=True) 
    
    print("creating training data loader...")    
    train_dict = load_data(
        args.dataset,
        args.dataset_path,
        "train",
    )
    dataset = LmkDataset(
        train_dict,
        scale=args.scale,
        trans_scale=args.trans_scale,
        image_size=args.image_size,
    )
    train_loader = get_dataloader(
        dataset, "train", batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # val data loader
    print("creating val data loader...")
    val_dict = load_data(
        args.dataset,
        args.dataset_path,
        "val"
    )
    
    val_dataset = LmkDataset(
        val_dict,
        scale=args.scale,
        trans_scale=args.trans_scale,
        image_size=args.image_size
    )
    
    val_loader = get_dataloader(
        val_dataset, "val", batch_size=args.batch_size, num_workers=args.num_workers
    )

    train_calibration_model(args, train_loader, val_loader)

if __name__ == "__main__":
    main()