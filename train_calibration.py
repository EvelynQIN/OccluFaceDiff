import json
import os
import random
import numpy as np
import torch
import wandb
from tqdm import tqdm
import pickle
from collections import defaultdict

from data_loaders.dataloader_calibration import get_dataloader, load_data, LmkDataset
from model.calibration_layer import Cam_Calibration
from model.FLAME import FLAME
from utils import utils_transform
from utils.famos_camera import batch_perspective_project_wo_distortion, batch_orth_proj
from utils import dist_util
import yaml 
import cv2
FOCAL_LEN = 1000.0
PRINCIPAL_POINT_OFFSET = 112.0
IMAGE_SIZE = 224 
MEAN_TRANS = torch.FloatTensor([0.004, 0.222, 1.200])   # guessing from training 
FIXED_INTRIN = torch.FloatTensor([
        [FOCAL_LEN, 0., PRINCIPAL_POINT_OFFSET],
        [0., FOCAL_LEN, PRINCIPAL_POINT_OFFSET],
        [0., 0., 1.]
    ])


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def verts_loss_weighted(gt, pred, weights=None):
    bs = pred.shape[0]
    
    lmk_diff_normed = torch.abs(
        (gt.reshape(bs,-1,2) - pred.reshape(bs,-1,2))).sum(-1)
    
    if weights is None:
        reprojection_loss_normed_weighted = torch.mean(lmk_diff_normed)
    else:
        k = torch.sum(weights) * 2.0
        reprojection_loss_normed_weighted = torch.mean(
            torch.matmul(lmk_diff_normed, weights.unsqueeze(-1)) * 1.0 / k)
    
    return reprojection_loss_normed_weighted

def lmk_reproj_loss_train(
        model_output, 
        lmk_2d, 
        verts_2d, 
        flame_params, 
        flame, 
        verts_loss_weights, 
        vis=False, 
        vis_id=None,
        verbose=False
    ):
    # fixed focal length and principal point offset
    device = model_output.device
    bs = model_output.shape[0]
    intrin = FIXED_INTRIN.unsqueeze(0).expand(bs, 3, 3).to(device) # (bs, 3, 3)
    mean_trans = MEAN_TRANS.unsqueeze(0).expand(bs, -1).to(device)
    T = (model_output + mean_trans).unsqueeze(-1) # (bs, 3, 1)
    R = torch.eye(3).unsqueeze(0).expand(bs,-1, -1).to(device) # (bs, 3, 3)
    extrin = torch.cat([R, T], dim=-1)  # (bs, 3, 4)

    shape = flame_params[:, :300]
    exp = flame_params[:, 300:400]
    rot_6d = flame_params[:, 400:]
    rot_aa = utils_transform.sixd2aa(rot_6d.reshape(-1, 6)).reshape(bs, -1)
    verts_3d, lmk_3d = flame(shape, exp, rot_aa)

    lmk2d_reproject = batch_perspective_project_wo_distortion(lmk_3d, intrin, extrin)
    lmk2d_reproject_normed = lmk2d_reproject / IMAGE_SIZE * 2.0 - 1
    
    verts_2d_reproject = batch_perspective_project_wo_distortion(verts_3d, intrin, extrin)
    verts_2d_reproject_normed = verts_2d_reproject / IMAGE_SIZE * 2.0 - 1
    
    shape = flame_params[:, :300]
    exp = flame_params[:, 300:400]
    rot_6d = flame_params[:, 400:]
    rot_aa = utils_transform.sixd2aa(rot_6d.reshape(-1, 6)).reshape(bs, -1)

    weights_lmk = verts_loss_weights['lmk_2d'].to(device)
    lmk2d_loss_weighted = verts_loss_weighted(lmk_2d, lmk2d_reproject_normed, weights_lmk)

    weights_flame = verts_loss_weights['flame_verts'].to(device)
    verts2d_loss_weighted = verts_loss_weighted(verts_2d, verts_2d_reproject_normed, weights_flame)

    output_mean = None
    if verbose:
        output_mean = torch.mean(T, dim=0)
    
    reg_trans = 10.0 * torch.mean(model_output ** 2) * 1.0 / 2
    
    loss_dict = {
        "lmk2d_loss": lmk2d_loss_weighted,
        "verts_2d_loss": verts2d_loss_weighted,
        "loss": lmk2d_loss_weighted + reg_trans + verts2d_loss_weighted
    }

    log_imgs = None
    
    if vis: 
        log_imgs = []
        lmk_2d_denormed = (lmk_2d + 1) * IMAGE_SIZE / 2
        for i in vis_id:
            image_arr = plot_kpts(lmk_2d_denormed[i].reshape(-1, 2), lmk2d_reproject[i].reshape(-1, 2))
            image = wandb.Image(image_arr)
            log_imgs.append(image)

    return loss_dict, output_mean, log_imgs

def lmk_reproj_loss_val(
        model_output, 
        lmk_2d, 
        verts_2d, 
        flame_params, 
        flame, 
        verts_loss_weights, 
        vis=False, 
        vis_id=None,
        verbose=False
    ):
    # fixed focal length and principal point offset
    device = model_output.device
    bs = model_output.shape[0]
    intrin = FIXED_INTRIN.unsqueeze(0).expand(bs, 3, 3).to(device) # (bs, 3, 3)
    mean_trans = MEAN_TRANS.unsqueeze(0).expand(bs, -1).to(device)
    T = (model_output + mean_trans).unsqueeze(-1) # (bs, 3, 1)
    R = torch.eye(3).unsqueeze(0).expand(bs,-1, -1).to(device) # (bs, 3, 3)
    extrin = torch.cat([R, T], dim=-1)  # (bs, 3, 4)

    shape = flame_params[:, :300]
    exp = flame_params[:, 300:400]
    rot_6d = flame_params[:, 400:]
    rot_aa = utils_transform.sixd2aa(rot_6d.reshape(-1, 6)).reshape(bs, -1)
    verts_3d, lmk_3d = flame(shape, exp, rot_aa)

    lmk2d_reproject = batch_perspective_project_wo_distortion(lmk_3d, intrin, extrin)
    lmk2d_reproject_normed = lmk2d_reproject / IMAGE_SIZE * 2.0 - 1
    
    verts_2d_reproject = batch_perspective_project_wo_distortion(verts_3d, intrin, extrin)
    verts_2d_reproject_normed = verts_2d_reproject / IMAGE_SIZE * 2.0 - 1
    
    shape = flame_params[:, :300]
    exp = flame_params[:, 300:400]
    rot_6d = flame_params[:, 400:]
    rot_aa = utils_transform.sixd2aa(rot_6d.reshape(-1, 6)).reshape(bs, -1)

    lmk2d_loss_weighted = verts_loss_weighted(lmk_2d, lmk2d_reproject_normed)

    verts2d_loss_weighted = verts_loss_weighted(verts_2d, verts_2d_reproject_normed)

    output_mean = None
    if verbose:
        output_mean = torch.mean(T, dim=0)
    
    loss_dict = {
        "lmk2d_loss": lmk2d_loss_weighted,
        "verts_2d_loss": verts2d_loss_weighted,
    }

    log_imgs = None
    
    if vis: 
        log_imgs = []
        lmk_2d_denormed = (lmk_2d + 1) * IMAGE_SIZE / 2
        for i in vis_id:
            image_arr = plot_kpts(lmk_2d_denormed[i].reshape(-1, 2), lmk2d_reproject[i].reshape(-1, 2))
            image = wandb.Image(image_arr)
            log_imgs.append(image)

    return loss_dict, output_mean, log_imgs

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

def train_calibration_model(args, train_loader, valid_loader):
    print("creating MLP model...")
    num_gpus = torch.cuda.device_count()
    args.num_workers = args.num_workers * num_gpus

    flame = FLAME(flame_model_path=args.flame_model_path, flame_lmk_embedding_path=args.flame_lmk_embedding_path)

    flame_vmask_path = "flame_2020/FLAME_masks.pkl"
    with open(flame_vmask_path, 'rb') as f:
        flame_v_mask = pickle.load(f, encoding="latin1")
    
    # define landamark loss weights
    lmk_weights = torch.ones((68,))
    # face contour 
    lmk_weights[5:7] = 2
    lmk_weights[10:12] = 2
    # eye points
    lmk_weights[36:48] = 2
    lmk_weights[36] = 4
    lmk_weights[39] = 4
    lmk_weights[42] = 4
    lmk_weights[45] = 4
    # nose points
    lmk_weights[30] = 1.5
    lmk_weights[31] = 1.5
    lmk_weights[35] = 1.5
    # inner mouth
    lmk_weights[60:68] = 1.5
    lmk_weights[48:60] = 1.5
    lmk_weights[48] = 4
    lmk_weights[54] = 4

    # define flame_verts loss weights
    verts_weights = torch.ones((5023,)) * 0.5
    face_ids = flame_v_mask['face']
    verts_weights[face_ids] = 3.0

    verts_loss_weights = {
        "lmk_2d": lmk_weights,
        "flame_verts": verts_weights
    }

        
    model = Cam_Calibration(
        lmk2d_dim=args.lmk2d_dim, # input feature dim 68 x 2
        n_shape=args.n_shape,
        output_feature_dim=args.output_feature_dim, # number of cam params (one set per frame)
        latent_dim=args.latent_dim,
        ckpt_path=None,
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

    vis_id = [0, 50, 70, 120]    # frame id for visualization (in validation)

    # train + val for each epoch
    for epoch in tqdm(range(args.num_epoch)):
        model.train()
        x, y, z = [], [], []
        for occluded_lmk2d, shape, lmk_2d, verts_2d, target in tqdm(train_loader):
            device = dist_util.dev()
            occluded_lmk2d = occluded_lmk2d.to(device)
            shape = shape.to(device)
            verts_2d = verts_2d.to(device)
            lmk_2d = lmk_2d.to(device)
            flame_params = target.to(device)
            cam_pred = model(occluded_lmk2d, shape)

            loss_dict, trans_mean, _ = lmk_reproj_loss_train(cam_pred, lmk_2d, verts_2d, flame_params, flame, verts_loss_weights, verbose=True)
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
            tx, ty, tz = trans_mean.detach().cpu().numpy()
            x.append(tx)
            y.append(ty)
            z.append(tz)
            log_dict = {
                "train/epoch": epoch
            }
            for k in loss_dict:
                log_dict[f'train/{k}'] = loss_dict[k].item()
            
            wandb.log(loss_dict)
            nb_iter += 1
        
        print(f"Train for epoch {epoch}:")
        print(f'trans_x: mean = {np.mean(x)} || std = {np.std(x)}')
        print(f'trans_y: mean = {np.mean(y)} || std = {np.std(y)}')
        print(f'trans_z: mean = {np.mean(z)} || std = {np.std(z)}')
        
        model.eval()
        total_loss = defaultdict(float)
        eval_steps = 0
        x, y, z = [], [], []
        log_imgs = None
        vis_step = 1
        with torch.no_grad():
            for occluded_lmk2d, shape, lmk_2d, verts_2d, target in tqdm(valid_loader):
                eval_steps += 1
                device = dist_util.dev()
                occluded_lmk2d = occluded_lmk2d.to(device)
                shape = shape.to(device)
                verts_2d = verts_2d.to(device)
                lmk_2d = lmk_2d.to(device)
                flame_params = target.to(device)
                cam_pred = model(occluded_lmk2d, shape)

                vis = True if ((epoch % 5 == 0) and (eval_steps == vis_step)) else False
                loss_dict, trans_mean, log_imgs_step = lmk_reproj_loss_val(
                    cam_pred, lmk_2d, verts_2d, flame_params, flame, verts_loss_weights, vis, vis_id, verbose=True)
                if vis:
                    log_imgs = log_imgs_step
                for k in loss_dict:
                    total_loss[k] += float(loss_dict[k].item())
                tx, ty, tz = trans_mean.detach().cpu().numpy()
                x.append(tx)
                y.append(ty)
                z.append(tz)
            
            log_dict = {"val/epoch": epoch}
            for k in total_loss:
                log_dict[f'val/{k}'] = total_loss[k] / eval_steps

            print(f"Val for epoch {epoch}:")
            print(f'trans_x: mean = {np.mean(x)} || std = {np.std(x)}')
            print(f'trans_y: mean = {np.mean(y)} || std = {np.std(y)}')
            print(f'trans_z: mean = {np.mean(z)} || std = {np.std(z)}')
            
            if epoch % 5 == 0:
                log_dict['val/imgs'] = log_imgs
                
            wandb.log(log_dict)
    
        if epoch % args.save_interval == 0 or epoch == args.num_epoch-1:
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
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    
    # init wandb log
    wandb.init(
        project="pred_cam_with_shape",
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
    train_dict, norm_dict = load_data(
        args.dataset,
        args.dataset_path,
        "train",
    )
    dataset = LmkDataset(
        train_dict,
        norm_dict,
        occlusion_mask_prob=0.5
    )
    train_loader = get_dataloader(
        dataset, "train", batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # val data loader
    print("creating val data loader...")
    val_dict, _ = load_data(
        args.dataset,
        args.dataset_path,
        "test"
    )
    
    val_dataset = LmkDataset(
        val_dict,
        norm_dict,
        occlusion_mask_prob=0.5
    )
    
    val_loader = get_dataloader(
        val_dataset, "val", batch_size=args.batch_size, num_workers=args.num_workers
    )

    train_calibration_model(args, train_loader, val_loader)

if __name__ == "__main__":
    main()