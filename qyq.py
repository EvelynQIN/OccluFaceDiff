import torch
from munch import Munch, munchify
from model.motion_prior import L2lVqVae
from configs.config import get_cfg_defaults
from model.FLAME import FLAME_mediapipe
import h5py

model_cfg = get_cfg_defaults().model
flame = FLAME_mediapipe(model_cfg)

ckpt_path = 'pretrained/MotionPrior/models/FLINT/checkpoints/model-epoch=0120-val/loss_total=0.131580308080.ckpt'
# ckpt = torch.load(ckpt_path)

f = open('pretrained/MotionPrior/models/FLINT/cfg.yaml')
cfg = Munch.fromYAML(f)
flint = L2lVqVae(cfg)
flint.load_model_from_checkpoint(ckpt_path)
flint.freeze_model()

test_motion_rec_path = 'dataset/mead_25fps/processed/reconstructions/EMICA-MEAD_flame2020/M011/front/angry/level_1/001/shape_pose_cam.hdf5'
data_dict = {}
with h5py.File(test_motion_rec_path, 'r') as f:
    for k in f:
        data_dict[k] = torch.from_numpy(f[k][:, 32:64, :])
        print(f"{k}: {data_dict[k].shape}")
data_dict['exp'] = data_dict['exp'][...,:50]

inputs = torch.cat([data_dict['exp'], data_dict['jaw']], dim=-1)
print("inputs: ", inputs.shape)

z = flint.motion_encoder(inputs)
print("z: ", z.shape)

rec = flint.motion_decoder(z)
print("rec: ", rec.shape)

assert rec.shape == inputs.shape

rec_loss = torch.mean((rec - inputs).abs())
print(f"rec loss is {rec_loss}")


verts_gt, _ = flame(
    data_dict['shape'][0],
    data_dict['exp'][0],
    torch.cat([data_dict['global_pose'][0], data_dict['jaw'][0]], dim=-1)
)

verts_rec, _ = flame(
    data_dict['shape'][0],
    rec[0,:,:50],
    torch.cat([data_dict['global_pose'][0], rec[0,:,50:],], dim=-1)
)

verts_loss = torch.mean(
    torch.norm(verts_gt - verts_rec, p=2, dim=-1)
) * 1000.0
print(f"verts reconstruction loss is {verts_loss}")

# B, T, C = 2, 64, 53
# inputs = torch.rand((B, T, C))

# z = flint.motion_encoder(inputs)    # 2, 4, 128
# print(z.shape)

# rec = flint.motion_decoder(z)   # 2, 32, 53
# print(rec.shape)