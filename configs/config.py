import argparse
import os

from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Settings for pretrained models
# ---------------------------------------------------------------------------- #

cfg = CN()

# ---------------------------------------------------------------------------- #
# Defaults for DECA
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.expression_net_path = 'pretrained/ResNet50/checkpoints/deca-epoch=01-val_loss_total/dataloader_idx_0=1.27607644.ckpt'
cfg.model.topology_path = os.path.join('flame_2020' , 'head_template_mesh.obj')
# texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
cfg.model.dense_template_path = os.path.join('flame_2020', 'texture_data_256.npy')
cfg.model.flame_model_path = os.path.join('flame_2020', 'generic_model.pkl')
cfg.model.flame_mediapipe_lmk_embedding_path = os.path.join('flame_2020', 'mediapipe_landmark_embedding.npz')
cfg.model.flame_lmk_embedding_path = os.path.join('flame_2020', 'landmark_embedding.npy')
cfg.model.face_mask_path = os.path.join('flame_2020', 'uv_face_mask.png')
cfg.model.face_eye_mask_path = os.path.join('flame_2020', 'uv_face_eye_mask.png')
cfg.model.mean_tex_path = os.path.join('flame_2020', 'mean_texture.jpg')
cfg.model.tex_path = os.path.join('flame_2020', 'FLAME_albedo_from_BFM.npz')
cfg.model.tex_type = 'BFM' # BFM, FLAME, albedoMM
cfg.model.use_texture = True   # whether use predicted texture in renderer
cfg.model.uv_size = 256
cfg.model.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
cfg.model.n_shape = 300
cfg.model.n_tex = 50
cfg.model.n_exp = 100
cfg.model.n_cam = 3
cfg.model.n_pose = 6    # aa representation of neck + jaw
cfg.model.n_light = 27
cfg.model.jaw_type = 'aa' # default use axis angle, another option: euler. Note that: aa is not stable in the beginning
cfg.model.image_size = 224

# diffusion
cfg.model.flint_ckpt_path = 'pretrained/MotionPrior/models/FLINTv2/checkpoints/model-epoch=0758-val/loss_total=0.113977119327.ckpt'
cfg.model.flint_config_path = 'pretrained/MotionPrior/models/FLINTv2/cfg.yaml'

# ---------------------------------------------------------------------------- #
# Defaults for EMOCA
# ---------------------------------------------------------------------------- #
cfg.emoca = CN()
cfg.emoca.uv_size = 256
cfg.emoca.topology_path = os.path.join('flame_2020' , 'head_template_mesh.obj')
# texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
cfg.emoca.dense_template_path = os.path.join('flame_2020', 'texture_data_256.npy')
cfg.emoca.flame_model_path = os.path.join('flame_2020', 'generic_model.pkl')
cfg.emoca.flame_mediapipe_lmk_embedding_path = os.path.join('flame_2020', 'mediapipe_landmark_embedding.npz')
cfg.emoca.flame_lmk_embedding_path = os.path.join('flame_2020', 'landmark_embedding.npy')
cfg.emoca.face_mask_path = os.path.join('flame_2020', 'uv_face_mask.png')
cfg.emoca.face_eye_mask_path = os.path.join('flame_2020', 'uv_face_eye_mask.png')
cfg.emoca.mean_tex_path = os.path.join('flame_2020', 'mean_texture.jpg')
cfg.emoca.tex_path = os.path.join('flame_2020', 'FLAME_albedo_from_BFM.npz')
cfg.emoca.tex_type = 'BFM' # BFM, FLAME, albedoMM
cfg.emoca.use_texture = False   # whether use predicted texture in renderer
cfg.emoca.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
cfg.emoca.n_shape = 100
cfg.emoca.n_tex = 50
cfg.emoca.n_exp = 50
cfg.emoca.n_cam = 3
cfg.emoca.n_pose = 6    # aa representation of neck + jaw
cfg.emoca.n_light = 27
cfg.emoca.jaw_type = 'aa' # default use axis angle, another option: euler. Note that: aa is not stable in the beginning
cfg.emoca.image_size = 224
cfg.emoca.ckpt_path = 'pretrained/EMOCA/detail/checkpoints/deca-epoch=03-val_loss/dataloader_idx_0=9.44489288.ckpt' # 'pretrained/deca_model.tar'
cfg.emoca.pretrained_spectre_path = 'pretrained/spectre_model.tar'
cfg.emoca.spectre_backbone = 'mobilenetv2' # perceptual encoder backbone'

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