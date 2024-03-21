# MIT License
# Copyright (c) 2022 Guy Tevet
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from diffusion import gaussian_diffusion as gd
from diffusion.respace import space_timesteps, SpacedDiffusion
from model.meta_model import MultiBranchMLP
from model import denoising_model
from model.calibration_layer import Cam_Calibration
import importlib

def find_model_using_name(model_dir, model_name):
    # Taken from https://github.com/Zielon/MICA/blob/master/utils/util.py#L27
    model_filename = model_dir + "." + model_name
    modellib = importlib.import_module(model_filename, package=model_dir)

    # In the file, the class called ModelName() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '')
    for name, cls in modellib.__dict__.items():
        # if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print("In %s.py, there should be a class with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if len(unexpected_keys) != 0:
        state_dict_new = {}
        for key in state_dict.keys():
            state_dict_new[key.replace("module.", "")] = state_dict[key]
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict_new, strict=False
        )
    assert len(unexpected_keys) == 0
    assert all([k.startswith("clip_model.") for k in missing_keys])

def create_model_and_diffusion(args):
    arch = args.arch  
    if "Enc" in arch:
        denoise_model = denoising_model.TransformerEncoder(**get_transformer_args(args))
    elif "Trans" in arch:
        denoise_model = denoising_model.FaceTransformer(**get_transformer_args(args))
    elif "GRU" in arch:
        denoise_model = denoising_model.GRUDecoder(**get_transformer_args(args))
    elif "MLP" in arch:
        denoise_model = MultiBranchMLP(**get_mlp_args(args))
    else:
        raise  ValueError("Invalid architecture name!")
    diffusion = create_gaussian_diffusion(args)
    return denoise_model, diffusion

def get_transformer_args(args):
    return {
        "arch": args.arch,
        "nfeats": args.target_nfeat,
        "lmk3d_dim": args.lmk3d_dim,
        "lmk2d_dim": args.lmk2d_dim,
        "latent_dim": args.latent_dim,
        "ff_size": args.ff_size,
        "num_enc_layers": args. num_enc_layers,
        "num_dec_layers": args.num_dec_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "dataset": args.dataset,
        "use_mask": args.use_mask,
        "cond_mask_prob": args.cond_mask_prob,
        "n_shape": args.n_shape,
        "n_exp": args.n_exp,
        "n_pose": args.n_pose,
        "n_trans": args.n_trans
   }

def get_cam_args(args):
    return {
        "lmk2d_dim": args.lmk2d_dim,
        "n_shape": args.n_shape,
        "output_feature_dim": args.output_nfeat,
        "latent_dim": args.latent_dim,
        "ckpt_path": args.ckpt_path,
    }

def get_mlp_args(args, mica_args):

    return {
        "mica_args": mica_args,
        "nfeats": args.target_nfeat,
        "lmk3d_dim": args.lmk3d_dim,
        "lmk2d_dim": args.lmk2d_dim,
        "cond_latent_dim": args.cond_latent_dim,
        "shape_num_layers": args.shape_num_layers,
        "shape_latent_dim": args.shape_latent_dim,
        "motion_latent_dim": args.motion_latent_dim,
        "motion_num_layers": args.motion_num_layers,
        "trans_num_layers": args.trans_num_layers,
        "trans_latent_dim": args.trans_latent_dim,
        "dropout": args.dropout,
        "dataset": args.dataset,
        "cond_mask_prob": args.cond_mask_prob,
        "input_motion_length": args.input_motion_length,
    }


def create_gaussian_diffusion(args):
    predict_xstart = True
    steps = args.diffusion_steps  # 1000
    scale_beta = 1.0
    timestep_respacing = args.timestep_respacing
    learn_sigma = False
    rescale_timesteps = False
    
    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        dataset=args.dataset,
        flame_model_path = args.flame_model_path,
        flame_lmk_embedding_path = args.flame_lmk_embedding_path,
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
