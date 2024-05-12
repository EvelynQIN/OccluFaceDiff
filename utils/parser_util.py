# MIT License
# Copyright (c) 2022 Guy Tevet
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import json
import os
from argparse import ArgumentParser


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_transformer_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ["dataset", "transformer", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    assert os.path.exists(args_path), f"Arguments json file -- {args_path} was not found!"
    with open(args_path, "r") as fr:
        model_args = json.load(fr)
    for a in args_to_overwrite:
        if a in model_args.keys():
            # Use the chosen dataset, or use the dataset that is used to train the model
            if a == "dataset":
                if args.__dict__[a] is None:
                    args.__dict__[a] = model_args[a]
            else:
                args.__dict__[a] = model_args[a]
        else:
            print(
                "Warning: was not able to load [{}], using default value [{}] instead.".format(
                    a, args.__dict__[a]
                )
            )
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError("group_name was not found.")


def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument("model_path")
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except Exception:
        raise ValueError("model_path argument must be specified.")


def add_base_options(parser):
    group = parser.add_argument_group("base")
    
    group.add_argument(
        "--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU."
    )
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument(
        "--batch_size", default=64, type=int, help="Batch size during training."
    )
    group.add_argument(
        "--timestep_respacing", default="", type=str, help="ddim timestep respacing."
    )
    group.add_argument(
        "--config_path", default='', type=str, help="the path to the config yaml file"
    )


def add_diffusion_options(parser):
    group = parser.add_argument_group("diffusion")
    group.add_argument(
        "--noise_schedule",
        default="cosine",
        choices=["linear", "cosine"],
        type=str,
        help="Noise schedule type",
    )
    group.add_argument(
        "--diffusion_steps",
        default=1000,
        type=int,
        help="Number of diffusion steps (denoted T in the paper)",
    )
    group.add_argument(
        "--sigma_small", default=True, type=bool, help="Use smaller sigma values."
    )

def add_transformer_options(parser):
    group = parser.add_argument_group("transformer")
    group.add_argument(
        "--arch",
        default="",
        type=str,
        help="Architecture types as reported in the paper.",
    )
    group.add_argument(
        "--target_nfeat", default=433, type=int, help="motion feature dimension"
    )

    group.add_argument(
        "--lmk3d_dim", default=68*3, type=int, help="3d lmks signal feature dimension"
    )
    
    group.add_argument(
        "--lmk2d_dim", default=68*2, type=int, help="2d lmks signal feature dimension"
    )
    group.add_argument(
        "--num_enc_layers", default=1, type=int, help="Number of encoder layers."
    )
    group.add_argument(
        "--num_dec_layers", default=1, type=int, help="Number of decoder layers."
    )
    group.add_argument(
        "--num_heads", default=4, type=int, help="Number of multi heads in attention"
    )
    group.add_argument(
        "--ff_size", default=1024, type=int, help="latent dim of point-wise feed forword net"
    )
    group.add_argument(
        "--dropout", default=0.1, type=float, help="dropout rate"
    )
    group.add_argument(
        "--latent_dim", default=512, type=int, help="latent dimension"
    )
    group.add_argument(
        "--cond_mask_prob",
        default=0.0,
        type=float,
        help="The probability of masking the condition during training."
        " For classifier-free guidance learning.",
    )
    group.add_argument(
        "--audio_mask_prob",
        default=0.0,
        type=float,
        help="The probability of masking all the audio input.",
    )
    group.add_argument(
        "--use_mask", default=True, type=bool, help="whether to use alibi casual mask for decoder self attention."
    )
    group.add_argument(
        "--no_normalization",
        action="store_true",
        help="no data normalisation for the 6d motions",
    )
    group.add_argument(
        "--flame_model_path", default='flame_2020/generic_model.pkl', type=str, help="the path to the flame model"
    )
    group.add_argument(
        "--flame_lmk_embedding_path", default='flame_2020/landmark_embedding.npy', type=str, help="the path to the flame landmark embeddings"
    )


def add_mlp_options(parser):
    group = parser.add_argument_group("mlp")
    group.add_argument(
        "--arch",
        default="DiffMLP",
        type=str,
        help="Architecture types as reported in the paper.",
    )
    
    group.add_argument(
        "--lmk2d_dim", default=68*2, type=int, help="2d lmks signal feature dimension"
    )

    group.add_argument(
        "--cond_latent_dim", default=512, type=int, help="latent dimension of the sparse landmarks."
    )
    
    group.add_argument(
        "--num_layers", default=8, type=int, help="Number of layers of the diffMLP motion branch."
    )
    
    group.add_argument(
        "--input_latent_dim", default=512, type=int, help="latent dimension of the diffMLP motion branch."
    )

    group.add_argument(
        "--dropout", default=0.1, type=float, help="dropout rate."
    )

    group.add_argument(
        "--cond_mask_prob",
        default=0.0,
        type=float,
        help="The probability of masking the condition during training."
        " For classifier-free guidance learning.",
    )

    group.add_argument(
        "--audio_mask_prob",
        default=0.0,
        type=float,
        help="The probability of masking the audio condition during training."
        " For classifier-free guidance learning.",
    )

    group.add_argument(
        "--input_motion_length",
        default=150,
        type=int,
        help="Limit for the maximal number of frames.",
    )

    group.add_argument(
        "--no_normalization",
        action="store_true",
        help="no data normalisation for the 6d motions",
    )


def add_data_options(parser):
    group = parser.add_argument_group("dataset")
    group.add_argument(
        "--dataset",
        default=None,
        choices=[
            "FaMoS",
            "multiface",
            "mead_25fps",
            "vocaset"
        ],
        type=str,
        help="Dataset name (choose from list).",
    )
    group.add_argument(
        "--dataset_path",
        default="./dataset/FaMoS/",
        type=str,
        help="Dataset path",
    )

    group.add_argument(
        "--scale", default=1.5, type=float, help="scale to crop the lmks."
    )

    group.add_argument(
        "--trans_scale", default=0, type=float, help="scale to translate the image center."
    )

    group.add_argument(
        "--image_size", default=224, type=int, help="size of image to scrop."
    )
    
    group.add_argument(
        "--n_shape", default=100, type=int, help="number of flame params."
    )

    group.add_argument(
        "--n_exp", default=50, type=int, help="number of flame params."
    )
    
    group.add_argument(
        "--n_pose", default=30, type=int, help="number of flame params."
    )

    group.add_argument(
        "--n_trans", default=3, type=int, help="number of flame params."
    )

    group.add_argument(
        "--fps", default=30, type=int, help="fps of the motion sequence."
    )

    group.add_argument(
        "--load_tex", action="store_true", help="whether load tex from emica reconstruction."
    )

    group.add_argument(
        "--use_iris", action="store_true", help="whether use iris lmks from mediapipe."
    )
    

def add_training_options(parser):
    group = parser.add_argument_group("training")
    group.add_argument(
        "--save_dir",
        # required=True,
        type=str,
        help="Path to save checkpoints and results.",
    )
    group.add_argument(
        "--wandb_log", default=False, type=bool, help="use wandb as the log in platform."
    )
    group.add_argument(
        "--overwrite",
        action="store_true",
        help="If True, will enable to use an already existing save_dir.",
    )
    group.add_argument(
        "--train_platform_type",
        default="NoPlatform",
        choices=["NoPlatform", "ClearmlPlatform", "TensorboardPlatform"],
        type=str,
        help="Choose platform to log results. NoPlatform means no logging.",
    )
    group.add_argument("--lr", default=2e-4, type=float, help="Learning rate.")
    group.add_argument(
        "--weight_decay", default=0.0, type=float, help="Optimizer weight decay."
    )
    group.add_argument(
        "--lr_anneal_steps",
        default=0,
        type=int,
        help="Number of learning rate anneal steps.",
    )
    group.add_argument(
        "--cosine_scheduler",
        action="store_true",
        help="If True, use the cosine scheduler with warm up steps.",
    )
    group.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Number of warm up steps of the cosine scheduler.",
    )
    
    group.add_argument(
        "--train_dataset_repeat_times",
        default=1000,
        type=int,
        help="Repeat the training dataset to save training time",
    )

    group.add_argument(
        "--log_interval", default=100, type=int, help="Log losses each N steps"
    )
    group.add_argument(
        "--save_interval",
        default=1,
        type=int,
        help="Save checkpoints and run evaluation each N epochs",
    )
    group.add_argument(
        "--num_epoch",
        default=100,
        type=int,
        help="Training will stop after the specified number of epochs.",
    )
    group.add_argument(
        "--resume_checkpoint",
        default="",
        type=str,
        help="If not empty, will start from the specified checkpoint (path to model###.pt file).",
    )
    group.add_argument(
        "--load_optimizer",
        action="store_true",
        help="If True, will also load the saved optimizer state for network initialization",
    )
    group.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of dataloader workers.",
    )

    group.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="update gradient every n steps",
    )

    group.add_argument(
        "--occlusion_mask_prob",
        default=0,
        type=float,
        help="Probability for adding random occlusion mask.",
    )
    
    group.add_argument(
        "--freeze_audio_encoder_interval",
        default=1,
        type=int,
        help="Unfreeze audio encoder every i epoch.",
    )

def add_sampling_options(parser):
    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--split",
        default="test",
        type=str,
        help="data split from which to sample",
    )

    group.add_argument(
        "--vis",
        action="store_true",
        help="visualize the output during evaluation",
    )

    group.add_argument(
        "--fix_noise",
        action="store_true",
        help="fix init noise for the output",
    )

    group.add_argument(
        "--model_path",
        type=str,
        help="Path to model####.pt file to be sampled.",
    )
    group.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="path to the folder to store the rendered video"
    )
    
    group.add_argument(
        "--occlusion_mask_prob",
        default=0,
        type=float,
        help="Probability for adding random occlusion mask.",
    )

    group.add_argument(
        "--mixed_occlusion_prob",
        default=0,
        type=float,
        help="Probability for adding mixed random occlusion mask.",
    )

def add_predict_options(parser):
    group = parser.add_argument_group("predict")

    group.add_argument(
        "--fix_noise",
        action="store_true",
        help="fix init noise for the output.",
    )

    group.add_argument(
        "--input_motion_length",
        default=20,
        type=int,
        help="motion length to chunk over the original sequence.",
    )
    
    group.add_argument(
        "--sld_wind_size",
        default=15,
        type=int,
        help="slide window size.",
    )
    
    group.add_argument(
        "--save_folder",
        default="vis_result",
        type=str,
        help="folder to save visualization result.",
    )
    
    group.add_argument(
        "--model_path",
        type=str,
        help="Path to model####.pt file to be sampled.",
    )

    
    group.add_argument(
        "--with_audio",
        action="store_true",
        help="whether the input with audio.",
    )

    group.add_argument(
        "--exp_name",
        type=str,
        default='',
        help='expriment name of the prediction'
    )
    
    # for in_the_wild mode #
    group.add_argument(
        "--video_path",
        type=str,
        help="video to track, must provide with in_the_wild test_mode.",
    )

def add_test_options(parser):
    group = parser.add_argument_group("test")

    group.add_argument(
        "--fix_noise",
        action="store_true",
        help="fix init noise for the output.",
    )

    group.add_argument(
        "--input_motion_length",
        default=30,
        type=int,
        help="motion length to chunk over the original sequence.",
    )

    group.add_argument(
        "--split",
        default="test",
        type=str,
        help="split in the dataset",
    )
    
    group.add_argument(
        "--sld_wind_size",
        default=20,
        type=int,
        help="slide window size.",
    )
    
    group.add_argument(
        "--save_folder",
        default="vis_result",
        type=str,
        help="folder to save visualization result.",
    )
    
    group.add_argument(
        "--model_path",
        type=str,
        help="Path to model####.pt file to be sampled.",
    )
    
    group.add_argument(
        "--exp_name",
        type=str,
        help="name of the experiment.",
    )
    
    group.add_argument(
        "--subject_id",
        default=None,
        type=str,
        help="subject id.",
    )

    group.add_argument(
        "--level",
        default=None,
        type=str,
        help="emotion level.",
    )

    group.add_argument(
        "--sent",
        default=None,
        type=str,
        help="sent id in MEAD [3 digit].",
    )

    group.add_argument(
        "--emotion",
        default=None,
        type=str,
        help="emotion id in MEAD.",
    )

    group.add_argument(
        "--vis",
        action="store_true",
        help="whether to visualize the output.",
    )

    group.add_argument(
        "--with_audio",
        action="store_true",
        help="whether the input with audio.",
    )

    
def add_evaluation_options(parser):
    group = parser.add_argument_group("eval")
    group.add_argument(
        "--model_path",
        # required=True,
        type=str,
        help="Path to model####.pt file to be sampled.",
    )


def train_mlp_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_mlp_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()

def train_trans_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_transformer_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()

def sample_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    return parse_and_load_from_model(parser)


def predict_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_predict_options(parser)
    return parse_and_load_from_model(parser)

def test_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_test_options(parser)
    return parse_and_load_from_model(parser)