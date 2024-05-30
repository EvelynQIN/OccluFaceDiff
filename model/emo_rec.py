"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""

from model.deca import MLP
import torch
from torch import nn
from torch.nn import BatchNorm1d, LayerNorm, InstanceNorm1d, Module
import numpy as np
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
import sys


class EmoRec(Module):
    """
    Code adapted from EMOCA
    EmoDECA loads a pretrained DECA-based face reconstruction net and uses it to predict emotion
    """

    def __init__(self, config):
        super().__init__()
        # which latent codes are being used
        self.config = config
        in_size = 0
        if self.config.model.use_identity:
            in_size += config.model.deca_cfg.model.n_shape
        if self.config.model.use_expression:
            in_size += config.model.deca_cfg.model.n_exp
        if self.config.model.use_global_pose:
            in_size += 3
        if self.config.model.use_jaw_pose:
            in_size += 3
        # if self.config.model.use_detail_code:
        #     in_size += config.model.deca_cfg.model.n_detail

        if 'mlp_dimension_factor' in self.config.model.keys():
            dim_factor = self.config.model.mlp_dimension_factor
            dimension = in_size * dim_factor
        elif 'mlp_dim' in self.config.model.keys(): # 2048
            dimension = self.config.model.mlp_dim
        else:
            dimension = in_size

            
        hidden_layer_sizes = config.model.num_mlp_layers * [dimension]

        out_size = 0
        if self.config.model.predict_expression:
            # self.num_classes = 9
            self.num_classes = self.config.data.n_expression if 'n_expression' in self.config.data.keys() else 9
            out_size += self.num_classes
        if self.config.model.predict_valence:
            out_size += 1
        if self.config.model.predict_arousal:
            out_size += 1
        # if self.config.predict_AUs: # false
        #     out_size += self.predicts_AUs()

        if 'mlp_norm_layer' in self.config.model.keys():
            batch_norm = BatchNorm1d
        else:
            batch_norm = None
        self.mlp = MLP(in_size, out_size, hidden_layer_sizes, batch_norm=batch_norm)

        self.emonet = None

        # activation funtion
        self.exp_activation = F.log_softmax
        self.a_activation = None
        self.v_activation = None
    
    def load_from_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.load_state_dict(checkpoint['state_dict'], strict=False)

    def forward(self, flame_rec):

        shapecode = flame_rec['shape']
        # texcode = values['texcode']
        expcode = flame_rec['exp']

        global_pose = flame_rec['global_pose']
        jaw_pose = flame_rec['jaw']

        values = {}
        if self.mlp is not None:
            input_list = []

            if self.config.model.use_identity:
                input_list += [shapecode]

            if self.config.model.use_expression:
                input_list += [expcode]

            if self.config.model.use_global_pose:
                input_list += [global_pose]

            if self.config.model.use_jaw_pose:
                input_list += [jaw_pose]

            input = torch.cat(input_list, dim=1)
            output = self.mlp(input)

            out_idx = 0
            if self.config.model.predict_expression:
                expr_classification = output[:, out_idx:(out_idx + self.num_classes)]
                if self.exp_activation is not None:
                    expr_classification = self.exp_activation(output[:, out_idx:(out_idx + self.num_classes)], dim=1)
                out_idx += self.num_classes
            else:
                expr_classification = None

            if self.config.model.predict_valence:
                valence = output[:, out_idx:(out_idx+1)]
                if self.v_activation is not None:
                    valence = self.v_activation(valence)
                out_idx += 1
            else:
                valence = None

            if self.config.model.predict_arousal:
                arousal = output[:, out_idx:(out_idx+1)]
                if self.a_activation is not None:
                    arousal = self.a_activation(output[:, out_idx:(out_idx + 1)])
                out_idx += 1
            else:
                arousal = None

            values["valence"] = valence
            values["arousal"] = arousal
            values["expr_classification"] = expr_classification

        return values