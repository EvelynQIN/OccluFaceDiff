# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import resnet
import os

class ResnetEncoder(nn.Module):
    def __init__(self, outsize):
        super(ResnetEncoder, self).__init__()

        feature_size = 2048

        self.encoder = resnet.load_ResNet50Model() #out: 2048
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )

    def forward(self, inputs):
        inputs_ = inputs
        if inputs.ndim == 5: # batch of videos
            B, T, C, H, W = inputs.shape
            inputs_ = inputs.view(B * T, C, H, W)
        features = self.encoder(inputs_)
        parameters = self.layers(features)
        if inputs.ndim == 5: # batch of videos
            parameters = parameters.view(B, T, -1)
        return parameters

class EMOCA(nn.Module):
    def __init__(self, model_cfg):
        super(EMOCA, self).__init__()

        # self.model_cfg = model_cfg
        self.model_path = model_cfg.ckpt_path
        self.n_param = model_cfg.n_shape + model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam, model_cfg.n_light]
        self.param_dict = {i: model_cfg.get('n_' + i) for i in model_cfg.param_list}
        
        # layers
        self.E_flame = ResnetEncoder(self.n_param)
        self.E_expression = ResnetEncoder(model_cfg.n_exp)

        # load from ckpt
        self._load_model_from_ckpt()
        
        # freeze encoders
        self._freeze_encoders()
    
    def _load_model_from_ckpt(self):
        # resume model from ckpt path
        if os.path.exists(self.model_path):
            print(f"[EMOCA] Pretrained model found at {self.model_path}.")
            checkpoint = torch.load(self.model_path)

            if 'state_dict' in checkpoint.keys():
                checkpoint = checkpoint['state_dict']
            else:
                checkpoint = checkpoint

            processed_checkpoint = {}
            processed_checkpoint["E_flame"] = {}
            processed_checkpoint["E_expression"] = {}

            if 'deca' in list(checkpoint.keys())[0]:
                for key in checkpoint.keys():
                    k = key.replace("deca.","")
                    if "E_flame" in key:
                        processed_checkpoint["E_flame"][k.replace("E_flame.","")] = checkpoint[key]
                    elif "E_expression" in key:
                        processed_checkpoint["E_expression"][k.replace("E_expression.","")] = checkpoint[key]#.replace("E_flame","")
                    else:
                        pass
            else:
                processed_checkpoint = checkpoint
            self.E_flame.load_state_dict(processed_checkpoint['E_flame'], strict=True) 
            self.E_expression.load_state_dict(processed_checkpoint['E_expression'], strict=True)
        else:
            raise(f'please check model path: {self.model_path}')
    
    def _freeze_encoders(self,):
         # freeze the encoders throughout training
        self.E_flame.eval()
        self.E_expression.eval()
        self.E_flame.requires_grad_(False)
        self.E_expression.requires_grad_(False)
        print(f'[EMOCA] All encoders are frozen and set to be eval mode.')
    
    def decompose_deca_code(self, code):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0

        for key in self.param_dict:
            end = start + int(self.param_dict[key])
            code_dict[key] = code[..., start:end]
            start = end
            if key == 'light':
                dims_ = code_dict[key].ndim -1 # (to be able to handle batches of videos)
                code_dict[key] = code_dict[key].reshape(*code_dict[key].shape[:dims_], 9, 3)
        return code_dict

    def forward(self, inputs):
        deca_code = self.E_flame(inputs)
        exp_code = self.E_expression(inputs)
        code_dict = self.decompose_deca_code(deca_code)
        code_dict['exp'] = exp_code
        return code_dict


class ExpressionLossNet(nn.Module):
    """ Code borrowed from EMOCA https://github.com/radekd91/emoca """
    def __init__(self):
        super(ExpressionLossNet, self).__init__()

        self.backbone = resnet.load_ResNet50Model() #out: 2048

        self.linear = nn.Sequential(
            nn.Linear(2048, 10))

    def forward2(self, inputs):
        features = self.backbone(inputs)
        out = self.linear(features)
        return features, out

    def forward(self, inputs):
        features = self.backbone(inputs)
        return features