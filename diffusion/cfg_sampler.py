"""
This implementation is copied from MDM https://github.com/GuyTevet/motion-diffusion-model/blob/dd0d0030b3659255fe9779e14852d2028269985f/model/cfg_sampler.py#L8
"""

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        # # pointers to inner model
        # self.rot2xyz = self.model.rot2xyz
        # self.translation = self.model.translation

    def forward(self, x, timesteps, y=None, **model_kwargs):
        
        # full conditioning with audio + landmark
        out = self.model(x, timesteps, **model_kwargs)

        # condition only on audio
        y_cond_audio = deepcopy(model_kwargs)
        y_cond_audio['uncond_lmk'] = True
        out_cond_audio = self.model(x, timesteps, **y_cond_audio)

        # fully unconditioned
        y_uncond = deepcopy(model_kwargs)
        y_uncond['uncond_all'] = True
        out_uncond = self.model(x, timesteps, **y_uncond)

        scale_full = y['scale_all'].view(-1, 1, 1)
        scale_audio = y['scale_audio'].view(-1, 1, 1)

        final_out = (1-scale_full-scale_audio) * out_uncond + scale_full * out + scale_audio * out_cond_audio

        return final_out
