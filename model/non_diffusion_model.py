import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  
from utils import dist_util
import math
from model.wav2vec import Wav2Vec2Model
from model.networks import TransformerDecoderFiLM, TransformerDecoderLayerFiLM, TransformerEncoderFiLM, TransformerEncoderLayerFiLM
from model.denoising_model import *

# use FiLM layer to inject diffusion timestep information
class FaceTransformerFLINT(nn.Module):
    def __init__(
        self,
        arch,
        latent_dim=256, 
        ff_size=1024, 
        num_enc_layers=2, 
        num_heads=4, 
        dropout=0.1,
        activation="gelu", 
        dataset='FaMoS',
        **kwargs):
        super().__init__()

        self.tag = 'FaceTransformerFLINT_non_diffusion'
        self.input_feats = 128  # latent dim of the FLINT motion prior
        self.dataset = dataset
        self.use_mask = use_mask
        self.latent_dim_condition = latent_dim // 3
        self.latent_dim_transformer_encoder = latent_dim
        self.latent_dim_transformer_decoder = latent_dim // 2


        self.ff_size = ff_size
        self.num_enc_layers = num_enc_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.cond_mask_prob = kwargs.get('cond_mask_prob', 0.)
        self.audio_mask_prob = kwargs.get('audio_mask_prob', 0.)
        self.arch = arch
        
        ### layers
        # self.image_encoder = torch.hub.load('pytorch/vision:v0.8.1', 'mobilenet_v2', pretrained=True)
        # image_feature_dim = 1280
        # self.image_process = InputProcess(image_feature_dim, self.latent_dim)

        self.lmk2d_dim = 468 * 2
        self.lmk_process = InputProcess(self.lmk2d_dim, self.latent_dim_condition)

        # wav2vec 2.0 weights initialization
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder.feature_extractor._freeze_parameters()
        audio_feature_dim = 768
        self.audio_process = InputProcess(audio_feature_dim, self.latent_dim_condition * 2)
       
        self.input_process = InputProcess(self.input_feats, self.latent_dim_transformer_decoder)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim_transformer_encoder, self.dropout)
        self.sequence_pos_decoder = PositionalEncoding(self.latent_dim_transformer_decoder, self.dropout)

        # embed the diffusion timestep
        self.embed_timestep = TimestepEmbedder(self.latent_dim_transformer_decoder, self.sequence_pos_decoder)
        target_mask = neighborhood_mask(target_size=200, num_heads = self.num_heads, bias_step=2, symm=False)
        self.register_buffer('tgt_mask', target_mask)
        
        print(f"[{self.tag}] Using transformer as backbone.")

        # for feature fusion of conditions
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.latent_dim_transformer_encoder,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=dropout,
            activation=self.activation,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=self.num_enc_layers
        )

         # to downsample seqlen to seqlen // 8
        self.squash_layer = self.create_squasher(
            input_dim=self.latent_dim_transformer_encoder, 
            hidden_dim=self.latent_dim_transformer_encoder, 
            quant_factor=3)
        
        self.outputprocess_motion = MotionOutput(output_feats=self.input_feats, latent_dim=self.latent_dim_transformer_encoder)
    
     # downsample the fps of the original video sequence to T / 8 (temporal CNN)
    def create_squasher(self, input_dim, hidden_dim, quant_factor):
        layers = [nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim,5,stride=2,padding=2,
                            padding_mode='replicate'),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm1d(hidden_dim))]
        for _ in range(1, quant_factor):
            layers += [nn.Sequential(
                        nn.Conv1d(hidden_dim,hidden_dim,5,stride=1,padding=2,
                                    padding_mode='replicate'),
                        nn.LeakyReLU(0.2, True),
                        nn.BatchNorm1d(hidden_dim),
                        nn.MaxPool1d(2)
                        )]
        squasher = nn.Sequential(*layers)
        return squasher
    
    def mask_audio_cond(self, audio_emb):
        """
        audio_emb: [bs, c]
        """
        bs = audio_emb.shape[0]
        mask = torch.bernoulli(
            torch.ones(bs, device=audio_emb.device) * self.audio_mask_prob
        )
        mask = mask.view(bs, 1)
        # 1-> use null_cond, 0-> use real cond
        return audio_emb * (1.0 - mask)

    def freeze_wav2vec(self):
        self.audio_encoder.freeze_encoder()
    
    def unfreeze_wav2vec(self):
        self.audio_encoder.unfreeze_encoder()
    
    def forward(self, image=None, lmk_2d=None, img_mask=None, lmk_mask=None, audio_input=None, **kwargs):
        """
        timesteps: [bs] (int)
        images: [bs, nframes, 3, 224, 224]
        """
        bs, n = lmk_2d.shape[:2]
        # print(lmk_mask.shape)
        # print(lmk_2d.shape)
        lmk_2d = lmk_2d[...,:2].clone() * (lmk_mask.unsqueeze(-1))

        vis_cond = self.lmk_process(lmk_2d.reshape(bs, n, -1))  # [seqlen, bs, d]
        
        audio_input = self.mask_audio_cond(audio_input)
        audio_emb = self.audio_encoder(audio_input, frame_num=n).last_hidden_state
        audio_cond = self.audio_process(audio_emb)
        
        # # concat the condition
        cond_emb = torch.cat([vis_cond, audio_cond], dim=-1)
        condseq = self.sequence_pos_encoder(cond_emb)  # [seqlen, bs, 2d]

        # transformer encoder to get memory
        encoder_output = self.transformer_encoder(condseq)

        # downsample tp match x
        encoder_output = self.squash_layer(encoder_output.permute(1,2,0)).permute(2,0,1)    # [seqlen//8, bs, d]

        output = self.outputprocess_motion(encoder_output)  # [bs, seqlen//8, input_nfeats]
        return output