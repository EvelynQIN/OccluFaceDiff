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
        num_dec_layers=3,
        num_heads=4, 
        dropout=0.1,
        activation="gelu", 
        dataset='FaMoS',
        **kwargs):
        super().__init__()

        self.tag = 'FaceTransformerFLINT_non_diffusion'
        self.input_feats = 128  # latent dim of the FLINT motion prior
        self.dataset = dataset
        self.latent_dim_condition = latent_dim // 3
        self.latent_dim_transformer_encoder = latent_dim
        self.latent_dim_transformer_decoder = latent_dim // 2


        self.ff_size = ff_size
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
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
        self.audio_process = InputProcessBN(audio_feature_dim, self.latent_dim_condition * 2)
       
        self.input_process = InputProcessBN(self.input_feats, self.latent_dim_transformer_decoder)
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
            hidden_dim=self.latent_dim_transformer_decoder, 
            quant_factor=3)
        
        # for feature fusion of conditions
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model = self.latent_dim_transformer_decoder,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=dropout,
            activation=self.activation,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=transformer_decoder_layer,
            num_layers=self.num_dec_layers
        )

        self.outputprocess_motion = MotionOutput(output_feats=self.input_feats, latent_dim=self.latent_dim_transformer_decoder)
    
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
    
    def forward(self, image=None, lmk_2d=None, img_mask=None, lmk_mask=None, audio_input=None, teacher_forcing=False, **kwargs):
        """
        timesteps: [bs] (int)
        images: [bs, nframes, 3, 224, 224]
        target: latent code of FLINT [bs, n//8, 128]
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
        cond_emb = torch.cat([vis_cond, audio_cond], dim=-1) # [bs, seqlen, 2d]
        condseq = self.sequence_pos_encoder(cond_emb)  # [seqlen, bs, 2d]
        # transformer encoder to get memory
        encoder_output = self.transformer_encoder(condseq)
        # downsample tp match x
        src_seq = self.squash_layer(encoder_output.permute(1,2,0)).permute(2,0,1)    # [seqlen//8, bs, d] transformer condition

        if teacher_forcing:
            target = kwargs['target']
             # cross attention of the sparse cond & motion_output
            target = self.input_process(target)   # [seqlen//8, bs, d]
            target_input = target[:-1] # shift one position
            target_seq = self.sequence_pos_decoder(target_input)

            # alibi mask for self attention in transformer decoder
            T = target_seq.shape[0]
            tgt_mask = self.tgt_mask[:, :T, :T].clone().detach().to(device=target_seq.device)    # (num_heads, seqlen, seqlen)
            tgt_mask = tgt_mask.repeat(bs, 1, 1)

            # alignment mask for cross attention
            memory_mask = enc_dec_mask(target_seq.device, target_seq.shape[0], src_seq.shape[0])
            decoder_output = self.transformer_decoder(target_seq, src_seq, tgt_mask=tgt_mask, memory_mask=memory_mask)
            output = self.outputprocess_motion(decoder_output)
        else:
            for i in range(n//8):
                if i == 0:
                    target_input = torch.zeros((1, bs, self.latent_dim_transformer_decoder)).to(src_seq.device)
                target_seq = self.sequence_pos_decoder(target_input)
                 # alibi mask for self attention in transformer decoder
                T = target_seq.shape[0]
                tgt_mask = self.tgt_mask[:, :T, :T].clone().detach().to(device=target_seq.device)    # (num_heads, seqlen, seqlen)
                tgt_mask = tgt_mask.repeat(bs, 1, 1)

                # alignment mask for cross attention
                memory_mask = enc_dec_mask(target_seq.device, target_seq.shape[0], src_seq.shape[0])
                decoder_output = self.transformer_decoder(target_seq, src_seq, tgt_mask=tgt_mask, memory_mask=memory_mask)
                output = self.outputprocess_motion(decoder_output)  # [bs, i, 128]

                new_output = self.input_process(output[:,-1:,:])
                target_input = torch.cat([target_input, new_output], dim=0)
        return output


class InputProcessBN(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.linear_map = nn.Linear(self.input_feats, self.latent_dim)
        self.bn = nn.BatchNorm1d(self.latent_dim)
        # self.inputEmbedding = nn.Sequential(
        #     nn.Linear(self.input_feats, self.latent_dim),
        #     nn.BatchNorm1d(self.latent_dim),
        #     # nn.LeakyReLU(0.2, inplace=True)
        # )

    def forward(self, x):
        # bs, nframes, motion_nfeats = x.shape
        x = self.linear_map(x)  # [bs, n, c] [bs, c, n]
        x = self.bn(x.permute(0, 2, 1)).permute(2, 0, 1)  # (nframes, bs, motion_nfeats)

        return x

class MotionOutputBN(nn.Module):
    def __init__(self, output_feats, latent_dim):
        super().__init__()
        self.output_feats = output_feats
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            # nn.Linear(self.latent_dim, self.latent_dim // 2),
            # nn.BatchNorm1d(self.latent_dim // 2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_dim, self.output_feats)
        )
        

    def forward(self, output):
        output = self.fc(output)  # [nframes, bs, nfeats]
        output = output.permute(1, 0, 2)  # [bs, nframes, nfeats]
        return output