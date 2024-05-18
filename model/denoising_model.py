import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  
from utils import dist_util
import math
from model.wav2vec import Wav2Vec2Model
from model.networks import TransformerDecoderFiLM, TransformerDecoderLayerFiLM

def neighborhood_mask(target_size, num_heads, bias_step, symm=False):        
    """compute the target mask, decay the weight for longer steps on the left (casual mask)

    Args:
        target_size: size of the input mask
        weight_decay: slope of decaying weight
        bias_step: number of steps to decay the weight
        symm: symm=False return the casual left mask for tgt_mask, symm=True return symmetric mask for transformer-encoder attention
    Returns:
        mask: shape (target_size, target_size)
    """
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
        
    weight_decay = torch.Tensor(get_slopes(num_heads))
    bias = torch.arange(start=0, end=target_size, step=bias_step).unsqueeze(1).repeat(1,bias_step).view(-1) // (bias_step)
    bias_right = - bias
    bias_left = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(target_size, target_size)
    if symm:
        for i in range(target_size):
            alibi[i, :i+1] = bias_left[-(i+1):]
            alibi[i, i+1:] = bias_right[:target_size-i-1]
    else:
        for i in range(target_size):
            alibi[i, :i+1] = bias_left[-(i+1):]
    alibi = weight_decay.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(target_size, target_size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    alibi = mask + alibi
    return alibi    # (num_heads, target_size, target_size)

# Alignment Bias
def enc_dec_mask(device, T, S):
    mask = torch.ones((T, S))
    for i in range(T):
        mask[i, i] = 0
    return (mask==1).to(device=device)

class FaceTransformer(nn.Module):
    def __init__(
        self,
        arch,
        latent_dim=256, 
        ff_size=1024, 
        num_enc_layers=2, 
        num_dec_layers=2,
        num_heads=4, 
        dropout=0.1,
        activation="gelu", 
        dataset='FaMoS',
        use_mask=True,
        n_exp = 50,
        n_pose = 6,
        **kwargs):
        super().__init__()

        self.tag = 'FaceTransformer'
        self.nexp = n_exp
        self.npose = n_pose
        self.input_feats = n_exp + n_pose
        self.dataset = dataset
        self.use_mask = use_mask
        if self.use_mask:
            print(f"[{self.tag}] Using alibi mask for decoder.")
        self.latent_dim_condition = latent_dim // 2
        self.latent_dim_transformer = latent_dim

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
        self.audio_process = InputProcess(audio_feature_dim, self.latent_dim_condition)
       
        self.input_process = InputProcess(self.input_feats, self.latent_dim_transformer)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim_transformer, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim_transformer, self.sequence_pos_encoder)
        target_mask = neighborhood_mask(target_size=2000, num_heads = self.num_heads, bias_step=20)
        self.register_buffer('tgt_mask', target_mask)
        
        print(f"[{self.tag}] Using transformer as backbone.")
        self.transformer = nn.Transformer(
            d_model=self.latent_dim_transformer,
            nhead=self.num_heads,
            num_encoder_layers=self.num_enc_layers,
            num_decoder_layers=self.num_dec_layers,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            norm_first=False
        )
        
        self.outputprocess_motion = MotionOutput(output_feats=self.input_feats, latent_dim=self.latent_dim_transformer)

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[1]
        ndim = len(cond.shape)
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            )
            if ndim == 5:
                mask = mask.view(1, bs, 1, 1, 1)
            elif ndim == 3:
                mask = mask.view(1, bs, 1)
            else: 
                raise ValueError( f"Invalid dimension for conditioning mask {cond.shape}")
              # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond
    
    def mask_audio_cond(self, audio_emb):
        """
        audio_emb: [bs, n, c]
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
    
    def forward(self, x, timesteps, image=None, lmk_2d=None, img_mask=None, lmk_mask=None, audio_input=None, force_mask=False, **kwargs):
        """
        x: [bs, nframes, nfeats] 
        timesteps: [bs] (int)
        images: [bs, nframes, 3, 224, 224]
        """
        bs, n = x.shape[:2]
        lmk_2d = lmk_2d[...,:2].clone() * (lmk_mask.unsqueeze(-1))
        ts_emb = self.embed_timestep(timesteps)  # [1, bs, d]

        # # conditions feature extraction
        # image = image.clone() * (img_mask.unsqueeze(2))
        # image_cond = self.image_encoder.features(image.view(bs*n, *image.shape[2:]))
        # image_cond = nn.functional.adaptive_avg_pool2d(image_cond, (1, 1)).squeeze(-1).squeeze(-1).view(bs, n, -1) # [bs, n, image_feature_dim]
        # image_cond = self.image_process(image_cond) # [seqlen, bs, d]
        vis_cond = self.lmk_process(lmk_2d.reshape(bs, n, -1))  # [seqlen, bs, d]
        
        audio_input = self.mask_audio_cond(audio_input)
        audio_emb = self.audio_encoder(audio_input, frame_num=n).last_hidden_state
        
        audio_cond = self.audio_process(audio_emb)
        
        # # if concat
        cond_emb = torch.cat([vis_cond, audio_cond], dim=-1)
        cond_emb = self.mask_cond(cond_emb, force_mask=force_mask)   # [seqlen, bs, 2d]
        
        condseq = self.sequence_pos_encoder(cond_emb)  # [seqlen, bs, 2d]
        
        tgt_mask=None
        if self.use_mask:
            T = x.shape[1]
            tgt_mask = self.tgt_mask[:, :T, :T].clone().detach().to(device=x.device)    # (num_heads, seqlen, seqlen)
            tgt_mask = tgt_mask.repeat(bs, 1, 1)

        # cross attention of the sparse cond & motion_output
        x = self.input_process(x)
        x = x + ts_emb   # broadcast add, [seqlen, bs, d]
        xseq = self.sequence_pos_encoder(x)
        
        # bias alignement mask
        # memory_mask = enc_dec_mask(x.device, xseq.shape[0], condseq.shape[0])
        
        decoder_output = self.transformer(xseq, condseq, tgt_mask=tgt_mask, memory_mask=None) # [seqlen, bs, d]
        output = self.outputprocess_motion(decoder_output)  # [bs, seqlen, input_nfeats]
        return output

# use FiLM layer to inject diffusion timestep information
class FaceTransformerFiLM(nn.Module):
    def __init__(
        self,
        arch,
        latent_dim=256, 
        ff_size=1024, 
        num_enc_layers=2, 
        num_dec_layers=2,
        num_heads=4, 
        dropout=0.1,
        activation="gelu", 
        dataset='FaMoS',
        use_mask=True,
        n_exp = 50,
        n_pose = 6,
        **kwargs):
        super().__init__()

        self.tag = 'FaceTransformer'
        self.nexp = n_exp
        self.npose = n_pose
        self.input_feats = n_exp + n_pose
        self.dataset = dataset
        self.use_mask = use_mask
        if self.use_mask:
            print(f"[{self.tag}] Using alibi mask for decoder.")
        self.latent_dim_condition = latent_dim // 2
        self.latent_dim_transformer = latent_dim

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
        self.audio_process = InputProcess(audio_feature_dim, self.latent_dim_condition)
       
        self.input_process = InputProcess(self.input_feats, self.latent_dim_transformer)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim_transformer, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim_transformer, self.sequence_pos_encoder)
        target_mask = neighborhood_mask(target_size=200, num_heads = self.num_heads, bias_step=10, symm=True)
        self.register_buffer('tgt_mask', target_mask)
        
        print(f"[{self.tag}] Using transformer as backbone.")

        # for feature fusion of conditions
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.latent_dim_transformer,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=dropout,
            activation=self.activation,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=self.num_enc_layers
        )

        # for feature fusion for condition with noisy input
        transformer_decoder_layer = TransformerDecoderLayerFiLM(
            d_model = self.latent_dim_transformer,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=dropout,
            activation=self.activation,
        )
        self.transformer_decoder = TransformerDecoderFiLM(
            decoder_layer=transformer_decoder_layer,
            num_layers=self.num_dec_layers,
        )
        
        self.outputprocess_motion = MotionOutput(output_feats=self.input_feats, latent_dim=self.latent_dim_transformer)

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[1]
        ndim = len(cond.shape)
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            )
            if ndim == 5:
                mask = mask.view(1, bs, 1, 1, 1)
            elif ndim == 3:
                mask = mask.view(1, bs, 1)
            else: 
                raise ValueError( f"Invalid dimension for conditioning mask {cond.shape}")
              # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond
    
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
    
    def forward(self, x, timesteps, image=None, lmk_2d=None, img_mask=None, lmk_mask=None, audio_input=None, force_mask=False, **kwargs):
        """
        x: [bs, nframes, nfeats] 
        timesteps: [bs] (int)
        images: [bs, nframes, 3, 224, 224]
        """
        bs, n = x.shape[:2]
        lmk_2d = lmk_2d[...,:2].clone() * (lmk_mask.unsqueeze(-1))
        ts_emb = self.embed_timestep(timesteps)  # [1, bs, d]

        # # conditions feature extraction
        # image = image.clone() * (img_mask.unsqueeze(2))
        # image_cond = self.image_encoder.features(image.view(bs*n, *image.shape[2:]))
        # image_cond = nn.functional.adaptive_avg_pool2d(image_cond, (1, 1)).squeeze(-1).squeeze(-1).view(bs, n, -1) # [bs, n, image_feature_dim]
        # image_cond = self.image_process(image_cond) # [seqlen, bs, d]
        vis_cond = self.lmk_process(lmk_2d.reshape(bs, n, -1))  # [seqlen, bs, d]
        
        audio_input = self.mask_audio_cond(audio_input)
        audio_emb = self.audio_encoder(audio_input, frame_num=n).last_hidden_state
        audio_cond = self.audio_process(audio_emb)
        
        # # if concat
        cond_emb = torch.cat([vis_cond, audio_cond], dim=-1)
        cond_emb = self.mask_cond(cond_emb, force_mask=force_mask)   # [seqlen, bs, 2d]
        
        condseq = self.sequence_pos_encoder(cond_emb)  # [seqlen, bs, 2d]
        
        tgt_mask=None
        if self.use_mask:
            T = x.shape[1]
            tgt_mask = self.tgt_mask[:, :T, :T].clone().detach().to(device=x.device)    # (num_heads, seqlen, seqlen)
            tgt_mask = tgt_mask.repeat(bs, 1, 1)

        # cross attention of the sparse cond & motion_output
        x = self.input_process(x)
        xseq = self.sequence_pos_encoder(x)
        
        # bias alignement mask
        memory_mask = enc_dec_mask(x.device, xseq.shape[0], condseq.shape[0])

        # transformer encoder to get memory
        encoder_output = self.transformer_encoder(condseq)
        decoder_output = self.transformer_decoder(xseq, encoder_output, ts_emb, tgt_mask=tgt_mask, memory_mask=memory_mask) # [seqlen, bs, d]
        output = self.outputprocess_motion(decoder_output)  # [bs, seqlen, input_nfeats]
        return output

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        arch,
        nfeats,
        sparse_dim=68*3,
        latent_dim=256, 
        ff_size=1024, 
        num_enc_layers=2, 
        num_heads=4, 
        dropout=0.1,
        activation="gelu", 
        dataset='FaMoS',
        nheads_pose=1,
        **kargs):
        super().__init__()

        self.input_feats = nfeats
        self.lmk_dim = sparse_dim
        self.dataset = dataset

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_enc_layers = num_enc_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.lmk_process = Lmk3dProcess(self.lmk_dim, self.latent_dim)
        self.input_process = InputProcess(self.input_feats, self.latent_dim)
        self.merge_input_with_cond = nn.Linear(self.latent_dim*2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.pose_latent_dim = (self.latent_dim // self.num_heads) * nheads_pose 
        self.expr_latent_dim = self.latent_dim - self.pose_latent_dim 
        
        # compute the biased mask 
        mask = neighborhood_mask(target_size=2000, weight_decay=0.2, bias_step=20, symm=True)
        self.register_buffer('mask', mask)
        
        print("Using encoder only as backbone ...")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(
            encoder_layer=seqTransEncoderLayer,
            num_layers=self.num_enc_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.outputprocess_pose = OutputProcess(input_feats=4*6, latent_dim=self.pose_latent_dim)
        self.outputprocess_expr = OutputProcess(input_feats=100, latent_dim=self.expr_latent_dim)

    def mask_cond_lmk(self, cond, force_mask=False):
        bs, n, c = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(
                bs, 1, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond


    def forward(self, x, timesteps, sparse_emb, force_mask=False, **kwargs):
        """
        x: [batch_size, nframes, nfeats] 
        timesteps: [batch_size] (int)
        sparse_emb: [batch_size, nframes, sparse_dim]
        """
        ts_emb = self.embed_timestep(timesteps)  # [1, bs, d]
        lmk_emb = self.lmk_process(
            self.mask_cond_lmk(sparse_emb, force_mask=force_mask)
        ) # [seqlen, bs, d]
        
        # self attention of the sparse cond & motion_output
        x = self.input_process(x)
        x = torch.cat([lmk_emb, x], dim=-1)  # [seqlen, bs, 2d]
        x = self.merge_input_with_cond(x)   # [seqlen, bs, d]
        x = torch.cat([ts_emb, x], dim=0)   # [seqlen+1, bs, d]
        x_seq = self.sequence_pos_encoder(x)
        encoder_output = self.seqTransEncoder(x_seq)[1:]    # [seqlen, bs, d]
        output_pose = self.outputprocess_pose(encoder_output[:, :, :self.pose_latent_dim])  # [bs, seqlen,  4*6]
        output_expr = self.outputprocess_expr(encoder_output[:, :, self.pose_latent_dim:])   # [bs, seqlen, 100]
        output = torch.cat([output_pose, output_expr], dim=-1)  # [bs, seqlen, input_nfeats]
        return output   
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.inputEmbedding = nn.Sequential(
            nn.Linear(self.input_feats, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # bs, nframes, motion_nfeats = x.shape
        x = x.permute(1, 0, 2)  # (nframes, bs, motion_nfeats)
        x = self.inputEmbedding(x)
        return x

class MotionOutput(nn.Module):
    def __init__(self, output_feats, latent_dim):
        super().__init__()
        self.output_feats = output_feats
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.LayerNorm(self.latent_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_dim // 2, self.output_feats)
        )
        

    def forward(self, output):
        output = self.fc(output)  # [nframes, bs, nfeats]
        output = output.permute(1, 0, 2)  # [bs, nframes, nfeats]
        return output
