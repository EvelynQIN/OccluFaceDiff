import numpy as np
import torch
import torch.nn as nn
from model.networks import DiffMLP
from model.wav2vec import Wav2Vec2Model
from utils import dist_util

class TimestepEmbeding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, timesteps):
        return self.pe[timesteps]

class SequenceBranchMLP(nn.Module):
    def __init__(
        self,
        input_nfeats, 
        output_nfeats,
        input_motion_length,
        cond_latent_dim,
        latent_dim=256,
        num_layers=8,
        dropout=0.1,
        dataset="FaMoS",
        **kargs
    ):
        super().__init__()

        self.dataset = dataset

        self.input_feats = input_nfeats
        self.output_feats = output_nfeats
        self.cond_latent_dim = cond_latent_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_motion_length = input_motion_length

        self.input_process = nn.Linear(self.input_feats, self.latent_dim) 
        
        self.embed_timestep = TimestepEmbeding(self.latent_dim)
        
        self.mlp = DiffMLP(
            self.cond_latent_dim, self.latent_dim, seq=self.input_motion_length, num_layers=num_layers)

        self.output_process = nn.Linear(self.latent_dim, self.output_feats)

    def forward(self, x, cond_emb, ts):
        """
        Args:
            x: [batch_size, nframes, nfeats], denoted x_t in the paper
            cond_emb: [batch_size, nframes, cond_latent_dim], the sparse features
            ts: [batch_size,] 
        """

        # Pass the input to a FC
        x = self.input_process(x)
        ts_emb = self.embed_timestep(ts)

        # Concat the sparse feature with the input
        x = torch.cat((cond_emb, x), dim=-1) # [bs, 2 x latent_dim]
        output = self.mlp(x, ts_emb)

        # Pass the output to a FC and reshape the output
        output = self.output_process(output)
        return output
   
class MultiBranchMLP(nn.Module):
    def __init__(
        self,
        lmk2d_dim=68*2,
        cond_latent_dim=256,
        input_latent_dim=128,
        num_layers=1,
        dropout=0.1,
        dataset="mead_25fps",
        n_exp = 50,
        n_pose = 6,
        **kargs,
    ):
        super().__init__()

        self.dataset = dataset
        self.n_exp = n_exp 
        self.n_pose = n_pose 
        self.n_feats = self.n_pose + self.n_exp
        self.lmk2d_dim = lmk2d_dim
        self.dropout = dropout  
        self.cond_latent_dim = cond_latent_dim

        self.num_layers =num_layers
        self.input_latent_dim = input_latent_dim

        self.input_motion_length = kargs.get("input_motion_length")
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.audio_mask_prob = kargs.get("audio_mask_prob", 0.0)

        ### layers
        # self.image_encoder = torch.hub.load('pytorch/vision:v0.8.1', 'mobilenet_v2', pretrained=True)
        # image_feature_dim = 1280
        # self.image_process = nn.Sequential(
        #     nn.Linear(image_feature_dim, self.cond_latent_dim)
        # )
        
        self.lmk2d_process = nn.Sequential(
            nn.Linear(self.lmk2d_dim, self.cond_latent_dim // 2),
            nn.LayerNorm(self.cond_latent_dim // 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        

        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder.feature_extractor._freeze_parameters()
        # wav2vec 2.0 weights initialization
        self.audio_feature_map = nn.Sequential(
            nn.Linear(768, self.cond_latent_dim // 2),
            nn.LayerNorm(self.cond_latent_dim // 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # get per_frame feature map
        self.motion_branch = SequenceBranchMLP(
            input_nfeats=self.n_feats, 
            output_nfeats=self.n_feats, 
            input_motion_length=self.input_motion_length,
            cond_latent_dim=self.cond_latent_dim,
            latent_dim=self.input_latent_dim, 
            num_layers=self.num_layers, 
            dataset=self.dataset)
    
    def freeze_wav2vec(self):
        self.audio_encoder.freeze_encoder()
    
    def unfreeze_wav2vec(self):
        self.audio_encoder.unfreeze_encoder()

    def mask_cond_sparse(self, cond, force_mask=True):  # mask the condition for classifier-free guidance
        bs = cond.shape[0]
        dim = len(cond.shape)
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ) # 1-> use null_cond, 0-> use real cond
            if dim == 3:
                mask = mask.view(bs, 1, 1)
            elif dim == 5:
                mask = mask.view(bs, 1, 1, 1, 1)
            else:
                mask = mask.view(bs, 1) 
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
        mask = mask.view(bs, 1, 1)
        # 1-> use null_cond, 0-> use real cond
        return audio_emb * (1.0 - mask)

    def forward(self, x, timesteps, lmk_2d, audio_input, lmk_mask, force_mask=False, **kwargs):
        """
        Args:
            x: (b, n, c)
            image: (b, n, 3, 224, 224)
            lmk2d: (b, n, v, 2)
            lmk_mask: (bs, n, v)
            audio_input: (bs, l)
            img_mask: (bs, n, 224, 224)
            timesteps: [batch_size] (int)
        """

        # occlude corresponding visual input
        # image = image.clone() * (img_mask.unsqueeze(2))
        lmk_2d = lmk_2d[...,:2].clone() * (lmk_mask.unsqueeze(-1))

        bs, n = x.shape[:2]
        # image_features = self.image_encoder.features(image.view(bs*n, *image.shape[2:]))
        # image_features = nn.functional.adaptive_avg_pool2d(image_features, (1, 1)).squeeze(-1).squeeze(-1).view(bs, n, -1) # [bs, n, image_feature_dim]
        # image_cond = self.image_process(image_features)

        lmk_cond = self.lmk2d_process(lmk_2d.reshape(bs, n, -1))

        audio_cond = self.audio_encoder(audio_input, frame_num=n).last_hidden_state
        audio_cond = self.mask_audio_cond(
            self.audio_feature_map(audio_cond))

        cond = torch.cat([lmk_cond, audio_cond], dim=-1) # (bs, n, cond_latent_dim)
        
        cond_masked = self.mask_cond_sparse(cond, force_mask=force_mask)

        # diffMLP motion branch
        output = self.motion_branch(x, cond_masked, timesteps)
        
        return output