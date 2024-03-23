import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  
from utils import dist_util
from model.mica import MICA 
import math

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

class SingleBranchDecoder(nn.Module):
    def __init__(
        self, 
        arch,
        input_nfeats,
        output_nfeats,
        latent_dim=256, 
        ff_size=1024, 
        num_dec_layers=2,
        num_heads=4, 
        dropout=0.1,
        activation="gelu", 
        dataset='FaMoS',
        **kargs):
        super().__init__()

        self.input_feats = input_nfeats
        self.output_feats = output_nfeats
        self.dataset = dataset

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=activation)
        
        self.seqTransDecoder = nn.TransformerDecoder(
            seqTransDecoderLayer,
            num_layers=self.num_dec_layers)

        self.output_process = OutputProcess(self.output_feats, self.latent_dim)

    def forward(self, x, lmk_emb, **kwargs):
        """
        x: [batch_size, nframes, latent_dim] 
        timesteps: [batch_size] (int)
        sparse_emb: [batch_size, nframes, sparse_dim]
        """

        # cross attention of the sparse cond & motion_output
        xseq = self.sequence_pos_encoder(x)
        decoder_output = self.seqTransDecoder(tgt=xseq, memory=lmk_emb) # [seqlen, bs, d] 

        output = self.output_process(decoder_output)  # [bs, nframes, nfeats]
        return output

class MultiBranchTransformer(nn.Module):
    def __init__(
        self,
        arch,
        nfeats,
        sparse_dim=68*3,
        latent_dim=256, 
        ff_size=1024, 
        num_enc_layers=2, 
        num_dec_layers=2,
        num_heads=4, 
        dropout=0.1,
        activation="gelu", 
        dataset='FaMoS',
        **kargs):
        super().__init__()

        self.input_feats = nfeats
        self.lmk_dim = sparse_dim
        self.dataset = dataset

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.lmk_process = Lmk3dProcess(self.lmk_dim, self.latent_dim)
        self.input_process = InputProcess(self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
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

        self.decoder_pose6d = SingleBranchDecoder(
            arch=self.arch,
            input_nfeats=self.input_feats,
            output_nfeats=4*6,
            latent_dim=self.latent_dim, 
            ff_size=self.ff_size, 
            num_dec_layers=self.num_dec_layers,
            num_heads=self.num_heads, 
            dropout=self.dropout,
            activation=self.activation, 
            dataset=self.dataset,
        )

        self.decoder_expr = SingleBranchDecoder(
            arch=self.arch,
            input_nfeats=self.input_feats,
            output_nfeats=100,
            latent_dim=self.latent_dim, 
            ff_size=self.ff_size, 
            num_dec_layers=self.num_dec_layers,
            num_heads=self.num_heads, 
            dropout=self.dropout,
            activation=self.activation, 
            dataset=self.dataset,
        )

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
        ) # [seqlen, 1, d]
        
        # put ts_emb & lmk_emb to transformer encoder layer for self attention
        lmkseq = self.sequence_pos_encoder(lmk_emb)  # [seqlen, bs, d]
        lmk_encoding = self.seqTransEncoder(lmkseq)  # [seqlen, bs, d]

        # cross attention of the sparse cond & motion_output
        x = self.input_process(x)
        x = x + ts_emb
        x_pose6d = self.decoder_pose6d(x, lmk_encoding)
        x_expr = self.decoder_expr(x, lmk_encoding)

        output = torch.cat([x_pose6d, x_expr], dim=-1)  # [bs, nframes, nfeats]
        return output

class FaceTransformer(nn.Module):
    def __init__(
        self,
        arch,
        nfeats,
        lmk3d_dim=68*3,
        lmk2d_dim=68*2,
        latent_dim=256, 
        ff_size=1024, 
        num_enc_layers=2, 
        num_dec_layers=2,
        num_heads=4, 
        dropout=0.1,
        activation="gelu", 
        dataset='FaMoS',
        use_mask=True,
        load_mica=False,
        n_shape = 100,
        n_exp = 50,
        n_pose = 5 * 6,
        n_trans = 3,
        **kwargs):
        super().__init__()

        self.tag = 'FaceTransformer'
        self.input_feats = nfeats
        self.lmk3d_dim = lmk3d_dim
        self.lmk2d_dim = lmk2d_dim 
        self.dataset = dataset
        self.use_mask = use_mask
        if self.use_mask:
            print(f"[{self.tag}] Using alibi mask for decoder.")
        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.cond_mask_prob = kwargs.get('cond_mask_prob', 0.)
        self.arch = arch
        
        self.nshape = n_shape
        self.nexp = n_exp
        self.npose = n_pose
        self.ntrans = n_trans
        
        ### layers
        
        # process the condition 
        self.lmk3d_process = InputProcess(self.lmk3d_dim, self.latent_dim)
        self.lmk2d_process = InputProcess(self.lmk2d_dim, self.latent_dim)
        self.mica_process = nn.Sequential(
            nn.Linear(self.nshape, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.cond_process = nn.Sequential(
            nn.Linear(self.latent_dim * 3, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.input_process = InputProcess(self.input_feats, self.latent_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        target_mask = neighborhood_mask(target_size=2000, num_heads = self.num_heads, bias_step=20)
        self.register_buffer('tgt_mask', target_mask)
        
        # load pretrained mica model
        if load_mica:
            assert "mica" in kwargs, "Pretrained MICA not passed into the kwargs!"
            self.mica = kwargs['mica']
        
        print(f"[{self.tag}] Using transformer as backbone.")
        self.transformer = nn.Transformer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            num_encoder_layers=self.num_enc_layers,
            num_decoder_layers=self.num_dec_layers,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            norm_first=False
        )
        
        self.outputprocess_shape = ShapeOutput()
        self.outputprocess_motion = MotionOutput(output_feats=self.input_feats, latent_dim=self.latent_dim)

    def mask_cond_lmk(self, cond, force_mask=False):
        bs = cond.shape[0]
        ndim = len(cond.shape)
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            )
            if ndim == 3:
                mask = mask.view(bs, 1, 1)
            elif ndim == 2:
                mask = mask.view(bs, 1)
            else: 
                raise ValueError( f"Invalid dimension for conditioning mask {cond.shape}")
              # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond
    
    def forward(self, x, timesteps, lmk_3d, lmk_2d, mica_shape, occlusion_mask, force_mask=False, **kwargs):
        """
        x: [batch_size, nframes, nfeats] 
        timesteps: [batch_size] (int)
        sparse_emb: [batch_size, nframes, nlmks, 2/3]
        occlusion_mask: [batch_size, nframes, nlmks]
        """
        bs, n = lmk_3d.shape[:2]
        ts_emb = self.embed_timestep(timesteps)  # [1, bs, d]

        # mask the occluded lmk to be 0
        # print(lmk_3d.shape)
        # print(occlusion_mask.shape)
        occlusion = (1-occlusion_mask).unsqueeze(-1)
        lmk_3d = (lmk_3d * occlusion).reshape(bs, n, -1)
        lmk_2d = (lmk_2d * occlusion).reshape(bs, n, -1)

        lmk3d_emb = self.lmk3d_process(
            self.mask_cond_lmk(lmk_3d, force_mask=force_mask)
        ) # [seqlen, bs, d]
        
        lmk2d_emb = self.lmk2d_process(
            self.mask_cond_lmk(lmk_2d, force_mask=force_mask)
        ) # [seqlen, bs, d]
        
        shape_mica_emb = self.mica_process(
            self.mask_cond_lmk(mica_shape, force_mask=force_mask)
        )

        cond_emb = self.cond_process(
            torch.cat([lmk3d_emb, lmk2d_emb, shape_mica_emb[None,:,:].repeat(n, 1, 1)], dim=-1))
        
        # cond_emb = lmk3d_emb + lmk2d_emb + shape_mica_emb[None,:, :]
        
        condseq = self.sequence_pos_encoder(cond_emb)  # [seqlen, bs, d]
        
        tgt_mask=None
        if self.use_mask:
            T = x.shape[1]
            tgt_mask = self.tgt_mask[:, :T, :T].clone().detach().to(device=x.device)    # (num_heads, seqlen, seqlen)

        # cross attention of the sparse cond & motion_output
        x = self.input_process(x)
        x = x + ts_emb   # broadcast add, [seqlen, bs, d]
        xseq = self.sequence_pos_encoder(x)
        
        decoder_output = self.transformer(condseq, xseq, tgt_mask=tgt_mask) # [seqlen, bs, d]
        output = self.outputprocess_motion(decoder_output)  # [bs, seqlen, input_nfeats]
        shape_seq = output[:, :, :self.nshape]
        shape_agg = self.outputprocess_shape(shape_seq).unsqueeze(1).repeat(1, output.shape[1], 1)
        output = torch.cat([shape_agg, output[:, :, self.nshape:]],dim=-1)
        return output

    def predict(self, x, timesteps, lmk_3d, lmk_2d, img_arr, occlusion_mask, force_mask=False, return_mica=False, **kwargs):
        """
        x: [batch_size, nframes, nfeats] 
        timesteps: [batch_size] (int)
        sparse_emb: [batch_size, nframes, nlmks, 2/3]
        occlusion_mask: [batch_size, nframes, nlmks]
        """
        bs, n = lmk_3d.shape[:2]
        ts_emb = self.embed_timestep(timesteps)  # [1, bs, d]

        # mask the occluded lmk to be 0
        # print(lmk_3d.shape)
        # print(occlusion_mask.shape)
        occlusion = (1-occlusion_mask).unsqueeze(-1)
        lmk_3d = (lmk_3d * occlusion).reshape(bs, n, -1)
        lmk_2d = (lmk_2d * occlusion).reshape(bs, n, -1)

        lmk3d_emb = self.lmk3d_process(
            self.mask_cond_lmk(lmk_3d, force_mask=force_mask)
        ) # [seqlen, bs, d]
        
        lmk2d_emb = self.lmk2d_process(
            self.mask_cond_lmk(lmk_2d, force_mask=force_mask)
        ) # [seqlen, bs, d]
        
        shape_mica = self.mica(img_arr)[:, :self.nshape] # [bs, nshape]
        shape_mica_emb = self.mica_process(
            self.mask_cond_lmk(shape_mica, force_mask=force_mask)
        )
        
        cond_emb = lmk3d_emb + lmk2d_emb + shape_mica_emb[None,:, :]
        
        condseq = self.sequence_pos_encoder(cond_emb)  # [seqlen, bs, d]
        
        tgt_mask=None
        if self.use_mask:
            T = x.shape[1]
            tgt_mask = self.tgt_mask[:, :T, :T].clone().detach().to(device=x.device)    # (num_heads, seqlen, seqlen)

        # cross attention of the sparse cond & motion_output
        x = self.input_process(x)
        x = x + ts_emb   # broadcast add, [seqlen, bs, d]
        xseq = self.sequence_pos_encoder(x)
        
        decoder_output = self.transformer(condseq, xseq, tgt_mask=tgt_mask) # [seqlen, bs, d]
        output = self.outputprocess_motion(decoder_output)  # [bs, seqlen, input_nfeats]
        shape_seq = output[:, :, :self.nshape]
        shape_agg = self.outputprocess_shape(shape_seq).unsqueeze(1).repeat(1, output.shape[1], 1)
        output = torch.cat([shape_agg, output[:, :, self.nshape:]],dim=-1)
        if return_mica:
            return output, shape_mica
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

    
class TransformerEncoder_TS(nn.Module):
    """Inject diffusion timestep to each attention layer
    """
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
        
        category_mask = torch.cat(
                    [torch.ones(4*6), torch.ones(100) * -1]
                )
        self.register_buffer('input_category_mask', category_mask)
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.lmk_process = Lmk3dProcess(self.lmk_dim, self.latent_dim)
        self.input_process = InputProcess(self.input_category_mask, self.input_feats, self.latent_dim, dist_util.dev())
        self.merge_input_with_cond = nn.Linear(self.latent_dim*2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.pose_latent_dim = (self.latent_dim // self.num_heads) * 2 
        self.expr_latent_dim = self.latent_dim - self.pose_latent_dim 
        
        self.TransEncoder = MultiEncoderLayers_TS(
            latent_dim=self.latent_dim, 
            ff_size=self.ff_size, 
            num_heads=self.num_heads, 
            dropout=self.dropout,
            activation=self.activation,
            num_layers=self.num_enc_layers
        )

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
        x_seq = self.sequence_pos_encoder(x)
        x = [x_seq, ts_emb]
        encoder_output = self.TransEncoder(x)    # [seqlen, bs, d]
        output_pose = self.outputprocess_pose(encoder_output[:, :, :self.pose_latent_dim])  # [bs, seqlen,  4*6]
        output_expr = self.outputprocess_expr(encoder_output[:, :, self.pose_latent_dim:])   # [bs, seqlen, 100]
        output = torch.cat([output_pose, output_expr], dim=-1)  # [bs, seqlen, input_nfeats]
        return output

class GRUDecoder(nn.Module):
    def __init__(
        self,
        arch,
        nfeats,
        sparse_dim=68*3,
        latent_dim=256, 
        dropout=0.1,
        activation="gelu", 
        dataset='FaMoS',
        **kargs):
        super().__init__()

        self.input_feats = nfeats
        self.lmk_dim = sparse_dim
        self.dataset = dataset

        self.latent_dim = latent_dim

        self.dropout = dropout
        self.activation = activation
        
        # category_mask = torch.cat(
        #             [torch.ones(4*6), torch.ones(100) * -1]
        #         )
        # self.register_buffer('input_category_mask', category_mask)
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.lmk_process = Lmk3dProcess(self.lmk_dim, self.latent_dim)
        self.input_process = InputProcess(self.input_feats, self.latent_dim)
        self.merge_input_with_cond = nn.Sequential(
            nn.Linear(self.latent_dim*3, self.latent_dim*2),
            nn.ReLU(),
            nn.Linear(self.latent_dim*2, self.latent_dim)
        )
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        self.pose_latent_dim = 64
        self.expr_latent_dim = 256
        self.num_layers = 4
        
        print("Using GRU as backbone...")
        
        self.gru_pose = nn.GRU(self.latent_dim, self.pose_latent_dim, num_layers=self.num_layers, dropout=self.dropout)
        self.gru_expr = nn.GRU(self.latent_dim, self.expr_latent_dim, num_layers=self.num_layers*2, dropout=self.dropout)
        
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
        bs, nframes, nfeats = x.shape
        ts_emb = self.embed_timestep(timesteps)  # [1, bs, d]
        ts_gru = ts_emb.repeat(nframes, 1, 1)   # [seqlen, bs, d]
        
        lmk_emb = self.lmk_process(
            self.mask_cond_lmk(sparse_emb, force_mask=force_mask)
        ) # [seqlen, 1, d]
        
        # self attention of the sparse cond & motion_output
        x = self.input_process(x)
        x = torch.cat([ts_gru, lmk_emb, x], dim=-1)  # [seqlen, bs, 3d]
        x = self.merge_input_with_cond(x)   # [seqlen, bs, d]
    
        x_seq = self.sequence_pos_encoder(x)
        gru_pose, _ = self.gru_pose(x_seq)    # [seqlen, bs, d]
        gru_expr, _ = self.gru_expr(x_seq)
        output_pose = self.outputprocess_pose(gru_pose)  # [bs, seqlen,  4*6]
        output_expr = self.outputprocess_expr(gru_expr)   # [bs, seqlen, 100]
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
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_dim, self.output_feats)
        )
        

    def forward(self, output):
        output = self.fc(output)  # [nframes, bs, nfeats]
        output = output.permute(1, 0, 2)  # [bs, nframes, nfeats]
        return output

class ShapeOutput(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # agg over the the frames

    def forward(self, shape_seq):
        """

        Args:
            shape_seq: (bs, n, n_shape)

        Returns:
            shape_agg: (bs, n_shape)
        """
        shape_seq = shape_seq.transpose(1, 2) # [bs, nshape, n]
        shape_agg = self.avg_pool(shape_seq).squeeze(2)    # [bs, nshape]
        return shape_agg