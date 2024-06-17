# Reference from https://github.com/radekd91/inferno/blob/75f8f76352ad4fe9ee401c6e845228810eb7f459/inferno/models/temporal/motion_prior/L2lMotionPrior.py#L218
import torch 
import torch.nn as nn
import math

# downsample the fps of the original video sequence
def create_squasher(input_dim, hidden_dim, quant_factor):
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

def init_alibi_biased_mask_future(num_heads, max_seq_len):
    """
    Returns a mask for the self-attention layer. The mask is initialized according to the ALiBi paper but 
    not with the future masked out.
    The diagonal is filled with 0 and the lower triangle is filled with a linear function of the position. 
    The upper triangle is filled symmetrically with the lower triangle.
    That lowers the attention to the past and the future (the number gets lower the further away from the diagonal it is).
    """
    period = 1
    slopes = torch.Tensor(get_slopes(num_heads))
    bias = torch.arange(start=0, end=max_seq_len).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    # mask = alibi - torch.flip(alibi, [1, 2])
    mask = alibi + torch.flip(alibi, [1, 2])
    return mask

class L2lEncoder(nn.Module): 
    """
    Inspired by by the encoder from Learning to Listen.
    """

    def __init__(self, cfg, sizes):
        super().__init__()
        self.config = cfg
        self.sizes = sizes
        size = 103
        dim = self.config.feature_dim

        # self.squasher = nn.Sequential(*layers) 
        self.squasher = create_squasher(size, dim, sizes.quant_factor)
        # the purpose of the squasher is to reduce the FPS of the input sequence

        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=cfg.feature_dim, 
                    nhead=cfg.nhead, 
                    dim_feedforward=cfg.intermediate_size, 
                    activation=cfg.activation, # L2L paper uses gelu
                    dropout=cfg.dropout, 
                    batch_first=True
        )
        self.encoder_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        
        # None from ckpt
        self.encoder_pos_embedding = None

        # alibi future bias mask
        # attention_mask_cfg = munchify(OmegaConf.to_container(attention_mask_cfg))
        self.attention_mask = init_alibi_biased_mask_future(num_heads = cfg.nhead, max_seq_len = 600)

        self.encoder_linear_embedding = nn.Linear(
            self.config.feature_dim,
            self.config.feature_dim
        )

    def get_input_dim(self):
        return 103
        # return input_dim

    def forward(self, inputs, **kwargs):
        # ## downsample into path-wise length seq before passing into transformer
        # input is batch first: B, T, D but the convolution expects B, D, T
        # so we need to permute back and forth
        inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1)
        
        encoded_features = self.encoder_linear_embedding(inputs)

        # add positional encoding
        if self.encoder_pos_embedding is not None:
            encoded_features = self.encoder_pos_embedding(encoded_features)
        
        # add attention mask (if any)
        B, T = encoded_features.shape[:2]
        mask = None
        if self.attention_mask is not None:
            mask = self.attention_mask[:, :T, :T].clone() \
                .detach().to(device=encoded_features.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)
    
        encoded_features = self.encoder_transformer(encoded_features, mask=mask)
        
        return encoded_features

    def bottleneck_dim(self):
        return self.config.feature_dim
    
    def latent_temporal_factor(self): 
        return 2 ** self.sizes.quant_factor

    def quant_factor(self): 
        return self.sizes.quant_factor

class L2lEncoderWithGaussianHead(L2lEncoder): 

    def __init__(self, cfg, sizes) -> None:
        super().__init__(cfg, sizes) 
        self.mean = nn.Linear(self.config.feature_dim, self.config.feature_dim)
        self.logvar = nn.Linear(self.config.feature_dim, self.config.feature_dim)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(self.logvar.weight.device)
        self.N.scale = self.N.scale.to(self.logvar.weight.device)

    # def to(self, *args, device=None, **kwargs):
    #     super().to(*args, device=None, **kwargs)
    #     self.N.loc.to(device)
    #     self.N.scale.to(device)
    #     self.mean = self.mean.to(device)
    #     self.logvar = self.logvar.to(device)
    #     return self
    
    def forward(self, inputs, **kwargs):
        if self.N.loc.device != self.logvar.weight.device:
            self.N.loc = self.N.loc.to(self.logvar.weight.device)
            self.N.scale = self.N.scale.to(self.logvar.weight.device)
        encoded_feature = super().forward(inputs, **kwargs)

        B,T = encoded_feature.shape[:2]
        encoded_feature = encoded_feature.reshape(B*T, -1)
        mean = self.mean(encoded_feature)
        logvar = self.logvar(encoded_feature)
        std = torch.exp(0.5*logvar)
        # eps = self.N.sample(std.size())
        z = mean + std * self.N.sample(mean.shape)
        z = z.reshape(B,T,-1)

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim = 1), dim = 0)
        # batch["kl_divergence"] = kld_loss
        # batch[output_key] = z
        # batch[output_key + "_mean"] = mean.reshape(B,T,-1)
        # batch[output_key + "_logvar"] = logvar.reshape(B,T,-1)
        # batch[output_key + "_std"] = std.reshape(B,T,-1)
        
        return z

class L2lDecoder(nn.Module): 

    def __init__(self, cfg, sizes, out_dim): 
        super().__init__()
        is_audio = False
        self.cfg = cfg
        size = self.cfg.feature_dim
        dim = self.cfg.feature_dim
        self.expander = nn.ModuleList()
        self.expander.append(nn.Sequential(
                    nn.ConvTranspose1d(size,dim,5,stride=2,padding=2,
                                        output_padding=1,
                                        # padding_mode='replicate' # crashes, only zeros padding mode allowed
                                        padding_mode='zeros' 
                                        ),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm1d(dim)))
        num_layers = sizes.quant_factor + 2 \
            if is_audio else sizes.quant_factor
        # TODO: check if we need to keep the sequence length fixed
        seq_len = sizes.sequence_length*4 \
            if is_audio else sizes.sequence_length
        for _ in range(1, num_layers):
            self.expander.append(nn.Sequential(
                                nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                        padding_mode='replicate'),
                                nn.LeakyReLU(0.2, True),
                                nn.BatchNorm1d(dim),
                                ))
        decoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=cfg.feature_dim, 
                    nhead=cfg.nhead, 
                    dim_feedforward=cfg.intermediate_size, 
                    activation=cfg.activation, # L2L paper uses gelu
                    dropout=cfg.dropout, 
                    batch_first=True
        )
        self.decoder_transformer = torch.nn.TransformerEncoder(decoder_layer, num_layers=cfg.num_layers)

        

        self.decoder_pos_embedding = None
        self.attention_mask = init_alibi_biased_mask_future(num_heads = cfg.nhead, max_seq_len = 600)


        self.decoder_linear_embedding = nn.Linear(
            self.cfg.feature_dim,
            self.cfg.feature_dim
            )

        conv_smooth_layer = cfg.get('conv_smooth_layer', 'l2l_default')

        if conv_smooth_layer == 'l2l_default':
            self.cross_smooth_layer = nn.Conv1d(
                    cfg.feature_dim,
                    out_dim, 5, 
                    padding=2
                )
        elif conv_smooth_layer is False:
            self.cross_smooth_layer = None
        else:
            raise ValueError(f'conv_smooth_layer value {conv_smooth_layer} not supported')

        if cfg.get('post_transformer_proj', None):
            lin_out_dim = cfg.feature_dim if self.cross_smooth_layer is not None else out_dim
            self.post_transformer_linear = nn.Linear(
                cfg.feature_dim,
                lin_out_dim,
            )
            if cfg.post_transformer_proj.init == "zeros":
                torch.nn.init.zeros_(self.post_transformer_linear.weight)
                torch.nn.init.zeros_(self.post_transformer_linear.bias)
        else:
            self.post_transformer_linear = None


        if cfg.get('post_conv_proj', None):
            self.post_conv_proj = nn.Linear(
                out_dim,
                out_dim,
            )
            if cfg.post_conv_proj.init == "zeros":
                torch.nn.init.zeros_(self.post_conv_proj.weight)
                torch.nn.init.zeros_(self.post_conv_proj.bias)
        else:
            self.post_conv_proj = None

        # initialize the last layer of the decoder to zero 
        if cfg.get('last_layer_init', None) == "zeros":
            torch.nn.init.zeros_(self.decoder_transformer.layers[-1].linear2.weight)
            torch.nn.init.zeros_(self.decoder_transformer.layers[-1].linear2.bias)

    def forward(self, inputs, **kwargs):
        # dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        ## upsample to the original length of the sequence before passing into transformer
        for i, module in enumerate(self.expander):
            # input is batch first: B, T, D but the convolution expects B, D, T
            # so we need to permute back and forth
            inputs = module(inputs.permute(0,2,1)).permute(0,2,1)
            if i > 0:
                inputs = inputs.repeat_interleave(2, dim=1)
        
        decoded_features = self.decoder_linear_embedding(inputs)

        # add positional encoding
        if self.decoder_pos_embedding is not None:
            decoded_features = self.decoder_pos_embedding(decoded_features)
        
        # add attention mask bias (if any)
        B,T = decoded_features.shape[:2]
        mask = None
        if self.attention_mask is not None:
            mask = self.attention_mask[:, :T, :T].clone() \
                .detach().to(device=decoded_features.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)

        decoded_features = self.decoder_transformer(decoded_features, mask=mask)
        decoded_reconstruction = decoded_features
        if self.post_transformer_linear is not None:
            decoded_reconstruction = self.post_transformer_linear(decoded_reconstruction)
        if self.cross_smooth_layer is not None:
            decoded_reconstruction = self.cross_smooth_layer(decoded_reconstruction.permute(0,2,1)).permute(0,2,1)
        if self.post_conv_proj is not None:
            decoded_reconstruction = self.post_conv_proj(decoded_reconstruction)
        return decoded_reconstruction

class L2lVqVae(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.motion_encoder = L2lEncoderWithGaussianHead(cfg.model.sequence_encoder, cfg.model.sizes)
        self.motion_decoder = L2lDecoder(cfg.model.sequence_decoder, cfg.model.sizes, self.motion_encoder.get_input_dim())
    
    def load_model_from_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.load_state_dict(checkpoint['state_dict'], strict=False)
    
    def freeze_model(self):
        self.motion_encoder.requires_grad_(False)
        self.motion_encoder.eval()

        self.motion_decoder.requires_grad_(False)
        self.motion_decoder.eval()
        print(f"[FLINTv2] Frozen.")