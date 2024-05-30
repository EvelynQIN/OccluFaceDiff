import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
from model.denoising_model import neighborhood_mask

class VideoEmotionClassifier(nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int, 
                 latent_dim: int = 128, 
                 heads: int = 4, 
                 layers: int = 4, 
                 ff_size: int = 256,
                 max_pool: bool = True, 
                 dropout: float = 0.0) -> None:
        """
        :param num_classes: Number of classes.
        :param  latent: Embedding dimension
        :param heads: Number of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        :param dropout: dropout value to be applied between layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.dropout = dropout 
        self.max_pool = max_pool 
        self.ff_size = ff_size
        self.heads = heads
        self.layers = layers

        self.input_process = InputProcess(input_dim, latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.output_process = nn.Linear(self.latent_dim, self.num_classes)

        src_mask = neighborhood_mask(target_size=1000, num_heads = self.heads, bias_step=25, symm=True)
        self.register_buffer('src_mask', src_mask)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.latent_dim, dim_feedforward=self.ff_size,
            nhead=self.heads, dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.layers)

    def forward(self, x) -> torch.Tensor:
        """
        Function that encodes the source sequence.
        :param x: (bs, t, input_dim)
        
        :returns:
            -  output
                 (bs, num_classes)
        """
        bs, T = x.shape[:2]

        src_mask = self.src_mask[:, :T, :T].clone().detach().to(device=x.device)    # (num_heads, seqlen, seqlen)
        src_mask = src_mask.repeat(bs, 1, 1)
        x = self.input_process(x)
        x_seq = self.sequence_pos_encoder(x)

        encoder_output = self.transformer_encoder(x_seq, mask = src_mask)   # (T, bs, latent_dim)

        # pooling over the time dimensino
        if self.max_pool:
            output = encoder_output.max(dim=0).values
        else:
            output = encoder_output.mean(dim=0)  # (bs, latent_dim)
        
        output = self.output_process(output)    # (bs, num_classses)
        
        output = F.log_softmax(output, dim=1)
        return output
    
    def encode(self, x):
        bs, T = x.shape[:2]

        src_mask = self.src_mask[:, :T, :T].clone().detach().to(device=x.device)    # (num_heads, seqlen, seqlen)
        src_mask = src_mask.repeat(bs, 1, 1)
        x = self.input_process(x)
        x_seq = self.sequence_pos_encoder(x)

        encoder_output = self.transformer_encoder(x_seq, mask = src_mask)   # (T, bs, latent_dim)

        # pooling over the time dimension
        if self.max_pool:
            output = encoder_output.max(dim=0).values
        else:
            output = encoder_output.mean(dim=0)  # (bs, latent_dim)
        return output

    def predict(self, x):

        bs, T = x.shape[:2]

        src_mask = self.src_mask[:, :T, :T].clone().detach().to(device=x.device)    # (num_heads, seqlen, seqlen)
        src_mask = src_mask.repeat(bs, 1, 1)
        x = self.input_process(x)
        x_seq = self.sequence_pos_encoder(x)

        encoder_output = self.transformer_encoder(x_seq, mask = src_mask)   # (T, bs, latent_dim)

        # pooling over the time dimension
        if self.max_pool:
            output = encoder_output.max(dim=0).values
        else:
            output = encoder_output.mean(dim=0)  # (bs, latent_dim)
        
        output = self.output_process(output)    # (bs, num_classses)
        output = F.log_softmax(output, dim=1)
        return torch.argmax(output, dim=1)  # (bs, class_label)

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.inputEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # bs, nframes, motion_nfeats = x.shape
        x = x.permute(1, 0, 2)  # (nframes, bs, motion_nfeats)
        x = self.inputEmbedding(x)
        return x

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