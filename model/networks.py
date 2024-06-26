# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
DType = int

###############################
############ Layers ###########
###############################


class MLPblock_Sequence(nn.Module):
    def __init__(self, cond_dim, dim, seq0, seq1, first=False, w_embed=True):
        super().__init__()

        self.w_embed = w_embed
        self.fc0 = nn.Conv1d(seq0, seq1, 1) # convolution across nframes dimension

        if self.w_embed:
            if first:
                self.conct = nn.Linear(dim + cond_dim, dim)
            else:
                self.conct = nn.Identity()
            self.emb_fc = nn.Linear(dim, dim)

        self.fc1 = nn.Linear(dim, dim)
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, inputs):

        if self.w_embed:
            x = inputs[0]
            embed = inputs[1]
            x = self.conct(x) + self.emb_fc(self.act(embed))
        else:
            x = inputs

        x_ = self.norm0(x)
        x_ = self.fc0(x_)
        x_ = self.act(x_)
        x = x + x_

        x_ = self.norm1(x)
        x_ = self.fc1(x_)
        x_ = self.act(x_)

        x = x + x_

        if self.w_embed:
            return x, embed
        else:
            return x

class MLPblock(nn.Module):
    def __init__(self, cond_dim, dim, dropout=0, first=False, w_embed=True):
        super().__init__()

        self.w_embed = w_embed

        if self.w_embed:
            if first:
                self.conct = nn.Linear(cond_dim + cond_dim, dim)
            else:
                self.conct = nn.Identity()
            self.emb_fc = nn.Linear(dim, dim)

        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):

        if self.w_embed:
            x = inputs[0]
            embed = inputs[1]
            x = self.conct(x) + self.emb_fc(self.act(embed))
        else:
            x = inputs

        x_ = self.norm(x)
        x_ = self.fc(x_)
        x_ = self.act(x_)
        x_ = self.dropout(x_)
        x = x + x_

        if self.w_embed:
            return x, embed
        else:
            return x

class BaseMLP(nn.Module):
    def __init__(self, cond_dim, dim, seq, num_layers, w_embed=True):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(
                MLPblock_Sequence(cond_dim, dim, seq, seq, first=i == 0 and w_embed, w_embed=w_embed)
            )

        self.mlps = nn.Sequential(*layers)

    def forward(self, x):
        """
        :args:
            x: [motion-input, ts embedding]
        """
        x = self.mlps(x)
        return x


###############################
########### Networks ##########
###############################


class DiffMLP(nn.Module):
    def __init__(self, cond_dim=256, latent_dim=512, seq=98, num_layers=12):
        """
        :args:
            num_layers: number of MLP blocks
            seq: input motion length
        """
        super(DiffMLP, self).__init__()

        self.motion_mlp = BaseMLP(cond_dim=cond_dim, dim=latent_dim, seq=seq, num_layers=num_layers)

    def forward(self, motion_input, embed):
        """
        :args:
            motion_input: concat of sparse signal & motion sequence
            embed: timestep embedding
        """

        motion_feats = self.motion_mlp([motion_input, embed])[0]

        return motion_feats

class DiffMLPSingleFrame(nn.Module):
    def __init__(self, cond_dim=256, latent_dim=512, num_layers=12, dropout=0, w_embed=True):
        """
        :args:
            num_layers: number of MLP blocks
        """
        super(DiffMLPSingleFrame, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                MLPblock(cond_dim, latent_dim, dropout=dropout,first=i == 0 and w_embed, w_embed=w_embed)
            )

        self.mlps = nn.Sequential(*layers)

    def forward(self, motion_input, embed):
        """
        :args:
            motion_input: concat of sparse signal & motion sequence
            embed: timestep embedding
        """

        motion_feats = self.mlps([motion_input, embed])[0]

        return motion_feats


class PureMLP(nn.Module):
    def __init__(
        self, latent_dim=512, seq=98, num_layers=12, input_dim=68*3, output_dim=133
    ):
        super(PureMLP, self).__init__()

        self.input_fc = nn.Linear(input_dim, latent_dim)
        self.motion_mlp = BaseMLP(
            dim=latent_dim, seq=seq, num_layers=num_layers, w_embed=False
        )
        self.output_fc = nn.Linear(latent_dim, output_dim)

    def forward(self, motion_input):

        motion_feats = self.input_fc(motion_input)
        motion_feats = self.motion_mlp(motion_feats)
        motion_feats = self.output_fc(motion_feats)

        return motion_feats

##############################################################
############ Transformer Decoder layers with FiLM ###########
##############################################################
class TransformerDecoderLayerFiLM(nn.Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and FiLM layer, feedforward network.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.
    """

    __constants__ = ['norm_first']

    def __init__(
            self, d_model, nhead, dim_feedforward = 2048, 
            dropout = 0.1, activation = nn.functional.relu,
            layer_norm_eps = 1e-5, batch_first = False, norm_first = False,
            bias = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayerFiLM, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=bias, **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 bias=bias, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=bias, **factory_kwargs)
        self.dropout1 = nn. Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # FiLM layer
        self.FiLM_activation = nn.functional.mish
        self.FiLM_linear1 = nn.Linear(d_model, d_model * 2)
        self.FiLM_linear2 = nn.Linear(d_model, d_model * 2)
        self.FiLM_linear3 = nn.Linear(d_model, d_model * 2)
        self.d_model = d_model

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = nn.modules.transformer._get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.functional.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt,
        memory,
        ts_emb,
        tgt_mask = None,
        memory_mask = None,
        tgt_key_padding_mask = None,
        memory_key_padding_mask = None,
        tgt_is_causal = False,
        memory_is_causal = False,
    ):
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = self._FiLM_block1(x, ts_emb)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = self._FiLM_block2(x, ts_emb)
            x = x + self._ff_block(self.norm3(x))
            x = self._FiLM_block3(x, ts_emb)
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self._FiLM_block1(x, ts_emb)
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self._FiLM_block2(x, ts_emb)
            x = self.norm3(x + self._ff_block(x))
            x = self._FiLM_block3(x, ts_emb)
        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
    
    def _FiLM_block1(self, x: Tensor, ts_emb: Tensor) -> Tensor:
        ts_scaler = self.FiLM_linear1(self.FiLM_activation(ts_emb))
        scale = ts_scaler[...,:self.d_model]
        shift = ts_scaler[...,self.d_model:]
        return (scale * x) + shift
    
    def _FiLM_block2(self, x: Tensor, ts_emb: Tensor) -> Tensor:
        ts_scaler = self.FiLM_linear2(self.FiLM_activation(ts_emb))
        scale = ts_scaler[...,:self.d_model]
        shift = ts_scaler[...,self.d_model:]
        return (scale * x) + shift

    def _FiLM_block3(self, x: Tensor, ts_emb: Tensor) -> Tensor:
        ts_scaler = self.FiLM_linear3(self.FiLM_activation(ts_emb))
        scale = ts_scaler[...,:self.d_model]
        shift = ts_scaler[...,self.d_model:]
        return (scale * x) + shift

class TransformerDecoderFiLM(nn.Module):
    """TransformerDecoder consiting of N decoder layers with FiLM layers to inject diffusion timesteps.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """

    __constants__ = ['norm']

    def __init__(
        self,
        decoder_layer,
        num_layers: int,
        norm: Optional[nn.Module] = None
    ) -> None:
        super(TransformerDecoderFiLM, self).__init__()
        self.layers = nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, ts_emb: Tensor, 
            tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
            memory_is_causal: bool = False) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            ts_emb: the time step embedding
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(output, memory, ts_emb, 
                         tgt_mask=tgt_mask,memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal,
                         memory_is_causal=memory_is_causal)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderFiLM(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers.

    Users can build the BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).
    """

    __constants__ = ['norm']

    def __init__(
        self,
        encoder_layer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = nn.modules.transformer._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ''
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first :
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (f"{enc_layer}.self_attn.batch_first was not True" +
                                          "(use batch_first for better inference performance)")
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
        elif encoder_layer.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn was passed bias=False"
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f"{enc_layer}.activation_relu_or_gelu was not True"
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        if enable_nested_tensor and why_not_sparsity_fast_path:
            self.use_nested_tensor = False


    def forward(
            self,
            src: Tensor,
            ts_emb: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = _canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=_none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = _canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask

        batch_first = first_layer.self_attn.batch_first


        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        for mod in self.layers:
            output = mod(output, ts_emb, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayerFiLM(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``

    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = nn.modules.transformer._get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

        # FiLM layer
        self.FiLM_activation = nn.functional.mish
        self.FiLM_linear1 = nn.Linear(d_model, d_model * 2)
        self.FiLM_linear2 = nn.Linear(d_model, d_model * 2)
        self.d_model = d_model

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(
            self,
            src: Tensor,
            ts_emb: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
        """

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = self._FiLM_block1(x, ts_emb)
            x = x + self._ff_block(self.norm2(x))
            x = self._FiLM_block2(x, ts_emb)
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self._FiLM_block1(x, ts_emb)
            x = self.norm2(x + self._ff_block(x))
            x = self._FiLM_block2(x, ts_emb)

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    def _FiLM_block1(self, x: Tensor, ts_emb: Tensor) -> Tensor:
        ts_scaler = self.FiLM_linear1(self.FiLM_activation(ts_emb))
        scale = ts_scaler[...,:self.d_model]
        shift = ts_scaler[...,self.d_model:]
        return (scale * x) + shift
    
    def _FiLM_block2(self, x: Tensor, ts_emb: Tensor) -> Tensor:
        ts_scaler = self.FiLM_linear2(self.FiLM_activation(ts_emb))
        scale = ts_scaler[...,:self.d_model]
        shift = ts_scaler[...,self.d_model:]
        return (scale * x) + shift

def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]

def _generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )

def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal

def _canonical_mask(
        mask: Optional[Tensor],
        mask_name: str,
        other_type: Optional[DType],
        other_name: str,
        target_type: DType,
        check_other: bool = True,
) -> Optional[Tensor]:

    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported")
            
        if not _mask_is_float:
            mask = (
                torch.zeros_like(mask, dtype=target_type)
                .masked_fill_(mask, float("-inf"))
            )
    return mask

def _none_or_dtype(input: Optional[Tensor]) -> Optional[DType]:
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")