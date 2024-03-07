# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import torch.nn as nn


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
