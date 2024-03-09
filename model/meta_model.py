import numpy as np
import torch
import torch.nn as nn
from model.networks import DiffMLP, DiffMLPSingleFrame
from utils import dist_util
from model.mica import MICA

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

class ShapeBranchMLP(nn.Module):
    def __init__(
        self,
        shape_dim=300, 
        cond_dim=300,
        latent_dim=256,
        num_layers=8,
        dropout=0.1,
        dataset="FaMoS",
        **kargs
    ):
        super().__init__()

        self.dataset = dataset

        self.shape_dim = shape_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embed_timestep = TimestepEmbeding(self.latent_dim)
        
        self.mlp = DiffMLPSingleFrame(
            self.cond_dim, self.latent_dim, num_layers=num_layers, dropout=self.dropout)

        self.output_process = nn.Linear(self.latent_dim, self.shape_dim)

    def forward(self, x, cond_emb, ts):
        """
        Args:
            x: [batch_size, nfeats], denoted x_t 
            cond_emb: [batch_size, cond_latent_dim]
            ts: [batch_size,] 
        """

        # Pass the input to a FC
        ts_emb = self.embed_timestep(ts).squeeze(1)

        # Concat the sparse feature with the input
        x = torch.cat((cond_emb, x), axis=-1) # [bs, 2 x latent_dim]
        output = self.mlp(x, ts_emb)

        # Pass the output to a FC and reshape the output
        output = self.output_process(output)
        return output

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

    def forward(self, x, sparse_emb, ts):
        """
        Args:
            x: [batch_size, nframes, nfeats], denoted x_t in the paper
            sparse_emb: [batch_size, nframes, cond_latent_dim], the sparse features
            ts: [batch_size,] 
        """

        # Pass the input to a FC
        x = self.input_process(x)
        ts_emb = self.embed_timestep(ts)

        # Concat the sparse feature with the input
        x = torch.cat((sparse_emb, x), axis=-1) # [bs, 2 x latent_dim]
        output = self.mlp(x, ts_emb)

        # Pass the output to a FC and reshape the output
        output = self.output_process(output)
        return output
   
class MultiBranchMLP(nn.Module):
    def __init__(
        self,
        mica_args,
        nfeats, # input feature dim
        lmk3d_dim=68*3,
        lmk2d_dim=68*2,
        cond_latent_dim=256,
        shape_num_layers=4,
        shape_latent_dim=512,
        motion_latent_dim=128,
        motion_num_layers=1,
        trans_num_layers=4,
        trans_latent_dim=128,
        dropout=0.1,
        dataset="FaMoS",
        **kargs,
    ):
        super().__init__()

        self.dataset = dataset
        self.mica_args = mica_args
        self.shape_dim = 300
        self.exp_dim = 100
        self.pose_dim = 5 * 6
        self.trans_dim = 3
        self.n_feats = nfeats 

        self.lmk3d_dim = lmk3d_dim
        self.lmk2d_dim = lmk2d_dim
        
        self.dropout = dropout  
        self.cond_latent_dim = cond_latent_dim

        self.shape_num_layers = shape_num_layers 
        self.shape_latent_dim = shape_latent_dim 

        self.motion_num_layers = motion_num_layers 
        self.motion_latent_dim = motion_latent_dim 

        self.trans_num_layers = trans_num_layers 
        self.trans_latent_dim = trans_latent_dim 

        self.input_motion_length = kargs.get("input_motion_length")
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)

        ### layers

        # load pretrained mica model
        self.mica = MICA(self.mica_args)
        
        # process condition
        self.lmk3d_process = nn.Linear(self.lmk3d_dim, self.cond_latent_dim)
        self.pred_shape_process = nn.Linear(self.shape_dim, self.cond_latent_dim)

        self.mica_shape_process = nn.Linear(self.shape_dim, self.shape_dim)

        self.lmk2d_process = nn.Linear(self.lmk2d_dim, self.cond_latent_dim)
        self.pred_motion_process = nn.Linear(self.exp_dim+self.pose_dim, self.cond_latent_dim)

        # pred shape
        self.shape_branch = ShapeBranchMLP(
            shape_dim=self.shape_dim, 
            cond_dim=self.shape_dim,    # pure mica shape code
            latent_dim=self.shape_latent_dim,
            num_layers=self.shape_num_layers,
            dropout=self.dropout,
            dataset=self.dataset,
            **kargs)

        # pred expression + pose
        self.motion_nfeats = self.exp_dim+self.pose_dim
        self.motion_branch = SequenceBranchMLP(
            input_nfeats=self.motion_nfeats, 
            output_nfeats=self.motion_nfeats, 
            input_motion_length=self.input_motion_length,
            cond_latent_dim=self.cond_latent_dim,
            latent_dim=self.motion_latent_dim, 
            num_layers=self.motion_num_layers, 
            dataset=self.dataset)
        
        # pred trans 
        self.trans_branch = SequenceBranchMLP(
            input_nfeats=self.trans_dim, 
            output_nfeats=self.trans_dim, 
            input_motion_length=self.input_motion_length,
            cond_latent_dim=self.cond_latent_dim,
            latent_dim=self.trans_latent_dim, 
            num_layers=self.trans_num_layers, 
            dataset=self.dataset)

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
        
    def get_model_parameters(self, module_list):
        vars_net = []
        for _sub_module in module_list:
            vars_net.extend([var[1] for var in _sub_module.named_parameters()])
        return vars_net
    
    # def freeze_branch_params(self, branch_name):
    #     if branch_name == "pose":
    #         module_list = [self.pose6d_branch]
    #     elif branch_name == "expr":
    #         module_list == [self.expr_branch]
    #     elif branch_name == "trans":
    #         module_list = [self.trans_branch]
    #     else:
    #         raise ValueError('branch_name not supported: must be in [pose, expr, trans] !')
        
    #     params = self.get_model_parameters(module_list)
    #     for p in params:
    #         p.requires_grad=False

    # def unfreeze_branch_params(self, branch_name):
    #     if branch_name == "pose":
    #         module_list = [self.pose6d_branch]
    #     elif branch_name == "expr":
    #         module_list == [self.expr_branch]
    #     elif branch_name == "trans":
    #         module_list = [self.trans_branch]
    #     else:
    #         raise ValueError('branch_name not supported: must be in [pose, expr, trans] !')
    #     params = self.get_model_parameters(module_list)
    #     for p in params:
    #         p.requires_grad = True

    def forward(self, x, timesteps, lmk_3d, lmk_2d, img_arr, force_mask=False, return_mica=False, **kwargs):
        """
        Args:
            x: [batch_size, nfeats, nframes], denoted x_t in the paper
            sparse: [batch_size, nframes, sparse_dim], the sparse features
            timesteps: [batch_size] (int)
        """

        # mask the condition 
        lmk3d_masked = self.mask_cond_sparse(lmk_3d, force_mask=force_mask)
        lmk2d_masked = self.mask_cond_sparse(lmk_2d, force_mask=force_mask)
        img_masked = self.mask_cond_sparse(img_arr, force_mask=force_mask)

        # separate target
        input_shape = x[:, 0, :self.shape_dim]  # (bs, 300)
        input_motion = x[:, :, self.shape_dim:-3]  # (bs,n,30+100)
        input_trans = x[:, :, -3:]  # (bs,n,3)

        # pred shape from cropped image using mica
        shape_mica = self.mica(img_masked) # (bs,300)
        shape_mica_emb = self.mica_shape_process(shape_mica)

        # process the condition to the same cond latent dim space
        lmk3d_emb = self.lmk3d_process(lmk3d_masked)
        lmk2d_emb = self.lmk2d_process(lmk2d_masked)

        # pred shape branch
        output_shape = self.shape_branch(input_shape, shape_mica_emb, timesteps)

        # pred motion branch
        shape_pred_emb = self.pred_shape_process(output_shape).unsqueeze(1) # (bs, 1, cond_dim)
        motion_cond_emb = shape_pred_emb + lmk3d_emb
        output_motion = self.motion_branch(input_motion, motion_cond_emb, timesteps)

        # pred trans branch
        motion_pred_emb = self.pred_motion_process(output_motion)
        trans_cond_emb = lmk2d_emb + motion_pred_emb + shape_pred_emb
        output_trans = self.trans_branch(input_trans, trans_cond_emb, timesteps)

        # concat the output
        output = torch.cat(
            [output_shape.unsqueeze(1).expand(-1,output_trans.shape[1],-1), output_motion, output_trans], 
            axis = -1
        )
        assert not shape_mica.isnan().any()
        assert not output_shape.isnan().any()
        assert not output_motion.isnan().any()
        assert not output_trans.isnan().any()

        if return_mica:
            return output, shape_mica
        
        return output