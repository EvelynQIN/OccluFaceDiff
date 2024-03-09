import torch 
from model.denoising_model import GRUDecoder, PositionalEncoding, TransformerEncoder_TS, FaceTransformer
device = "cuda"
kwargs = {
        "arch": "transTrans",
        "nfeats": 124,
        "latent_dim": 256,
        "sparse_dim": 176*3,
        "dropout": 0.1,
        "cond_mask_prob": 0,
        "dataset": "FaMoS",
        "ff_size": 512,
        "num_enc_layers": 4,
        "num_dec_layers": 4,
        "num_heads": 8, 
    }

model = FaceTransformer(**kwargs).to(device)
bs = 1
seqlen = 150
x = torch.rand(bs, seqlen, kwargs["nfeats"]).to(device)
sparse = torch.rand(bs, seqlen, kwargs["sparse_dim"]).to(device)
timestep = torch.randint(1, 1000, (bs,)).to(device)

output = model(x, timestep, sparse)
print(f"shape of output {output.shape}")