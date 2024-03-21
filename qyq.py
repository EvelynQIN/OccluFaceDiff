import torch 
from glob import glob 
import os 
from tqdm import tqdm
from configs.config import get_cfg_defaults 
from model.calibration_layer import Cam_Calibration


def main(args):
        folder = 'processed_data/FaMoS'
        device = 'cuda'

        model = Cam_Calibration(
                lmk2d_dim=args.lmk2d_dim, # input feature dim 68 x 2
                n_target=args.n_target,
                output_feature_dim=args.output_nfeat, # number of cam params (one set per frame)
                latent_dim=args.latent_dim,
                ckpt_path=args.ckpt_path,
        )
        model.to(device)
        model.eval()
        norm_dict_cam = torch.load('processed_data/FaMoS_CamCalib_norm_dict.pt')
        mean_target = norm_dict_cam['mean_target']
        std_target = norm_dict_cam['std_target']
        trans_offset = torch.FloatTensor(args.trans_offset).unsqueeze(0)

        phases = ['train', 'val', 'test']
        for phase in phases:
                print("process ", phase)
                files = glob(os.path.join(folder, f"{phase}/*"))
                for f in tqdm(files):
                        motion = torch.load(f)
                        flame_full = motion["target"] 
                        flame_180 = torch.cat([flame_full[:,:100], flame_full[:,300:350], flame_full[:,400:]], dim=-1) # (n, 180)
                        n = flame_180.shape[0]
                        lmk_2d = motion["lmk_2d"].reshape(n, -1).to(device) # (n, 68, 2)
                        # normalization
                        target = (flame_180 - mean_target) / (std_target + 1e-8)
                        target = target.to(device)
                        cam_t = model(lmk_2d, target).to('cpu')
                        cam_t = cam_t + trans_offset    # (n, 3)
                        motion["target"] = torch.cat([flame_180, cam_t], dim=-1)
                        assert motion["target"].shape[1] == 183
                        
                        # resave
                        torch.save(motion, f)

if __name__ == "__main__":
        
        args = get_cfg_defaults().cam
        main(args)