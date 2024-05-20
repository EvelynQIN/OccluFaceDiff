python3 train_diffTrans.py --config_path configs/train_diffusion_trans.yaml >&trainTrans.log &
python3 train_diffMLP.py --config_path configs/train_diffusion_mlp.yaml
python3 train_diffTrans_FLINT.py --config_path configs/train_diffusion_trans.yaml