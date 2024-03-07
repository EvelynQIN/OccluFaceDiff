# Face animation: use diffusion model to reconstruct the face motion

## Enviroment Setup

## Fit Flame Parameters for the Registration Meshes
```
python3 compute_flame_params.py --source_path dataset/FaMoS/registrations --to_folder dataset/FaMoS/flame_params >&flame_fitting.log &
```

## Dataset Preparation

Prepare for the FaMoS dataset

```
python3 prepare_FaMoS.py --root_dir ./dataset --save_dir ./processed_data
```
The generated dataset should look like this
```
./processed_data/FaMoS/
├──── train/
├──── val/
├──── test/
```

## Evaluation

To evaluate the model and generate test visualization:
```
python3 test.py --model_path checkpoints/DiffMLP/model_7.pt --fps 60 --output_dir vis_result --vis
```

## Training
To train the diffusion-model:
```
python3 train_diffTrans.py --config_path configs/train_diffusion_transformer.yaml
```