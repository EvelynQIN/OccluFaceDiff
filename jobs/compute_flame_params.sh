#!/bin/bash
conda activate general
python3 compute_flame_params.py --source_path dataset/FaMoS/registrations --to_folder dataset/FaMoS/flame_params >&flame_fitting.log &