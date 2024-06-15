# diffusion
python3 test_with_random_mask/test_motion_mead_lmk_flint.py --model_path checkpoints/Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/model_46.pt --split test --input_motion_length 64 --exp_name mouth --save_rec 
python3 test_with_random_mask/test_motion_mead_lmk_flint.py --model_path checkpoints/Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/model_46.pt --split test --input_motion_length 64 --exp_name upper --save_rec 
python3 test_with_random_mask/test_motion_mead_lmk_flint.py --model_path checkpoints/Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/model_46.pt --split test --input_motion_length 64 --exp_name random --save_rec
# --test_dataset RAVDESS --save_folder vis_RAVDESS

# emoca
python3 test_with_random_mask/test_motion_mead_lmk_EMOCA.py --sld_wind_size 64 --exp_name mouth --save_rec --mask_path vis_result/diffusion_Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/mouth/occ_mask
python3 test_with_random_mask/test_motion_mead_lmk_EMOCA.py --sld_wind_size 64 --exp_name upper --save_rec --mask_path vis_result/diffusion_Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/upper/occ_mask
python3 test_with_random_mask/test_motion_mead_lmk_EMOCA.py --sld_wind_size 64 --exp_name random --save_rec --mask_path vis_result/diffusion_Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/random/occ_mask

# spectre
python3 test_with_random_mask/test_motion_mead_lmk_SPECTRE.py --exp_name mouth --save_rec --mask_path vis_result/diffusion_Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/mouth/occ_mask
python3 test_with_random_mask/test_motion_mead_lmk_SPECTRE.py --exp_name upper --save_rec --mask_path vis_result/diffusion_Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/upper/occ_mask
python3 test_with_random_mask/test_motion_mead_lmk_SPECTRE.py --exp_name random --save_rec --mask_path vis_result/diffusion_Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/random/occ_mask

# # grid vis
python3 test_with_random_mask/grid_visualization.py --exp_name mouth --split test --with_audio
python3 test_with_random_mask/grid_visualization.py --exp_name upper --split test --with_audio
python3 test_with_random_mask/grid_visualization.py --exp_name random --split test --with_audio