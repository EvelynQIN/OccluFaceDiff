# python3 test_motion_mead_lmk_flint.py --model_path checkpoints/Transformer_768d_cat_mediapipelmk_FLINT_stage2_loadtex/model_53.pt --split test --input_motion_length 64 --exp_name cfg_0.5_10 --save_folder vis_result --subject_id M003 --emotion fear --level level_3 --guidance_param_all 0.5 --guidance_param_audio 10
python3 test_motion_mead_lmk_flint.py --model_path checkpoints/Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/model_46.pt --split test --input_motion_length 64 --exp_name all --save_rec --test_dataset RAVDESS --save_folder vis_RAVDESS
python3 test_motion_mead_lmk_flint.py --model_path checkpoints/Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/model_46.pt --split test --input_motion_length 64 --exp_name mouth --save_rec --test_dataset RAVDESS --save_folder vis_RAVDESS


































