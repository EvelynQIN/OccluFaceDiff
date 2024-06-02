# python3 test_motion_mead_lmk_flint.py --model_path checkpoints/Transformer_768d_cat_mediapipelmk_FLINT_video_classifier_s2/model_56.pt --split test --input_motion_length 64 --exp_name non_occ --save_folder vis_result_56 --save_rec
# --guidance_param_all 2.5 --guidance_param_audio 0.5
python3 test_motion_mead_lmk_flint.py --model_path checkpoints/Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/model_46.pt --split test --input_motion_length 64 --exp_name non_occ --save_folder vis_result --subject_id M005 --level level_3 --vis --emotion angry
python3 test_motion_mead_lmk_flint.py --model_path checkpoints/Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/model_46.pt --split test --input_motion_length 64 --exp_name all --save_rec --test_dataset RAVDESS --save_folder vis_RAVDESS
python3 test_motion_mead_lmk_flint.py --model_path checkpoints/Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/model_46.pt --split test --input_motion_length 64 --exp_name mouth --save_rec --test_dataset RAVDESS --save_folder vis_RAVDESS


































