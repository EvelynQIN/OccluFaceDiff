python3 predict.py --model_path checkpoints/e2e1/Transformer_68_256d_1l_4h_concat/model_11.pt --motion_id high_smile --subject_id FaMoS_subject_007 --save_folder vis_result --exp_name non_occ --split val --test_mode FaMoS

python3 predict.py --model_path checkpoints/e2e1/Transformer_68_256d_1l_4h_concat/model_11.pt --save_folder vis_result --exp_name non --test_mode in_the_wild --video_path videos/justin.mp4
