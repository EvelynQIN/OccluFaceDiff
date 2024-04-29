python3 test_motion_multiface.py --model_path checkpoints/Transformer_512d_2l_0.3occ_joint/model_26.pt --save_folder vis_result --exp_name random_occlusion --subject_id 002539136 --input_motion_length 20 --sld_wind_size 15 --occlusion_mask_prob 1

python3 test_motion_voca.py --model_path checkpoints/Transformer_512d_2l_0.3occ_joint/model_26.pt --save_folder vis_result --exp_name non_occ --subject_id FaceTalk_170728_03272_TA --input_motion_length 20 --sld_wind_size 15 --occlusion_mask_prob 0
