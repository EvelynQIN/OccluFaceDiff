import cv2
import numpy as np
import os 
from glob import glob
for occ in os.scandir('vis_result/diffusion_Transformer_768d_cat_mediapipelmk_FLINT_video_classifier_s2'):
    for mask_file in glob(os.path.join(occ.path, f'reconstruction/*/*/*/*/*_mask.npy')):
        os.system(f'rm {mask_file}')








