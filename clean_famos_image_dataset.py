import os
import glob 
import argparse
import json
import logging
import multiprocessing as mp
import os

import requests
from bs4 import BeautifulSoup

def main():
    keep_cams = ['23_C', '26_C']
    img_folder = 'dataset/FaMoS/downsampled_images_4'
    
    for sub in os.listdir(img_folder):
        if os.path.split(sub)[-1].endswith("readme"):
            continue
        for motion in os.listdir(os.path.join(img_folder, sub)):
            files = glob.glob(os.path.join(img_folder, sub, motion, '**/*.png'))
            for f in files:
                cam_type = os.path.split(f)[-1].split('.')[2]
                if cam_type not in keep_cams:
                    os.remove(f)
            

if __name__ == "__main__":
    main()