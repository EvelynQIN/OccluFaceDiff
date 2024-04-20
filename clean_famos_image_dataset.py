import os
import glob 
import os
import cv2

def main(img_folder):
    keep_cams = '26_C'
    
    for sub in os.listdir(img_folder):
        if 'readme' in sub:
            continue
        for motion in os.listdir(os.path.join(img_folder, sub)):
            files = glob.glob(os.path.join(img_folder, sub, motion, '*.jpg'))
            for f in files:
                if keep_cams not in f:
                    os.remove(f)
            

if __name__ == "__main__":
    img_folder = 'dataset/vocaset/image'
    main(img_folder)