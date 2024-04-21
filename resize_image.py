import os
import glob 
import os
import cv2
import numpy as np 
from tqdm import tqdm

def main(img_folder):
    h, w = 1200, 1600
    scale_factor = 4
    new_h, new_w = int(h / scale_factor), int(w / scale_factor)
    for sub in tqdm(os.listdir(img_folder)):
        if 'readme' in sub:
            continue
        for motion in os.listdir(os.path.join(img_folder, sub)):
            files = glob.glob(os.path.join(img_folder, sub, motion, '*.jpg'))
            for f in files:
                img = cv2.imread(f)
                if img is None:
                    print(f"{f} is None!")
                    img = np.zeros((new_h, new_w, 3)).astype(np.uint8)
                    cv2.imwrite(f, img)
                    continue

                if img.shape[0] == new_h:
                    continue
                
                img = cv2.resize(img, (new_w, new_h))
                cv2.imwrite(f, img)
                
            

if __name__ == "__main__":
    img_folder = 'dataset/vocaset/image'
    main(img_folder)