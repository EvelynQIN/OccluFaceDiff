import argparse
import os
import numpy as np
import torch
from tqdm import tqdm 
import glob
from utils.image_process import get_arcface_input, crop_np, batch_normalize_lmk_3d
from skimage.transform import estimate_transform, warp, resize, rescale

import face_alignment 
import cv2 
from loguru import logger

from mmseg.apis import inference_model, init_model

class VideoProcessor:
    def __init__(self, config=None, image_folder='./outputs'):
        self.device = 'cuda:0'
        self.config = config
        self.fps = config.fps 
        self.image_size = config.image_size[0]
        self.trans_scale = config.trans_scale
        self.scale = config.scale 
        self.image_folder = image_folder
        
        # landmark detector
        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device=self.device)
        
        # image segmentator 
        config_file = 'face_segmentation/deeplabv3plus_r101_512x512_face-occlusion.py'
        checkpoint_file = 'face_segmentation/deeplabv3plus_r101_512x512_face-occlusion-93ec6695.pth'
        self.segmentator = init_model(config_file, checkpoint_file, device=self.device)

    def video_to_frames(self, video_path):

        if not os.path.exists(video_path):
            logger.error(f'Video path {video_path} not existed!')
            exit(1) 
        video_name = os.path.split(video_path)[1].split('.')[0]
        frame_dst = os.path.join(self.image_folder, video_name)
        if not os.path.exists(frame_dst):
            os.makedirs(frame_dst)
            os.system(f'ffmpeg -i {video_path} -vf fps={self.fps} -q:v 1 {frame_dst}/%05d.png')

        self.image_paths = sorted(glob.glob(f'{frame_dst}/*.png'))

    def process_frame(self, frame, tform_prev):
        """detect landmarks and segmentation mask for the given frame
        Args:
            frame: BGR image array
        
        """
        lmks, scores, bbox = self.face_detector.get_landmarks_from_image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), return_bboxes=True, return_landmark_score=True)
        h, w = frame.shape[:2]
        
        if bbox is None:
            lmk_3d = np.zeros(68, 3)
            cropped_lmk2d = np.zeros(68, 2)
            assert tform_prev is not None, "tform not available!"
            tform = tform_prev
            arcface_input = None 
            occlusion_mask = np.ones(68)    # all landmarks occluded for non-detected face frame
            
        else:
            lmk_3d = lmks[0]    # (68, 3)
            score = scores[0]
            lmk_2d = lmk_3d.copy()[:,:2]    # (68, 2)
            
            lmk_5 = lmk_2d[[37, 44, 30, 60, 64], :]  # left eye, right eye, nose, left mouth, right mouth
            lmk_5[0, :] = lmk_2d[[38, 41], :].mean(0)  # center of left eye
            lmk_5[1, :] = lmk_2d[[44, 47], :].mean(0)  # center of right eye
            
            # crop information
            tform = crop_np(lmk_2d, self.trans_scale, self.scale, self.image_size)
            cropped_lmk2d = np.dot(tform.params, np.hstack([lmk_2d, np.ones([lmk_2d.shape[0],1])]).T).T[:,:2]   # (68, 2)
            
            arcface_input = get_arcface_input(lmk_5, frame)
            
            # occlusion information 
            seg_output = inference_model(self.segmentator, frame)
            seg_mask = np.asanyarray(seg_output.pred_sem_seg.values()[0].to('cpu')).squeeze(0)   # (h, w)
            
            occlusion_mask = np.zeros(lmk_2d.shape[0])
            for i in range(lmk_2d.shape[0]):
                x, y = lmk_2d[i].astype(int)
                if x < 0 or x >= w or y < 0 or y >= h or (seg_mask[y, x] and score[i] < 0.7) or score[i] < 0.65:
                    occlusion_mask[i] = 1
            
        frame = frame.astype(float) / 255.0
        cropped_image = warp(frame, tform.inverse, output_shape=(self.image_size, self.image_size)) 
        cropped_image_rgb = cropped_image.transpose(2,0,1)[[2, 1, 0], :, :] # (3, 224, 224) in RGB
        
        return cropped_lmk2d, lmk_3d, occlusion_mask, arcface_input, cropped_image_rgb, tform
 
    def run(self, video_path):
        self.video_to_frames(video_path)
        tform_prev = None
        arcface_inputs = []
        imgs = []
        lmk_2ds = []
        lmk_3ds = []
        occlusion_masks = []
        frame_ids = []

        for img_path in tqdm(self.image_paths):
            frame = cv2.imread(img_path)
            cropped_lmk2d, lmk_3d, occlusion_mask, arcface_input, cropped_image_rgb, tform = self.process_frame(frame, tform_prev)
            tform_prev = tform 
            occlusion_masks.append(occlusion_mask)
            lmk_2ds.append(cropped_lmk2d)
            lmk_3ds.append(lmk_3d)
            imgs.append(cropped_image_rgb)
            if arcface_input is not None:
                arcface_inputs.append(arcface_input)
            frame_id = os.path.split(img_path)[-1][:-4]
            frame_ids.append(int(frame_id))
        
        arcface_inputs = np.stack(arcface_inputs)
        imgs = np.stack(imgs)
        lmk_3ds = np.stack(lmk_3ds)
        lmk_2ds = np.stack(lmk_2ds)
        occlusion_masks = np.stack(occlusion_masks)
        img_mask = np.ones(len(frame_ids))
        
        # batch normalize 3d landmarks
        lmk_3ds = torch.from_numpy(lmk_3ds).float()
        lmk_3d_normed = batch_normalize_lmk_3d(lmk_3ds)              
    
        output = {
            "lmk_2d": torch.from_numpy(lmk_2ds).float(), # (n, 68, 2)
            "lmk_3d_normed": lmk_3d_normed, # (n, 68, 3)
            "frame_id": torch.LongTensor(frame_ids), # (n)
            "img_mask": torch.from_numpy(img_mask).bool(),   # (n)
            "arcface_input": torch.from_numpy(arcface_inputs).float(), # (n_imgs, 3, 112, 112)
            "cropped_imgs": torch.from_numpy(imgs).float(),   # (n_imgs, 3, 224, 224)
            "occlusion_mask": torch.from_numpy(occlusion_masks).float()
        }
        
        return output

if __name__ == "__main__":
    video_processor = VideoProcessor()
    output = video_processor.run('videos/justin.mp4')
    for k in output:
        if k == 'frame_id':
            print(f'{k}:', len(output[k]))
        else:
            print(f'{k}:', output[k].shape)