import argparse
import os
import numpy as np
import torch
from tqdm import tqdm 
import glob
from utils.data_util import get_arcface_input, crop_np, batch_normalize_lmk_3d, landmarks_interpolate
from skimage.transform import estimate_transform, warp, resize, rescale
from skimage.io import imread
import face_alignment 
import cv2 
from loguru import logger

from mmseg.apis import inference_model, init_model

class VideoProcessor:
    def __init__(self, config=None, image_folder='./outputs'):
        self.device = 'cuda:0'
        self.config = config
        self.fps = config.fps 
        self.image_size = config.image_size
        self.scale = config.scale 
        self.image_folder = image_folder
        self.K = config.input_motion_length # motion length to chunk
        
        # landmark detector
        face_detector_kwargs = {
            'back_model': False
        }

        self.face_detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, 
            flip_input=False,
            face_detector='blazeface',    # support detectors ['dlib', 'blazeface', 'cfd]
            face_detector_kwargs = face_detector_kwargs,
            device=self.device,)
        logger.info("Use Face Alignment Predictor for 2d LMK Detection.")
        
        # image segmentator 
        config_file = 'face_segmentation/deeplabv3plus_r101_512x512_face-occlusion.py'
        checkpoint_file = 'face_segmentation/deeplabv3plus_r101_512x512_face-occlusion-93ec6695.pth'
        self.segmentator = init_model(config_file, checkpoint_file, device=self.device)
        logger.info("Use mmseg for Face Segmentation.")
    
    def crop_face(self, landmarks):
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])

        size = int(old_size * self.scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        return tform

    def video_to_frames(self):
        if self.config.test_mode == 'in_the_wild':
            video_path = self.config.video_path
            if not os.path.exists(video_path):
                logger.error(f'Video path {video_path} not existed!')
                exit(1) 
            video_name = os.path.split(video_path)[1].split('.')[0]
            frame_dst = os.path.join(self.image_folder, video_name)
            if not os.path.exists(frame_dst):
                os.makedirs(frame_dst)
                os.system(f'ffmpeg -i {video_path} -vf fps={self.fps} -q:v 1 {frame_dst}/%05d.png')
        else:
            frame_dst = self.config.image_folder
            if not os.path.exists(frame_dst):
                logger.error(f'Frame dst folder {frame_dst} not existed!')
                exit(1) 
        
        self.image_paths = sorted(glob.glob(f'{frame_dst}/*.png'))
        self.num_frames = len(self.image_paths)
        logger.info(f"Motion Len = {self.num_frames}")
    
    def detection_lmks(self):
        lmk68 = []
        null_face_cnt = 0
        for img_path in tqdm(self.image_paths):
            frame = cv2.imread(img_path)
            h, w, _ = frame.shape
            lmks, scores, bbox = self.face_detector.get_landmarks_from_image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), return_bboxes=True, return_landmark_score=True)
            if bbox is None:
                null_face_cnt += 1
                lmk68.append(None)
            else:
                lmk68.append(lmks[0])
        
        # linear interpolate the missing landmarks
        self.lmk68 = landmarks_interpolate(lmk68)
        logger.info(f"There are {null_face_cnt} frames detected null faces.")

    def process_frame(self, frame, kpt, face_seg=False):
        """segmentation mask for the given frame
        Args:
            frame: BGR image array
            kpt: (68, 2)
        """
        h, w = frame.shape[:2]
        # crop information
        tform = self.crop_face(kpt)
        cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T
        cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1
        cropped_image = warp(frame, tform.inverse, output_shape=(self.image_size, self.image_size))
        cropped_image_rgb = cropped_image.transpose(2,0,1)[[2,1,0],:,:] # (3, 224, 224) in RGB
        if face_seg:
            # occlusion information 
            seg_output = inference_model(self.segmentator, cropped_image)
            seg_mask = np.asanyarray(seg_output.pred_sem_seg.values()[0].to('cpu')).squeeze(0)   # (224, 224)
            return cropped_kpt, cropped_image_rgb, seg_mask
        else:
            return cropped_kpt, cropped_image_rgb, None

    def preprocess_video(self):
        self.video_to_frames()
        self.detection_lmks()
        
    def prepare_chunk_motion(self, start_id, face_seg=False):

        # prepare test data for one motion chunk of length self.K due to memory limit
        imgs = []
        lmk_2d = []
        face_masks = []

        for i in range(start_id, min(self.num_frames, start_id+self.K)):
            img_path = self.image_paths[i]
            kpt = self.lmk68[i]
            frame = cv2.imread(img_path)
            cropped_kpt, cropped_image_rgb, seg_mask = self.process_frame(frame, kpt, face_seg)
            face_masks.append(seg_mask)
            imgs.append(cropped_image_rgb)
            lmk_2d.append(cropped_kpt)
        
        imgs = np.stack(imgs)
        lmk_2d = np.stack(lmk_2d)
        output = {
            "lmk_2d": torch.from_numpy(lmk_2d).float(), # (k, 68, 3)
            "image": torch.from_numpy(imgs).type(dtype = torch.float32),   # (k, 3, 224, 224)
        }
        if face_seg:
            face_masks = np.stack(face_masks)
            output["face_mask"] = torch.from_numpy(face_masks),   # (k, 224, 224)

        return output

if __name__ == "__main__":
    video_processor = VideoProcessor()
    output = video_processor.run('videos/justin.mp4')
    for k in output:
        if k == 'frame_id':
            print(f'{k}:', len(output[k]))
        else:
            print(f'{k}:', output[k].shape)