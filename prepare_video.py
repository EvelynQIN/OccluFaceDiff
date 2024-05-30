import argparse
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm 
import glob
from utils.data_util import linear_interpolate_landmarks, point2bbox, point2transform
from skimage.transform import estimate_transform, warp, resize, rescale
import face_alignment 
import cv2 
from loguru import logger
from transformers import Wav2Vec2Processor
import librosa
# from model.wav2vec import Wav2Vec2Model
# from mmseg.apis import inference_model, init_model
import h5py
from utils.mediapipe_landmark_detection import MediapipeDetector
from model.deca import EMOCA
from collections import defaultdict
from utils import utils_transform

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class VideoProcessor:
    def __init__(self, model_cfg, config=None, data_folder='dataset/in_the_wild'):
        self.device = 'cuda:0'
        self.config = config
        self.fps = config.fps 
        self.image_size = config.image_size
        self.scale = 1.35
        self.data_folder = data_folder
        self.K = config.input_motion_length # motion length to chunk
        self.wav_per_frame = int(16000 / self.fps)

        self.face_detector = MediapipeDetector()
        logger.info("Use Mediapipe Predictor for 2d LMK Detection.")
        
        # Create a image segmenter instance with the video mode for mediapipe
        model_path = 'face_segmentation/selfie_multiclass_256x256.tflite'
        BaseOptions = mp.tasks.BaseOptions
        base_options = BaseOptions(model_asset_path=model_path)
        self.ImageSegmenter = mp.tasks.vision.ImageSegmenter
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = ImageSegmenterOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.VIDEO,
            output_category_mask=True)
        logger.info("Use mediapipe multiclass segment for Face Segmentation.")

        self.emoca = EMOCA(model_cfg)
        self.emoca.to(self.device)

        # wav2vec processor
        self.audio_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h") 
        
    def warp_image_from_lmk(self, landmarks, img):    
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0 
        center_y = bottom - (bottom - top) / 2.0
        center = np.array([center_x, center_y])

        size = int(old_size * self.scale)

        tform = point2transform(center, size, self.image_size, self.image_size)
        output_shape = (self.image_size, self.image_size)
        dst_image = warp(img, tform.inverse, output_shape=output_shape, order=3)
        dst_landmarks = tform(landmarks[:, :2])

        return dst_image, dst_landmarks

    def downsample_video(self):
        video_path = self.config.video_path
        if not os.path.exists(video_path):
            logger.error(f'Video path {video_path} not existed!')
            exit(1) 
        self.video_name = os.path.split(video_path)[1].split('.')[0]
        self.processed_dst = os.path.join(self.data_folder, self.video_name)
        self.audio_path = os.path.join(self.processed_dst, 'audio.wav')
        self.video_path = os.path.join(self.processed_dst, 'video_25fps.mp4')
        if not os.path.exists(self.processed_dst):
            os.makedirs(self.processed_dst)
            os.system(f'ffmpeg -i {video_path} -r {self.fps} {self.video_path}')
            os.system("ffmpeg -i {} {} -y".format(video_path, self.audio_path))
    
    def landmark_detection(self):
        logger.info(f"Start landmark detection.")
        lmks = []
        null_face_cnt = 0
        video = cv2.VideoCapture()
        if not video.open(self.video_path):
            logger.error(f"Cannot open {self.video_path}!")
            exit(1)
        while True:
            _, frame = video.read()
            if frame is None:
                break
            lmk_2d = self.face_detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if lmk_2d is None:
                null_face_cnt += 1
            lmks.append(lmk_2d)
        video.release()
        # linear interpolate the missing landmarks
        valid_frames_idx, landmarks = linear_interpolate_landmarks(lmks)
        self.landmarks = np.stack(landmarks)[:,:,:2]   # (n, V, 2)
        self.valid_frames_idx = valid_frames_idx
        logger.info(f"There are {null_face_cnt} frames detected null faces.")

    def process_one_video(self):

        self.processed_fname = os.path.join(self.processed_dst, 'processed.hdf5')
        if os.path.exists(self.processed_fname):
            logger.info(f"Video alreadly processed!")
            return

        self.landmark_detection()
        # read the audio input and preprocess it
        speech_array, sampling_rate = librosa.load(self.audio_path, sr=16000)
        audio_input = np.squeeze(self.audio_processor(
                speech_array, 
                return_tensors=None, 
                padding="longest",
                sampling_rate=sampling_rate).input_values)
        
        video = cv2.VideoCapture()
        if not video.open(self.video_path):
            logger.error(f"Cannot open {self.video_path}!")
            exit(1)
        
        lmk_list = []
        frame_list = []
        seg_list = []

        frame_id = 0
        with self.ImageSegmenter.create_from_options(self.options) as segmenter:
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                dst_image, dst_landmarks = self.warp_image_from_lmk(self.landmarks[frame_id], frame)

                # normalize landmarks to [-1, 1]
                dst_landmarks = dst_landmarks / self.image_size * 2 - 1
                lmk_list.append(dst_landmarks)  # (V, 2)
                
                # perform image segmentation to the warped image
                seg_image = (dst_image * 255.).astype(np.uint8)[:,:,::-1].copy() # (h, w, 3) in RGB
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=seg_image)
                frame_timestamp_ms = int(1000 * frame_id / self.fps)   
                seg_mask = segmenter.segment_for_video(mp_image, frame_timestamp_ms).category_mask
                seg_mask = seg_mask.numpy_view() == 3   # only extract facial skins
                seg_list.append(seg_mask)

                dst_image = (dst_image[:,:,[2,1,0]]).transpose(2, 0, 1)   # (3, h, w) in RGB float
                frame_list.append(dst_image)
                frame_id += 1
            video.release()
        frame_list = np.stack(frame_list)   # (n, 3, 224, 224)
        seg_list = np.stack(seg_list)   # (n, 224, 224)
        lmk_list = np.stack(lmk_list)   # (n, V, 2)

        
        f = h5py.File(self.processed_fname, 'w')
        f.create_dataset('lmk_2d', data=lmk_list)
        f.create_dataset('valid_frames_idx', data=np.array(self.valid_frames_idx))
        f.create_dataset('image', data=frame_list)
        f.create_dataset('img_mask', data=seg_list)
        f.create_dataset('audio_input', data=audio_input)
        f.close()
    
    def run_emoca(self):

        self.emoca_rec_fname = os.path.join(self.processed_dst, 'emoca.pt')
        if os.path.exists(self.emoca_rec_fname):
            logger.info(f"Emoca already run on this video.")
            return

        with h5py.File(self.processed_fname, 'r') as f:
            image = torch.from_numpy(f['image'][:]).float()
        
        n_frames = image.shape[0]
        emoca_code = defaultdict(list)
        for i in range(0, n_frames, 20):
            batch = image[i:i+20].to(self.device)
            emoca_batch = self.emoca(batch)
            for key in emoca_batch:
                emoca_code[key].append(emoca_batch[key].to('cpu'))
        for key in emoca_code:
            emoca_code[key] = torch.cat(emoca_code[key], dim=0)
        
        emoca_code['jaw'] = emoca_code['pose'][...,3:]
        emoca_code['global_pose'] = emoca_code['pose'][...,:3]
        emoca_code.pop('pose', None)
        torch.save(emoca_code, self.emoca_rec_fname)
    
    def _get_lmk_mask_from_img_mask(self, img_mask, lmks):
        kpts = (lmks.clone() * 112 + 112).long()
        n, v = kpts.shape[:2]
        lmk_mask = torch.ones((n,v))
        for i in range(n):
            for j in range(v):
                x, y = kpts[i,j]
                if x<0 or x >=self.image_size or y<0 or y>=self.image_size or img_mask[i,y,x]==0:
                    lmk_mask[i,j] = 0
        return lmk_mask

    def preprocess_video(self):
        self.downsample_video()
        self.process_one_video()
        self.run_emoca()

    def get_processed_data(self):
        # get emoca code
        data_dict = torch.load(self.emoca_rec_fname)
        data_dict['shape'] = data_dict['shape'][...,:self.config.n_shape]
        data_dict['exp'] = data_dict['exp'][...,:self.config.n_exp]

        # get processed data
        with h5py.File(self.processed_fname, 'r') as f:
            for key in f:
                data_dict[key] = torch.from_numpy(f[key][:]).float()
        
        if not self.config.use_iris:
            data_dict['lmk_2d'] = data_dict['lmk_2d'][:,:468]
        num_frames = data_dict['lmk_2d'].shape[0]
        
        remain_audio_len = num_frames * self.wav_per_frame - len(data_dict['audio_input'])

        if remain_audio_len > 0:
            data_dict['audio_input'] = nn.functional.pad(
                data_dict['audio_input'], (0, remain_audio_len))
        else:
            # trim the audio to align with the video
            data_dict['audio_input'] = data_dict['audio_input'][:num_frames * self.wav_per_frame]
        
        data_dict['audio_input'] = data_dict['audio_input'].reshape(num_frames, -1)
        data_dict['lmk_mask'] = self._get_lmk_mask_from_img_mask(data_dict['img_mask'], data_dict['lmk_2d'])
            
        if self.config.mode == 'audio':
            data_dict['lmk_mask'][:,:] = 0
        elif self.config.mode == 'visual':
            data_dict['audio_input'][:,:] = 0

        return data_dict, self.video_name, self.audio_path