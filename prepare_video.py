import argparse
import os
import numpy as np
import torch
from tqdm import tqdm 
import glob
from utils.data_util import landmarks_interpolate
from skimage.transform import estimate_transform, warp, resize, rescale
import face_alignment 
import cv2 
from loguru import logger
from transformers import Wav2Vec2Processor
import librosa
from model.wav2vec import Wav2Vec2Model
from mmseg.apis import inference_model, init_model
from moviepy.editor import VideoFileClip

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

        # wav2vec
        self.audio_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h") 
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec.feature_extractor._freeze_parameters()
        self.wav2vec.to(self.device)

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
        video_path = self.config.video_path
        if not os.path.exists(video_path):
            logger.error(f'Video path {video_path} not existed!')
            exit(1) 
        video_name = os.path.split(video_path)[1].split('.')[0]
        frame_dst = os.path.join(self.image_folder, video_name)
        if not os.path.exists(frame_dst):
            os.makedirs(frame_dst)
            os.system(f'ffmpeg -i {video_path} -vf fps={self.fps} -q:v 1 {frame_dst}/%05d.png')
            if self.config.with_audio:
                self.audio_path = os.path.join(frame_dst, 'audio.wav')
                os.system("ffmpeg -i {} {} -y".format(video_path, self.audio_path))
        
        self.image_paths = sorted(glob.glob(f'{frame_dst}/*.png'))
        self.num_frames = len(self.image_paths)
        logger.info(f"Motion Len = {self.num_frames}")

        if self.config.with_audio:
            speech_array, sampling_rate = librosa.load(self.audio_path, sr=16000)
            self.audio_values = np.squeeze(
                self.audio_processor(
                    speech_array, 
                    return_tensors='pt', 
                    padding="longest",
                    sampling_rate=sampling_rate).input_values)
            audio_input = audio_input.float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                self.audio_emb = self.wav2vec(audio_input, frame_num = self.num_frames).last_hidden_state.squeeze(0).cpu()
        else:
            self.audio_emb = torch.zeros(self.num_frames, 768)
    
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

    def process_frame(self, frame, kpt):
        """segmentation mask for the given frame
        Args:
            frame: BGR image array
            kpt: (68, 2)
        """
        h, w = frame.shape[:2]
        # crop information
        tform = self.crop_face(kpt)
        cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T
        cropped_image = warp(frame, tform.inverse, output_shape=(self.image_size, self.image_size))
        cropped_image_rgb = cropped_image.transpose(2,0,1)[[2,1,0],:,:] # (3, 224, 224) in RGB

        # occlusion information 
        seg_output = inference_model(self.segmentator, (cropped_image * 255).astype(np.uint8))
        seg_mask = np.asanyarray(seg_output.pred_sem_seg.values()[0].to('cpu')).squeeze(0)   # (224, 224)

        # get lmk occlusion info from seg_mask
        lmk_mask = np.ones(68)
        for i in range(68):
            x, y = kpt[i]
            if x < 0 or x >= self.image_size or y < 0 or y >= self.image_size or seg_mask[int(y),int(x)] == 0:
                lmk_mask[i] = 0

        # kpt normalization
        cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1
        return cropped_kpt, cropped_image_rgb, seg_mask, lmk_mask

    def preprocess_video(self):
        self.video_to_frames()
        self.detection_lmks()
        
    def prepare_chunk_motion(self, start_id):

        # prepare test data for one motion chunk of length self.K due to memory limit
        imgs = []
        lmk_2d = []
        face_masks = []
        lmk_masks = []

        for i in range(start_id, min(self.num_frames, start_id+self.K)):
            img_path = self.image_paths[i]
            kpt = self.lmk68[i]
            frame = cv2.imread(img_path)
            cropped_kpt, cropped_image_rgb, seg_mask, lmk_mask = self.process_frame(frame, kpt)
            face_masks.append(seg_mask)
            imgs.append(cropped_image_rgb)
            lmk_2d.append(cropped_kpt)
            lmk_masks.append(lmk_mask)
        
        imgs = np.stack(imgs)
        lmk_2d = np.stack(lmk_2d)
        face_masks = np.stack(face_masks)
        lmk_masks = np.stack(lmk_masks)

        return {
            'image': torch.from_numpy(imgs).float(), # (n, 3, 224, 224)
            'lmk_2d': torch.from_numpy(lmk_2d).float(), # (n, 68, 3)
            'img_mask': torch.from_numpy(face_masks).float(), # (n, 224, 224)
            'lmk_mask': torch.from_numpy(lmk_masks).float(),   # (n, 68)
            'audio_emb': self.audio_emb[start_id:start_id + self.K]  # (N, 768)
        }

if __name__ == "__main__":
    video_processor = VideoProcessor()
    output = video_processor.run('videos/justin.mp4')
    for k in output:
        if k == 'frame_id':
            print(f'{k}:', len(output[k]))
        else:
            print(f'{k}:', output[k].shape)