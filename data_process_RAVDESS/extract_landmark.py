#!/usr/bin/env python

import argparse
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm 
import glob
from skimage.transform import estimate_transform, warp, resize, rescale
import face_alignment 
import cv2 
import h5py
from collections import defaultdict

# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
import sys
sys.path.append('./')
from utils.data_util import linear_interpolate_landmarks, point2bbox, point2transform
from utils.mediapipe_landmark_detection import MediapipeDetector

face_detector = MediapipeDetector()
dataset_path = 'dataset/RAVDESS'
video25_folder_path = os.path.join(dataset_path, 'video_25fps')
cropped_video_folder_path = os.path.join('dataset/RAVDESS/processed', 'cropped_videos')
cropped_landmarks_folder_path = os.path.join('dataset/RAVDESS/processed', 'cropped_landmarks_mediapipe')
if not os.path.exists(cropped_video_folder_path):
    os.makedirs(cropped_video_folder_path)
if not os.path.exists(cropped_landmarks_folder_path):
    os.makedirs(cropped_landmarks_folder_path)

def landmark_detection(video_path):
    lmks = []
    null_face_cnt = 0
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print(f"{video_path} open error!")
        exit(1)
    while True:
        _, frame = video.read()
        if frame is None:
            break
        lmk_2d = face_detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if lmk_2d is None:
            null_face_cnt += 1
        lmks.append(lmk_2d)
    video.release()
    # linear interpolate the missing landmarks
    valid_frames_idx, landmarks = linear_interpolate_landmarks(lmks)
    landmarks = np.stack(landmarks)[:,:,:2]   # (n, V, 2)
    valid_frames_idx = valid_frames_idx
    print(f"There are {null_face_cnt} frames detected null faces.")
    return landmarks, valid_frames_idx

def warp_image_from_lmk(landmarks, img, scale=1.35, image_size=224):    
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0 
        center_y = bottom - (bottom - top) / 2.0
        center = np.array([center_x, center_y])

        size = int(old_size * scale)

        tform = point2transform(center, size, image_size, image_size)
        output_shape = (image_size, image_size)
        dst_image = warp(img, tform.inverse, output_shape=output_shape, order=3)
        dst_landmarks = tform(landmarks[:, :2])

        return dst_image, dst_landmarks

def process_one_video(video, image_size=224):

    cropped_video_path = os.path.join(cropped_video_folder_path, video.name)
    lmk_path = os.path.join(cropped_landmarks_folder_path, video.name)
    if os.path.exists(cropped_video_path):
        print(f"{cropped_video_path} already exists!")
        return

    landmarks, valid_frames_idx = landmark_detection(video.path)

    video_cap = cv2.VideoCapture()
    if not video_cap.open(video.path):
        logger.error(f"Cannot open {video.path}!")
        exit(1)

    lmk_list = []
    frame_list = []
    frame_id = 0
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
        dst_image, dst_landmarks = warp_image_from_lmk(landmarks[frame_id], frame)

        # normalize landmarks to [-1, 1]
        dst_landmarks = dst_landmarks / image_size * 2 - 1
        lmk_list.append(dst_landmarks)  # (V, 2)

        frame_list.append(dst_image)    # float in BGR
        frame_id += 1
    video_cap.release()
    frame_list = np.stack(frame_list)   # (n, 224, 224, 3)
    frame_list = (frame_list * 255.).astype(np.uint8)
    lmk_list = np.stack(lmk_list)   # (n, V, 2)
    print(lmk_list.shape)
    print(frame_list.shape)

    # save lmk
    np.save(lmk_path, lmk_list)

    # write cropped frame to video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        cropped_video_path, fourcc, 25, 
        (image_size, image_size))
    
    for frame in frame_list:
        writer.write(frame)
    writer.release()

for video in os.scandir(video25_folder_path):
    process_one_video(video)


    


