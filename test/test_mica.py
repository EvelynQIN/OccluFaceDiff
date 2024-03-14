# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import argparse
import os
import random
from glob import glob
from pathlib import Path
import yaml 
from utils.config import Struct 
from utils.model_util import find_model_using_name
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import trimesh
# from insightface.app.common import Face
# from insightface.utils import face_align

from skimage.io import imread
from tqdm import tqdm


# from datasets.creation.util import get_arcface_input, get_center, draw_on
# from utils import util
# from utils.landmark_detector import LandmarksDetector, detectors


def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)

    cudnn.deterministic = True
    cudnn.benchmark = False

# def process(args, app, image_size=224, draw_bbox=False):
#     dst = Path(args.a)
#     dst.mkdir(parents=True, exist_ok=True)
#     processes = []
#     image_paths = sorted(glob(args.i + '/*.*'))
#     for image_path in tqdm(image_paths):
#         name = Path(image_path).stem
#         img = cv2.imread(image_path)
#         bboxes, kpss = app.detect(img)
#         if bboxes.shape[0] == 0:
#             logger.error(f'[ERROR] Face not detected for {image_path}')
#             continue
#         i = get_center(bboxes, img)
#         bbox = bboxes[i, 0:4]
#         det_score = bboxes[i, 4]
#         kps = None
#         if kpss is not None:
#             kps = kpss[i]
#         face = Face(bbox=bbox, kps=kps, det_score=det_score)
#         blob, aimg = get_arcface_input(face, img)
#         file = str(Path(dst, name))
#         np.save(file, blob)
#         processes.append(file + '.npy')
#         cv2.imwrite(file + '.jpg', face_align.norm_crop(img, landmark=face.kps, image_size=image_size))
#         if draw_bbox:
#             dimg = draw_on(img, [face])
#             cv2.imwrite(file + '_bbox.jpg', dimg)

#     return processes


# def to_batch(path):
#     src = path.replace('npy', 'jpg')
#     if not os.path.exists(src):
#         src = path.replace('npy', 'png')

#     image = imread(src)[:, :, :3]
#     image = image / 255.
#     image = cv2.resize(image, (224, 224)).transpose(2, 0, 1)
#     image = torch.tensor(image).cuda()[None]

#     arcface = np.load(path)
#     arcface = torch.tensor(arcface).cuda()[None]

#     return image, arcface

def get_arcface_input():
    img = torch.rand(size=(1, 16, 3, 112, 112)).to('cuda')
    img_mask = torch.randint(low=0, high=1, size=(1, 16)).bool().to('cuda')
    return img, img_mask

def main():
    device = 'cuda:0'
    cfg_path = 'configs/mica.yaml'
    with open(cfg_path, 'r') as infile:
        cfg = yaml.safe_load(infile)
    args = Struct(**cfg)

    mica = find_model_using_name(model_dir='model', model_name=args.name)(args)
    mica.load_model()
    mica.to(device)

    img_arr, img_mask = get_arcface_input()
    print(img_arr.shape)
    identity_code = mica.encode(img_arr, img_mask)
    print(identity_code.shape)
    pred_shape = mica.decode(identity_code)
    print(pred_shape.shape)
    # mica.eval()

    # Path(args.o).mkdir(exist_ok=True, parents=True)

    # app = LandmarksDetector(model=detectors.RETINAFACE)

    # with torch.no_grad():
    #     logger.info(f'Processing has started...')
    #     paths = process(args, app, draw_bbox=False)
    #     for path in tqdm(paths):
    #         name = Path(path).stem
    #         images, arcface = to_batch(path)
    #         codedict = mica.encode(images, arcface)
    #         opdict = mica.decode(codedict)
    #         meshes = opdict['pred_canonical_shape_vertices']
    #         code = opdict['pred_shape_code']
    #         lmk = mica.flame.compute_landmarks(meshes)

    #         mesh = meshes[0]
    #         landmark_51 = lmk[0, 17:]
    #         landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

    #         dst = Path(args.o, name)
    #         dst.mkdir(parents=True, exist_ok=True)
    #         trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.ply')  # save in millimeters
    #         trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.obj')
    #         np.save(f'{dst}/identity', code[0].cpu().numpy())
    #         np.save(f'{dst}/kpt7', landmark_7.cpu().numpy() * 1000.0)
    #         np.save(f'{dst}/kpt68', lmk.cpu().numpy() * 1000.0)

    #     logger.info(f'Processing finished. Results has been saved in {args.o}')


if __name__ == '__main__':
    deterministic(42)
    main()