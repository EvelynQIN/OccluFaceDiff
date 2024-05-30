""" Test emotion reconstruction by computing video emotion classification accuracy from the predicted flame params
"""
import os
import random
import numpy as np
from tqdm import tqdm
import cv2
from enum import Enum
import os.path
from glob import glob
from pathlib import Path
import subprocess
from loguru import logger
from time import time
from matplotlib import cm

from utils import utils_transform, utils_visualize
from utils.metrics import get_metric_function
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import test_args
from model.FLAME import FLAME_mediapipe
from configs.classifier_config import get_cfg_defaults as video_emo_classifier_cfg
from configs.config import get_cfg_defaults
from utils import dataset_setting
from utils.renderer import SRenderY
from utils.data_util import batch_orth_proj, face_vertices
from pathlib import Path
from collections import defaultdict
import argparse

import torch
import torchvision.transforms.functional as F_v
import torch.nn.functional as F
from pytorch3d.io import load_obj
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams
from pytorch3d.structures import Meshes
# pretrained

import imageio
from skimage.io import imread
from data_loaders.dataloader_MEAD_flint import load_test_data
import ffmpeg
import pickle
from model.video_emotion_classifier import VideoEmotionClassifier
from munch import Munch
import h5py
from torch.utils.data import DataLoader, Dataset
from scipy import stats
from data_loaders.dataloader_MEAD_classifer import video_id_2_emotion_class

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# emotion id correpondance
# emotion_to_id = {
#     "neutral": 0,
#     "happy": 1,
#     "sad": 2,
#     "surprised": 3,
#     "fear": 4,
#     "disgusted": 5,
#     "angry": 6,
#     "contempt": 7
# }

# id_to_emotion = {
#     0 : 'neutral',
#     1: "happy",
#     2: "sad",
#     3: "surprised",
#     4: "fear",
#     5: "disgusted",
#     6: "angry",
#     7: "contempt",
# }

def emotion_id_2_emotion_name():
    
    emotions = ['neutral', 'happy', 'sad', 'surprised', 'fear', 'disgusted', 'angry', 'contempt']
    levels = ['level_1', 'level_2', 'level_3']
    class_id = 0
    emotion_id_2_emotion_name_mapping = {}
    for emotion in emotions:
        for level in levels:
            if emotion == 'neutral' and level != 'level_1':
                continue
            emotion_id_2_emotion_name_mapping[class_id] = emotion
            class_id += 1
    return emotion_id_2_emotion_name_mapping

class TestMeadDataset(Dataset):
    def __init__(
        self,
        dataset_name, # list of dataset names
        dataset_path,
        rec_path,
        split_data,
        n_shape=100,
        n_exp=50,
        model_type='diffusion'
    ):
        self.split_data = split_data
        self.rec_path = rec_path
        self.dataset = dataset_name
        self.n_shape = n_shape 
        self.n_exp = n_exp
        self.model_type = model_type
        self.video_id_2_label = video_id_2_emotion_class()

        # paths to processed folder
        self.processed_folder = os.path.join(dataset_path, dataset_name, 'processed')
        self.reconstruction_folder = os.path.join(self.processed_folder, 'reconstructions/EMICA-MEAD_flame2020')

    def __len__(self):
        return len(self.split_data)

    def _get_emica_codes(self, motion_path):
        code_dict = {}

        with h5py.File(os.path.join(self.reconstruction_folder, motion_path, 'shape_pose_cam.hdf5'), "r") as f:
        # cam : (1, n, 3)
        # exp : (1, n, 100)
        # global_pose : (1, n, 3)
        # jaw : (1, n, 3)
        # shape : (1, n, 300)
            for k in f.keys():
                code_dict[k] = torch.from_numpy(f[k][0]).float()

        code_dict['shape'] = code_dict['shape'][:,:100]
        code_dict['exp'] = code_dict['exp'][:,:50]

        return code_dict 

    def _get_diffusion_reconstruction(self, motion_path):
        sample_path = os.path.join(self.rec_path, f"{motion_path}.npy")
        if not os.path.exists(sample_path):
            return None
        diffusion_sample = np.load(sample_path, allow_pickle=True)[()]
        diffusion_codes = {}
        diffusion_codes['jaw'] = torch.from_numpy(diffusion_sample[:,100:]).float()
        diffusion_codes['exp'] = torch.from_numpy(diffusion_sample[:,:50]).float()
        return diffusion_codes

    def _get_emoca_reconstruction(self, motion_path):
        sample_path = os.path.join(self.rec_path, f"{motion_path}.npy")
        if not os.path.exists(sample_path):
            return None
        emoca_codes = np.load(sample_path, allow_pickle=True)[()]

        rec_dict = {}
        rec_dict['jaw'] = torch.from_numpy(emoca_codes['pose'][:,3:]).float()
        if self.model_type in ['deca', 'spectre']:
            rec_dict['exp'] = torch.from_numpy(emoca_codes['exp']).float()
        else:
            rec_dict['exp'] = torch.from_numpy(emoca_codes['exp_emoca']).float()
        return rec_dict

    def __getitem__(self, idx):
        motion_path, seqlen = self.split_data[idx]

        subject, view, emotion, level, sent = motion_path.split('/')
        label = self.video_id_2_label[f"{emotion}_{level}"]
        seqlen = int(seqlen)
        if self.model_type == 'diffusion':
            rec_dict = self._get_diffusion_reconstruction(motion_path)
        elif self.model_type in ['emoca', 'deca', 'spectre']:
            rec_dict = self._get_emoca_reconstruction(motion_path)
        else:
            raise ValueError(f"{self.model_type} not supported!")
        if rec_dict is None:
            return None, None, None
        emica_rec = self._get_emica_codes(motion_path)

        rec_input = torch.cat([
            emica_rec['shape'], rec_dict['exp'], rec_dict['jaw']], dim=-1)
        
        emica_input = torch.cat([
            emica_rec['shape'], emica_rec['exp'], emica_rec['jaw']], dim=-1)
        
        return rec_input, emica_input, label


class EmoRecLoss:
    
    def __init__(self, config, test_data, device='cuda'):
        
        self.config = config
        self.device = device
        self.test_data = test_data
        self.original_data_folder = 'dataset/mead_25fps/original_data'
        # IO setups
                        
        # name of the tested motion sequence
        self.output_folder = self.config.output_folder
        
        logger.add(os.path.join(self.output_folder, f'test_mead_video_emotion_{config.model_type}.log'))
        logger.info(f"Using device {self.device}.")
        
        self.emotion_id_2_emotion_name = emotion_id_2_emotion_name()
        self.load_emoNet()

    def load_emoNet(self):
        args = video_emo_classifier_cfg()
        self.emo_classifier = VideoEmotionClassifier(
            args.input_dim,
            args.num_classes,
            args.latent_dim,
            args.heads,
            args.layers,
            args.ff_size,
            args.max_pool,
            args.dropout
        )
        ckpt_path = 'checkpoints/Emotion_Classifier_Transformer_128d_4l/model_20.pt'
        ckpt = torch.load(ckpt_path)
        self.emo_classifier.load_state_dict(ckpt)
        self.emo_classifier.to(self.device)
        self.emo_classifier.eval()
            
    def eval_emotion_accuracy(self):
        emo_acc_whole = defaultdict(list)   # 22 emotion class accuracy
        emo_acc = defaultdict(list) # 8 emotion class accuracy
        num_test_motions = len(self.test_data)

        for i in tqdm(range(num_test_motions)):
            diff_rec, emica_rec, label = self.test_data[i]
            if label is None:
                continue
            diff_rec = diff_rec.to(self.device)
            # emica_rec = emica_rec.to(self.device)
            
            with torch.no_grad():
                emotion_class_pred = self.emo_classifier.predict(diff_rec.unsqueeze(0)).cpu().numpy()[0]  # class label
                # emotion_class_pred = self.emo_classifier.predict(emica_rec.unsqueeze(0)).cpu().numpy()[0]    # (bs,)
            pred_emotion = self.emotion_id_2_emotion_name[emotion_class_pred]
            gt_emotion = self.emotion_id_2_emotion_name[label]
            
            video_emo_acc = 1 if emotion_class_pred == label else 0
            video_emo_8_class = 1 if pred_emotion == gt_emotion else 0
            emo_acc_whole[gt_emotion].append(video_emo_acc)
            emo_acc[gt_emotion].append(video_emo_8_class)

        logger.info(f"========Video Emotion accuracy for 8 emotion class: ===========")
        num_seq = 0
        all_acc = 0.
        for k in emo_acc:
            logger.info(f"{k}: {np.mean(emo_acc[k])}")
            num_seq += len(emo_acc[k])
            all_acc += np.sum(emo_acc[k])
        logger.info(f"========Video Frame Emotion accuracy across all motion sequences: {all_acc / num_seq}")

        logger.info(f"========Video Emotion accuracy for 22 emotion class: ===========")
        num_seq = 0
        all_acc = 0.
        for k in emo_acc_whole:
            logger.info(f"{k}: {np.mean(emo_acc_whole[k])}")
            num_seq += len(emo_acc_whole[k])
            all_acc += np.sum(emo_acc_whole[k])
        logger.info(f"========Video Emotion accuracy across all motion sequences: {all_acc / num_seq}")

def main():
    # sample use:
    # python3 test_video_emotion_classification.py --output_folder vis_result/EMOCA/non_occ --model_type emoca --rec_folder EMOCA_reconstruction
    # python3 test_video_emotion_classification.py --output_folder vis_result/SPECTRE/non_occ --model_type spectre --rec_folder SPECTRE_reconstruction
    # python3 test_video_emotion_classification.py --output_folder vis_result/diffusion_Transformer_768d_cat_mediapipelmk_FLINT_testsplit_largeocc/all --model_type diffusion --rec_folder diffusion_sample
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, help='folder to store diffusion sample.', required=True)
    parser.add_argument('--split', type=str, help='mead split for evaluation.', default='test')
    parser.add_argument('--dataset', type=str, default='mead_25fps', help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='./dataset',help='dataset name')
    parser.add_argument('--rec_folder', type=str, default='', required=True, help='folder to store reconstruction results.')
    parser.add_argument('--model_type', type=str, default='diffusion', required=True, help='should be in [diffusion, deca, emoca, spectre]')

    args = parser.parse_args()
    
    print("loading test data...")

    test_video_list = load_test_data(
        args.dataset, 
        args.dataset_path, 
        args.split)

    print(f"number of test sequences: {len(test_video_list)}")

    rec_folder = os.path.join(args.output_folder, args.rec_folder)
        
    test_dataset = TestMeadDataset(
        args.dataset,
        args.dataset_path,
        rec_folder,
        test_video_list,
        n_shape=100,
        n_exp=50,
        model_type=args.model_type
    )

    motion_tracker = EmoRecLoss(args, test_dataset, 'cuda')
    
    motion_tracker.eval_emotion_accuracy()

if __name__ == "__main__":
    main()
