""" Test emotion reconstruction by computing emotion classification accuracy of the rendered sequences
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
from model.emo_rec import EmoRec
from munch import Munch
import h5py
from torch.utils.data import DataLoader, Dataset
from scipy import stats

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# emotion id correpondance
emotion_to_id = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "surprised": 3,
    "fear": 4,
    "disgusted": 5,
    "angry": 6,
    "contempt": 7
}

id_to_emotion = {
    0 : 'neutral',
    1: "happy",
    2: "sad",
    3: "surprised",
    4: "fear",
    5: "disgusted",
    6: "angry",
    7: "contempt",
}

class TestRAVDESSDataset(Dataset):
    def __init__(
        self,
        dataset_name, 
        dataset_path,
        rec_path,
        split_data,
        model_type='diffusion'
    ):
        self.split_data = split_data
        self.rec_path = rec_path
        self.dataset = dataset_name
        self.model_type = model_type

        # paths to processed folder
        self.processed_folder = os.path.join(dataset_path, dataset_name, 'processed')
        self.cropped_landmark_folder = os.path.join(self.processed_folder,'cropped_landmarks_mediapipe')
        self.emoca_rec_folder = os.path.join(self.processed_folder, 'EMOCA_reconstruction')

        # Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
        self.id_to_emotion = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fear',
            '07': 'disgusted',
            '08': 'surprised'
        }
    
    def __len__(self):
        return len(self.split_data)

    def _get_emotion_name(self, motion_id):
        vocal, emotion, intensity, sent, rep, sbj = motion_id.split('-')
        return self.id_to_emotion[emotion]

    def _get_emica_codes(self, motion_id):
        fname = motion_id + '.npy'
        rec_path = os.path.join(self.emoca_rec_folder, fname)
        rec_dict = np.load(rec_path, allow_pickle=True)[()]

        rec_dict['global_pose'] = rec_dict['pose'][:,:3]
        rec_dict['jaw'] = rec_dict['pose'][:,3:]
        rec_dict.pop('pose', None)
        for key in rec_dict:
            rec_dict[key] = torch.from_numpy(rec_dict[key]).float()

        return rec_dict

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
        motion_path = self.split_data[idx]
        emotion_name = self._get_emotion_name(motion_path)

        if self.model_type == 'emica':
            rec_dict = self._get_emica_codes(motion_path)
        
        elif self.model_type in ['diffusion', 'emote']:
            rec_dict = self._get_emica_codes(motion_path)
            new_dict = self._get_diffusion_reconstruction(motion_path)
            if new_dict is None:
                return None, None
            rec_dict['exp'] = new_dict['exp']
            rec_dict['jaw'] = new_dict['jaw']
            seqlen = min(rec_dict['exp'].shape[0], rec_dict['global_pose'].shape[0])
            for key in rec_dict:
                rec_dict[key] = rec_dict[key][:seqlen]
        elif self.model_type in ['emoca', 'deca', 'spectre']:
            rec_dict = self._get_emica_codes(motion_path)
            new_dict = self._get_emoca_reconstruction(motion_path)
            if new_dict is None:
                return None, None
            rec_dict['exp'] = new_dict['exp']
            rec_dict['jaw'] = new_dict['jaw']
        else:
            raise ValueError(f"{self.model_type} not supported!")
        
        if rec_dict is None:
            return None, None
    
        return rec_dict, emotion_name

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

        code_dict['shape'] = code_dict['shape'][:,:self.n_shape]
        code_dict['exp'] = code_dict['exp'][:,:self.n_exp]

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
            print(f"skip {sample_path}, is None")
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
        emotion_name = motion_path.split('/')[-3]
        seqlen = int(seqlen)
        if self.model_type == 'diffusion':
            rec_dict = self._get_diffusion_reconstruction(motion_path)
        elif self.model_type in ['deca', 'emoca', 'spectre']:
            rec_dict = self._get_emoca_reconstruction(motion_path)
        if rec_dict is None:
            return None, None
        emica_rec = self._get_emica_codes(motion_path)
        
        rec_dict['shape'] = emica_rec['shape']
        rec_dict['global_pose'] = emica_rec['global_pose']
        
        return rec_dict, emotion_name


class EmoRecLoss:
    
    def __init__(self, config, test_data, device='cuda'):
        
        self.config = config
        self.device = device
        self.test_data = test_data
        self.original_data_folder = 'dataset/mead_25fps/original_data'
        # IO setups
                        
        # name of the tested motion sequence
        self.output_folder = self.config.output_folder
        
        logger.add(os.path.join(self.output_folder, f'test_mead_emotion_{config.model_type}.log'))
        logger.info(f"Using device {self.device}.")
        
        self.load_emoNet()

    def load_emoNet(self):
        f = open('pretrained/EMOCA-emorec/cfg.yaml')
        cfg = Munch.fromYAML(f)
        self.emoNet = EmoRec(cfg)
        ckpt_path = 'pretrained/EMOCA-emorec/checkpoints/deca-epoch=00-val_loss_total/dataloader_idx_0=1.38192022.ckpt'
        self.emoNet.load_from_ckpt(ckpt_path)
        self.emoNet.to(self.device)
        self.emoNet.eval()
            
    def eval_emotion_accuracy(self):
        emo_acc = defaultdict(list) # store emotion classification accuracy for each motion
        emo_acc_top3 = defaultdict(list)
        num_test_motions = len(self.test_data)
        print(f"len test data", num_test_motions)

        for i in tqdm(range(num_test_motions)):

            rec, emotion_name = self.test_data[i]
            if emotion_name is None:
                continue
            emotion_label = emotion_to_id[emotion_name]
            for k in rec:
                rec[k] = rec[k].to(self.device)
                # emica_rec[k] = emica_rec[k].to(self.device)
            
            with torch.no_grad():
                emotion_class_rec = self.emoNet(rec)['expr_classification'].cpu().numpy()  # (bs,8)
                # emotion_class_emica = self.emoNet(emica_rec)['expr_class'].cpu().numpy()    # (bs,)
            top_1_pred_label = np.argmax(emotion_class_rec, axis=-1)

            motion_emo_acc = np.sum(emotion_label == top_1_pred_label) / len(top_1_pred_label)
            emo_acc[emotion_name].append(motion_emo_acc)

            top_3_pred_label = np.argpartition(emotion_class_rec, -3, axis=-1)[:,-3:]
            video_emo_acc_top3 = []
            for pred_label in top_3_pred_label:
                top_3_acc = True if emotion_label in pred_label else False
                video_emo_acc_top3.append(top_3_acc)
            video_emo_acc_top3 = np.array(video_emo_acc_top3)
            
            emo_acc_top3[emotion_name].append(np.sum(video_emo_acc_top3) / len(video_emo_acc_top3))

        logger.info(f"========Top1 Emotion accuracy for each emotion class: ===========")
        num_seq = 0
        all_acc = 0.
        for k in emo_acc:
            logger.info(f"{k}: {np.mean(emo_acc[k])}")
            num_seq += len(emo_acc[k])
            all_acc += np.sum(emo_acc[k])
        logger.info(f"========Top1 Emotion accuracy across all motion sequences: {all_acc / num_seq}")

        logger.info(f"========Top3 Emotion accuracy for each emotion class: ===========")
        num_seq = 0
        all_acc = 0.
        for k in emo_acc_top3:
            logger.info(f"{k}: {np.mean(emo_acc_top3[k])}")
            num_seq += len(emo_acc_top3[k])
            all_acc += np.sum(emo_acc_top3[k])
        logger.info(f"========Top3 Emotion accuracy across all motion sequences: {all_acc / num_seq}")

def main():
    # sample use:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, help='folder to store diffusion sample.', required=True)
    parser.add_argument('--split', type=str, help='mead split for evaluation.', default='test')
    parser.add_argument('--dataset', type=str, default='mead_25fps', help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='./dataset',help='dataset name')
    parser.add_argument('--rec_folder', type=str, default='reconstruction', required=True, help='folder to store reconstruction results.')
    parser.add_argument('--model_type', type=str, default='diffusion', required=True, help='should be in [diffusion, deca, emoca, spectre]')

    args = parser.parse_args()
    rec_folder = os.path.join(args.output_folder, args.rec_folder)

    print("loading test data...")

    if args.dataset == 'mead_25fps':
        test_video_list = load_test_data(
            args.dataset, 
            args.dataset_path, 
            args.split)

        print(f"number of test sequences: {len(test_video_list)}")
            
        test_dataset = TestMeadDataset(
            args.dataset,
            args.dataset_path,
            rec_folder,
            test_video_list,
            n_shape=100,
            n_exp=50,
            model_type=args.model_type
        )
    elif args.dataset == 'RAVDESS':
        from data_loaders.dataloader_RAVDESS import load_RAVDESS_test_data

        # leave out calm class ('02)
        test_video_list = load_RAVDESS_test_data(
            args.dataset, 
            args.dataset_path,
            emotion_list=['01','03','04','05','06','07','08']
        )
        print(f"number of test sequences: {len(test_video_list)}")

        

        test_dataset = TestRAVDESSDataset(
            args.dataset,
            args.dataset_path,
            rec_folder,
            test_video_list,
            args.model_type
        )

    

    motion_tracker = EmoRecLoss(args, test_dataset, 'cuda')
    
    motion_tracker.eval_emotion_accuracy()

if __name__ == "__main__":
    main()
