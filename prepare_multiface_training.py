import os  
from glob import glob 
import numpy as np
import cv2 
from tqdm import tqdm
import argparse
from utils import dataset_setting
import sys
from configs.config import get_cfg_defaults
import librosa
from transformers import Wav2Vec2Processor
from utils import dataset_setting
from model.deca import EMOCA
from model.wav2vec import Wav2Vec2Model
from skimage.transform import estimate_transform, warp
from utils.data_util import crop_np
from mmseg.apis import inference_model, init_model
from collections import defaultdict
import torch


scale_facter = 1 / 1000.0 # convert mm to m
h, w = 2048, 1334
image_size = 224
scale = 1.5



# compute the landmarks and the distance between two points for each image 
def create_training_data(path_to_dataset, device, model_cfg):
    """prepare data dict for training the model

    Returns:
        data_dict: dict of training data, one per motion sequence
            image: torch.Size([bs, 3, 224, 224])
            lmk_2d: torch.Size([bs, 68, 3])
            img_mask: torch.Size([bs, 224, 224])
            audio_emb: torch.Size([bs, 768])
            shape: torch.Size([bs, 100])
            tex: torch.Size([bs, 50])
            exp: torch.Size([bs, 50])
            pose: torch.Size([bs, 6])
            cam: torch.Size([bs, 3])
            light: torch.Size([bs, 9, 3])
    """
    
    sbjs = [x.path for x in os.scandir(path_to_dataset) if x.is_dir()]

    # create pretrained model
    # emoca = EMOCA(model_cfg)
    # emoca.to(device)
    
    # wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    # wav2vec.feature_extractor._freeze_parameters()
    # wav2vec.to(device)
    # audio processor
    audio_processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h")  # HuBERT uses the processor of Wav2Vec 2.0
    
    # init facial segmentator
    # config_file = 'face_segmentation/deeplabv3plus_r101_512x512_face-occlusion.py'
    # checkpoint_file = 'face_segmentation/deeplabv3plus_r101_512x512_face-occlusion-93ec6695.pth'
    # face_segment = init_model(config_file, checkpoint_file, device='cuda:0')
    # print_shape = True
    for sbj in sbjs:
        to_folder = os.path.join(sbj, "processed_data")
        if not os.path.exists(to_folder):
            os.makedirs(to_folder)
        for motion in os.scandir(os.path.join(sbj, "images")):
            out_fname = os.path.join(to_folder, motion.name + ".pt")
            # if os.path.exists(out_fname):
            #     continue
            print(f"Processing {motion.name}")
            data_dict = torch.load(out_fname)
            
            # data = np.load(os.path.join(to_folder, motion.name + '.npy'), allow_pickle=True)[()]
            # lmk68 = data['lmk68']
            # kpt_list = []
            # mask_list = []
            # image_list = []
            # emoca_codes = defaultdict(list)
            
            
            # for i, img_path in enumerate(sorted(glob(os.path.join(motion.path, '*/*.png')))):
            #     frame = cv2.imread(img_path)
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            #     kpt = lmk68[i]
            #     tform = crop_np(kpt, trans_scale=0, scale=scale, image_size=image_size)
            #     cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T
            #     cropped_kpt[:,:2] = cropped_kpt[:,:2]/image_size * 2  - 1
            #     cropped_image = warp(frame, tform.inverse, output_shape=(image_size, image_size))   # (224, 224, 3) in rgb
            #     seg_result = inference_model(face_segment, cropped_image[:,:,[2,1,0]])
            #     seg_mask = np.asanyarray(seg_result.pred_sem_seg.values()[0].to('cpu'))[0]
                
            #     image_input = cropped_image.transpose(2, 0, 1)  # (3, 223, 223)
                
            #     kpt_list.append(cropped_kpt)
            #     image_list.append(image_input)
            #     mask_list.append(seg_mask)
                
            #     image_input = torch.from_numpy(image_input).float().unsqueeze(0) # (1, 3, 224, 224)
            #     with torch.no_grad():
            #         emoca_code = emoca(image_input.to(device))
                
            #     for k in emoca_code:
            #         emoca_codes[k].append(emoca_code[k].cpu().squeeze(0))
            
            # kpt_list = np.stack(kpt_list)
            # image_list = np.stack(image_list)
            # mask_list = np.stack(mask_list)
            # for k in emoca_codes:
            #     emoca_codes[k] = torch.stack(emoca_codes[k])
                
           
            if 'SEN' in motion.name:
                audio_path = os.path.join(sbj, "audio", f"{motion.name}.wav")
                speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
                audio_values = audio_processor(
                    speech_array, return_tensors='pt', padding="longest",sampling_rate=sampling_rate).input_values.squeeze()
                data_dict['audio_input'] = audio_values
            
            # audio_input = data['audio']
            #     with torch.no_grad():
            #         audio_emb = wav2vec(audio_input, frame_num = kpt_list.shape[0]).last_hidden_state.squeeze(0).cpu()
            # else:
            #     audio_emb = torch.zeros((kpt_list.shape[0], 768))
            
            # data_dict = {
            #     'image': torch.from_numpy(image_list).float(),
            #     'lmk_2d': torch.from_numpy(kpt_list).float(),
            #     'img_mask': torch.from_numpy(mask_list).float(),
            #     'audio_emb': audio_emb,
            # }
            
            # data_dict.update(emoca_codes)
            
            # if print_shape:
            #     for k in data_dict:
            #         print(f"shape of {k}: {data_dict[k].shape}")
            #     print_shape = False
            
            torch.save(data_dict, out_fname)
                
                
                

if __name__ == '__main__':
    path_to_dataset = 'dataset/multiface'
    device = 'cpu'
    model_cfg = get_cfg_defaults().model
    create_training_data(path_to_dataset, device, model_cfg)