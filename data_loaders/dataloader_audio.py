import glob
import os

import torch
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from skimage.transform import estimate_transform, warp
from collections import defaultdict
import random
from utils import dataset_setting
from utils import utils_transform

class TrainDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        split_data,
        input_motion_length=120,
        train_dataset_repeat_times=1,
        fps=30
    ):
        self.split_data = split_data
        self.fps = fps
        self.data_fps_original = 60 if dataset_name != 'multiface' else 30
        skip_frames = self.data_fps_original // self.fps
        self.input_motion_length = input_motion_length * skip_frames
        
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization
        self.input_motion_length = input_motion_length
        self.occlusion_mask_prob = occlusion_mask_prob
       
    def __len__(self):
        return len(self.split_data['img_folders']) * self.train_dataset_repeat_times
    
    def crop_face(self, landmarks, scale=1.0):
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])

        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], 
                            [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        return tform

    def get_occlusion_mask(self, mask_array):
        # add random occlusion mask
        add_mask = torch.bernoulli(torch.ones(1) * self.occlusion_mask_prob)[0]
        if not add_mask:
            return mask_array

        num_frames = mask_array.shape[0]

        # select occlusion type
        occlusion_type = torch.randint(low=0, high=3, size=(1,))[0]

        if occlusion_type == 0:
            # occlude fixed region: top left coords and w, h of occlusion rectangle
            x, y = torch.randint(low=10, high=200, size=(2,)) 
            dx, dy = torch.randint(low=20, high=112, size=(2,))    
            mask_array[:, y:y+dy, x:x+dx] = 0
        elif occlusion_type == 1:
            # occlude random regions for each frame
            mask_bbx = torch.randint(low=20, high=200, size=(num_frames,4)) 
            for i in range(num_frames):
                mask_array[i, mask_bbx[i,0]:mask_bbx[i,1], mask_bbx[i,2]:mask_bbx[i,3]] = 0
        else:
            # occlude random num of frames
            occluded_frame_ids = torch.randint(low=0, high=num_frames, size=(num_frames // 2,))
            mask_array[occluded_frame_ids] = 0

        return mask_array

    def get_lmk_mask(self, lmk2d, img_mask):
        lmk_mask = []
        pix_pos = ((lmk2d + 1) * self.image_size / 2).long()
        pix_pos = torch.clamp(pix_pos, min=0, max=self.image_size-1)
        for i in range(lmk2d.shape[0]):
            lmk_mask.append(img_mask[i, pix_pos[i, :, 1], pix_pos[i, :, 0]])
        return torch.stack(lmk_mask)

    def image_augment(self, image):
        # image augmentation
        add_augmentation = torch.bernoulli(torch.ones(1) * 0.7)[0]
        if add_augmentation:
            transf_order, b, c, s, h = transforms.ColorJitter.get_params(
                brightness=(1, 2),
                contrast=(1, 1.5),
                saturation=(1, 1),
                hue=(-0.1,0.1))
            
            image = random_color_jitter_to_video(image, b, c, s, h, transf_order)

            sigma = transforms.GaussianBlur.get_params(sigma_min=0.1, sigma_max=2)
            blur_transf = lambda img: F.gaussian_blur(img, kernel_size=9,sigma=sigma)
            image = blur_transf(image)
        return image

    def __getitem__(self, idx):
        id = idx % len(self.split_data['img_folders'])
        processed_data = torch.load(self.split_data['processed_paths'][id])
        img_folder = self.split_data['img_folders'][id]
        dataset = self.split_data['dataset'][id]
        dataset_info = self.dataset_info[dataset]

        seqlen = processed_data['lmk_2d'].shape[0]

        if seqlen == dataset_info['input_motion_length']: 
            start_id = 0
        else:
            start_id = torch.randint(0, int(seqlen - dataset_info['input_motion_length']), (1,))[0]     # random crop a motion seq

        # randomly sample rotation params
        angle = np.random.uniform(self.rot_angle[0], self.rot_angle[1])
        M = cv2.getRotationMatrix2D((112, 112), angle, scale=1.0)

        if dataset == 'multiface':
            # keys of processed_data: ['image', 'img_mask', 'lmk_2d', 'audio_emb', 'shape', 'tex', 'cam', 'light']:
            image_array = processed_data['image'][start_id:start_id + dataset_info['input_motion_length']:dataset_info['skip_frames']]
            image_array = (image_array.permute(0,2,3,1).numpy() * 255).astype(np.uint8)
            kpt_array = processed_data['lmk_2d'][start_id:start_id + dataset_info['input_motion_length']:dataset_info['skip_frames']].numpy() * 112 + 112
            img_mask_array = (processed_data['img_mask'][start_id:start_id + dataset_info['input_motion_length']:dataset_info['skip_frames']].numpy() * 255).astype(np.uint8)
            
            audio_emb = processed_data['audio_emb'][start_id:start_id + dataset_info['input_motion_length']:dataset_info['skip_frames']]
            shape = processed_data['shape'][start_id:start_id + dataset_info['input_motion_length']:dataset_info['skip_frames']]
            pose = processed_data['pose'][start_id:start_id + dataset_info['input_motion_length']:dataset_info['skip_frames']]
            exp = processed_data['exp'][start_id:start_id + dataset_info['input_motion_length']:dataset_info['skip_frames']]
            jaw_6d = utils_transform.aa2sixd(pose[...,3:])
            target = torch.cat([jaw_6d, exp], dim=-1)

            # read images as input 
            images_list = []
            kpt_list = []
            img_mask_list = []
            for i in range(image_array.shape[0]):
                frame = image_array[i]
                kpt = kpt_array[i]
                img_mask = img_mask_array[i]
                frame = cv2.warpAffine(frame, M, (self.image_size, self.image_size)) / 255
                img_mask = cv2.warpAffine(img_mask, M, (self.image_size, self.image_size)) / 255
                kpt_homo = np.concatenate([kpt[...,:2], np.ones((68, 1))], axis=-1)
                kpt = M.dot(kpt_homo.T).T
                kpt = kpt/self.image_size * 2  - 1
                images_list.append(frame.transpose(2,0,1)) # (3, 224, 224)
                kpt_list.append(kpt)
                img_mask_list.append(img_mask)
            image_array = torch.from_numpy(np.stack(images_list)).type(dtype = torch.float32) 
            kpt_array = torch.from_numpy(np.stack(kpt_list)).type(dtype = torch.float32) 
            img_mask = torch.from_numpy(np.stack(img_mask_list)).type(dtype = torch.float32) 
            
        else:
            motion_id = os.path.split(img_folder)[-1]
            lmk_2d = processed_data['lmk_2d'][start_id:start_id + dataset_info['input_motion_length']:dataset_info['skip_frames']]  # (n, 68, 2)
            target = processed_data['target'][start_id:start_id + dataset_info['input_motion_length']:dataset_info['skip_frames']]
            shape = processed_data['shape'][start_id:start_id + dataset_info['input_motion_length']:dataset_info['skip_frames']]
            if 'audio_emb' in processed_data:
                audio_emb = processed_data['audio_emb'][start_id:start_id + dataset_info['input_motion_length']:dataset_info['skip_frames']]
            else:
                audio_emb = torch.zeros((target.shape[0], 768))
        
            # read images as input 
            images_list = []
            kpt_list = []
            
            frame_id = processed_data['frame_id'][start_id:start_id + dataset_info['input_motion_length']:dataset_info['skip_frames']]
            for i, fid in enumerate(frame_id):
                frame_id = "%06d"%(fid)
                kpt = lmk_2d[i].numpy()
                
                img_path = os.path.join(img_folder, f"{motion_id}.{frame_id}.{self.cam_id}.jpg")
                if not os.path.exists(img_path):
                    frame = np.zeros((dataset_info['original_image_size'][0], dataset_info['original_image_size'][1], 3)).astype(np.uint8)
                else:
                    frame = cv2.imread(img_path)
                    if frame is None:
                        frame = np.zeros((dataset_info['original_image_size'][0], dataset_info['original_image_size'][1], 3)).astype(np.uint8)

                # apply random rotation to both lmks and frame
                frame = cv2.warpAffine(frame, M, (dataset_info['original_image_size'][1], dataset_info['original_image_size'][0]))
                kpt_homo = np.concatenate([kpt[...,:2], np.ones((68, 1))], axis=-1)
                kpt = M.dot(kpt_homo.T).T

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tform = self.crop_face(kpt, self.scale) 
                cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T
                cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1
                cropped_image = warp(frame, tform.inverse, output_shape=(self.image_size, self.image_size))

                images_list.append(cropped_image.transpose(2,0,1)) # (3, 224, 224)
                kpt_list.append(cropped_kpt)

            image_array = torch.from_numpy(np.stack(images_list)).type(dtype = torch.float32) 
            kpt_array = torch.from_numpy(np.stack(kpt_list)).type(dtype = torch.float32) 
            img_mask = torch.ones((image_array.shape[0], self.image_size, self.image_size))

        # apply random color jitter and gaussian blur
        image_array = self.image_augment(image_array)
        # get random occlusion mask 
        img_mask = self.get_occlusion_mask(img_mask)
        lmk_mask = self.get_lmk_mask(kpt_array, img_mask)

        return {
            'image': image_array, # (n, 3, 224, 224)
            'lmk_2d': kpt_array[...,:2], # (n, 68, 2)
            'img_mask': img_mask.float(), # (n, 224, 224)
            'lmk_mask': lmk_mask.float(),   # (n, 68)
            'audio_emb': audio_emb.float(),
            'shape': shape.float(),
            'target': target.float()
        }