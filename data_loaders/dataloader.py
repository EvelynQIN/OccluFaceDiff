# Copyri[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_mot[start_id:start_id + input_motion_length]ion_length][start_id:start_id + input_motion_length][start_id:start_id + input_motion_length][start_id:start_id + input_motion_targetght (c) Meta Platforms, Inc. All Rights Reserved
import glob
import os

import torch
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from skimage.transform import estimate_transform, warp
from collections import defaultdict
import random
from utils.landmark_mask import REGIONS
from utils import dataset_setting
import cv2 

class TrainDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        image_path, # list of motion path folder
        processed_path, # path to .npy file
        input_motion_length=120,
        train_dataset_repeat_times=1,
        no_normalization=True,
        occlusion_mask_prob=0.5,
        mixed_occlusion_prob=0.3,
        fps=30
    ):
        self.dataset_name = dataset_name
        self.image_path = image_path
        self.processed_path = processed_path
        self.image_size = 224 
        self.scale = 1.6
        
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization
        self.input_motion_length = input_motion_length
        self.occlusion_mask_prob = occlusion_mask_prob
        self.num_mask_regions = len(REGIONS)
        self.mask_regions = list(REGIONS.keys())
        self.mixed_occlusion_prob = mixed_occlusion_prob
        self.fps = fps
        self.skip_frames = 2 if self.fps == 30 and self.dataset_name == 'FaMoS' or self.dataset_name == 'vocaset' else 1 

    def __len__(self):
        return len(self.processed_path) * self.train_dataset_repeat_times

    def inv_transform(self, target):
        
        pass # TODO
    
    def crop_face(self, frame, landmarks, scale=1.0):
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        h, w, _ = frame.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])

        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        return tform

    def __getitem__(self, idx):
        id = idx % len(self.processed_path)
        processed_data = np.load(self.processed_path[id], allow_pickle=True)[()]
        frame_id = processed_data['frame_id']
        seqlen = len(frame_id)
        
        if self.train_dataset_repeat_times == 1:
            # do not repeat
            input_motion_length = 20 
            
        elif self.input_motion_length is None:  
            # in transformer, randomly clip a subseq
            input_motion_length = torch.randint(min(20, seqlen), min(seqlen+1, 41), (1,))[0]
        else:
            # fix motion len
            input_motion_length = self.input_motion_length 

        if seqlen == input_motion_length: 
            start_id = 0
        else:
            start_id = torch.randint(0, int(seqlen - input_motion_length), (1,))[0]     # random crop a motion seq
        if processed_data['lmk68'][start_id][0][0] == 0 and processed_data['lmk68'][start_id][0][1] == 0:
            start_id = 0
        frame_id = frame_id[start_id:start_id + input_motion_length:self.skip_frames]
        lmk68 = processed_data['lmk68'][start_id:start_id + input_motion_length:self.skip_frames]  # (n, 68, 2)
        mouth_closure_3d = processed_data['mouth_closure_3d'][start_id:start_id + input_motion_length:self.skip_frames]  # (n, 8)
        eye_closure_3d = processed_data['eye_closure_3d'][start_id:start_id + input_motion_length:self.skip_frames] # (n, 4)
        
        # read images as input 
        images_list = []
        kpt_list = []
        for i, fid in enumerate(frame_id):
            img_path = os.path.join(self.image_path[id][0], "%06d.png"%(fid))
            frame = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
            kpt = lmk68[i].copy()
            if kpt[0][0] == 0 and kpt[0][1] == 0:
                # use previous results
                images_list.append(images_list[-1])
                kpt_list.append(kpt_list[-1])

            else:
                tform = self.crop_face(frame, kpt, self.scale) 
                cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T
                cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1
                cropped_image = warp(frame, tform.inverse, output_shape=(self.image_size, self.image_size))
                images_list.append(cropped_image.transpose(2,0,1)) # (3, 224, 224)
                kpt_list.append(cropped_kpt)

        images_array = torch.from_numpy(np.stack(images_list)).type(dtype = torch.float32) # (n, 3, 224, 224) # TODO add occlusion mask to image embedding
        kpt_array = torch.from_numpy(np.stack(kpt_list)).type(dtype = torch.float32) # (n, 68, 3)
        
        return images_array, kpt_array, torch.from_numpy(mouth_closure_3d).float(), torch.from_numpy(eye_closure_3d).float()
    
class TestDataset(Dataset):
    def __init__(
        self,
        dataset,
        norm_dict,
        motions,
        no_normalization=True,
        occlusion_mask_prob=0,
    ):
        self.dataset = dataset
        self.mean = norm_dict['mean']
        self.std = norm_dict['std']
        self.motions = motions
        self.no_normalization = no_normalization
        self.occlusion_mask_prob = occlusion_mask_prob

    def __len__(self):
        return len(self.motion_path_list)

    def inv_transform(self, target):
        
        target = target * self.std["target"] + self.mean["target"]
        
        return target

    def __getitem__(self, idx):
        
        id = idx % len(self.motion_path_list)

        motion_dict =self.motions[id]
        
        seqlen = motion_dict['target'].shape[0]
        
        lmk_2d = motion_dict['lmk_2d']  # (n, 68, 2)
        lmk_3d_normed = motion_dict['lmk_3d_normed'] # (n, 68, 3)
        target = motion_dict['target'] # (n, shape300 + exp100 + rot30 + trans3)
        motion_id = os.path.split(self.motion_path_list[id])[1].split('.')[0]
        
        
        n_imgs = torch.sum(motion_dict['img_mask'])
        img_arr = motion_dict['arcface_input'] # (n_imgs, 3, 112, 112)

        # make sure there are always 4 images 
        needed_imgs = 4 - n_imgs
        if needed_imgs < 0:
            # sample 4 images with equal intervals
            img_ids = torch.arange(0, n_imgs, n_imgs // 4)[:4]
            img_arr = img_arr[img_ids]
        elif needed_imgs > 0:
            # repeat needed images
            img_arr_added_ids = torch.randint(0, n_imgs, size=(needed_imgs,))
            img_arr_repeated = motion_dict['arcface_input'][img_arr_added_ids]
            img_arr = torch.cat([img_arr, img_arr_repeated], dim=0)
        assert (not img_arr.isnan().any()) and img_arr.shape[0] == 4
            
        # Normalization 
        if not self.no_normalization:    
            lmk_3d_normed = ((lmk_3d_normed.reshape(-1, 3) - self.mean['lmk_3d_normed']) / (self.std['lmk_3d_normed'] + 1e-8)).reshape(seqlen, -1, 3)
            target = (target - self.mean['target']) / (self.std['target'] + 1e-8)
        
        # add random occlusion mask
        occlusion_mask = self.add_random_occlusion_mask(lmk_2d)  
        
        return target.float(), lmk_2d.float(), lmk_3d_normed.float(), img_arr.float(), occlusion_mask, motion_id

    def add_random_occlusion_mask(self, lmk_2d, **model_kwargs):
        input_motion_length, num_lmks = lmk_2d.shape[:2]
        occlusion_mask = torch.zeros(input_motion_length, num_lmks) # (n, v)
        add_mask = torch.bernoulli(torch.ones(1) * self.occlusion_mask_prob)[0]
        if add_mask == 0:
            return occlusion_mask
        
        # select occlusion type
        
        occlusion_type = model_kwargs.get("occlusion_type", torch.randint(low=0, high=3, size=(1,))[0])

        if occlusion_type == 0:
            # occlude fixed set of lmks
            if "occlude_lmks" in model_kwargs:
                occlude_lmks  = model_kwargs["occlude_lmks"]
            else:
                occlude_center_lmk_id = model_kwargs.get("occlude_center_lmk_id", torch.randint(low=0, high=num_lmks, size=(1,))[0])
                occlude_radius = model_kwargs.get("occlude_radius", torch.rand(1)[0] * 1.5)
                lmk_2d_dist_to_center = torch.norm(
                    lmk_2d[0] - lmk_2d[0, occlude_center_lmk_id][None],
                    2,
                    -1
                )
                occlude_lmks = lmk_2d_dist_to_center < occlude_radius
            occlusion_mask[:, occlude_lmks] = 1
        elif occlusion_type == 1:
            # occlude random set of lmks for each frame
            for i in range(input_motion_length):
                occlude_center_lmk_id = torch.randint(low=0, high=num_lmks, size=(1,))[0]
                occlude_radius = torch.rand(1)[0] * 1.5
                lmk_2d_dist_to_center = torch.norm(
                    lmk_2d[i] - lmk_2d[i, occlude_center_lmk_id][None],
                    2,
                    -1
                )
                occlude_lmks = lmk_2d_dist_to_center < occlude_radius
                occlusion_mask[i, occlude_lmks] = 1
        else:
            # occlude random num of frames
            if "occlude_frame_ids" in model_kwargs:
                occlude_frame_ids = model_kwargs["occlusion_type"]
            else:
                num_occluded_frames = torch.randint(low=1, high=input_motion_length//2, size=(1,))[0]
                occlude_frame_ids =  torch.LongTensor(random.sample(range(input_motion_length), num_occluded_frames))

            occlusion_mask[occlude_frame_ids] = 1
        return occlusion_mask

def get_path(dataset_path, dataset, split):
    if dataset == 'multiface':
        split_id = dataset_setting.multiface_split[split]
        selected_motions = [dataset_setting.multiface_motion_id[i] for i in split_id]
    subjects = [subject.path for subject in os.scandir(os.path.join(dataset_path, dataset)) if subject.is_dir()]
    image_path = []
    processed_path =[]
    for subject in subjects:
        for motion in os.scandir(os.path.join(subject, 'images')):
            if motion.name in selected_motions:
                processed_motion_path = os.path.join(subject, 'processed_data', motion.name+'.npy')
                processed = np.load(processed_motion_path, allow_pickle=True)[()]
                if processed['frame_id'].shape[0] < 20:
                    continue
                image_path.append(glob.glob(os.path.join(subject, 'images', motion.name+"/*")))
                processed_path.append(processed_motion_path)
    
    return image_path, processed_path

def load_data(dataset, dataset_path, split):
    """
    Collect the data for the given split

    Args:
        - For test:
            dataset : the name of the testing dataset
            split : test or train
        - For train:
            dataset : the name of the training dataset
            split : train or test
            input_motion_length : the input motion length
    """
    split_path = os.path.join('./processed', dataset, split+'.npy')
    if os.path.exists(split_path):
        split_processed = np.load(split_path, allow_pickle=True)[()]
        image_path = split_processed['image_path']
        processed_path = split_processed['processed_path']
    else:
        folder_path = os.path.join('./processed', dataset)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        image_path, processed_path = get_path(dataset_path, dataset, split)
        split_processed = {
            'image_path':  image_path,
            'processed_path': processed_path
        }
        np.save(split_path, split_processed)

    return  image_path, processed_path

def get_dataloader(
    dataset,
    split,
    batch_size,
    num_workers=32,
):

    if split == "train":
        shuffle = True
        drop_last = True
        num_workers = num_workers
    else:
        shuffle = False
        drop_last = False
        num_workers = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=False,
    )
    return loader

if __name__ == "__main__":
    
    load_data(
        'dataset', 
        'multiface',
        'train'
    )
