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
import cv2 
from torchvision import transforms
import torchvision.transforms.functional as F 

def random_color_jitter_to_video(imgs, brightness, contrast, saturation, hue, order):
    #imgs of shape [N, 3, h, w]
    vid_transforms = []
    vid_transforms.append(lambda img: F.adjust_brightness(img, brightness))
    vid_transforms.append(lambda img: F.adjust_contrast(img, contrast))
    vid_transforms.append(lambda img: F.adjust_saturation(img, saturation))
    vid_transforms.append(lambda img: F.adjust_hue(img, hue))
    
    transform = transforms.Compose([vid_transforms[id] for id in order])

    return transform(imgs)

class TrainDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        split_data,
        input_motion_length=120,
        train_dataset_repeat_times=1,
        no_normalization=True,
        occlusion_mask_prob=0.5,
        fps=30
    ):
        self.split_data = split_data
        self.dataset_name = dataset_name
        self.data_fps_original = 60 if dataset_name != 'multiface' else 30
        self.original_image_size = dataset_setting.image_size[self.dataset_name]

        # image process
        self.image_size = 224 
        self.scale = [1.2, 1.8]
        self.trans = 0
        self.cam_id = '26_C'
        self.rot_angle = [-10, 10]  # random rotation
        
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization
        self.input_motion_length = input_motion_length
        self.occlusion_mask_prob = occlusion_mask_prob
        self.fps = fps
        self.skip_frames = int(self.data_fps_original / self.fps) 
        self.input_motion_length = input_motion_length * self.skip_frames

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

    def get_occlusion_mask(self, num_frames, with_audio):
        # add random occlusion mask
        mask_array = torch.ones((num_frames, self.image_size, self.image_size))
        add_mask = torch.bernoulli(torch.ones(1) * self.occlusion_mask_prob)[0]
        if not add_mask:
            return mask_array

        # select occlusion type
        occlusion_type = torch.randint(low=0, high=3, size=(1,))[0]

        # select occlusion type
        occlusion_type = torch.randint(low=0, high=3, size=(1,))[0]

        if with_audio:
            if occlusion_type == 0:
                # occlude all visual cues
                mask_array[:,:,:] = 0
            elif occlusion_type == 1:
                # occlude the whole mouth region
                mask_array[:,100:,:] = 0
            else:
                # occlude random regions for each frame
                mask_bbx = torch.randint(low=4, high=220, size=(num_frames,4)) 
                for i in range(num_frames):
                    mask_array[i, mask_bbx[i,0]:mask_bbx[i,1], mask_bbx[i,2]:mask_bbx[i,3]] = 0
        else:
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
        transf_order, b, c, s, h = transforms.ColorJitter.get_params(
            brightness=(1, 2),
            contrast=(1, 1.5),
            saturation=(1, 1),
            hue=(-0.1,0.1))
        
        return random_color_jitter_to_video(image, b, c, s, h, transf_order)

    def __getitem__(self, idx):
        id = idx % len(self.split_data['img_folders'])
        processed_data = torch.load(self.split_data['processed_paths'][id])
        img_folder = self.split_data['img_folders'][id]
        motion_id = os.path.split(img_folder)[-1]

        frame_id = processed_data['frame_id']
        seqlen = len(frame_id)

        if seqlen == self.input_motion_length: 
            start_id = 0
        else:
            start_id = torch.randint(0, int(seqlen - self.input_motion_length), (1,))[0]     # random crop a motion seq

        frame_id = frame_id[start_id:start_id + self.input_motion_length:self.skip_frames]
        lmk_2d = processed_data['lmk_2d'][start_id:start_id + self.input_motion_length:self.skip_frames]  # (n, 68, 2)
        target = processed_data['target'][start_id:start_id + self.input_motion_length:self.skip_frames]
        shape = processed_data['shape'][start_id:start_id + self.input_motion_length:self.skip_frames]
        if 'audio_emb' in processed_data:
            audio_emb = processed_data['audio_emb'][start_id:start_id + self.input_motion_length:self.skip_frames]
            with_audio = True
        else:
            audio_emb = torch.zeros((target.shape[0], 768))
            with_audio = False
        
        # read images as input 
        images_list = []
        kpt_list = []
        # randomly sample image augmentation params
        scale = np.random.uniform(self.scale[0], self.scale[1])
        angle = np.random.uniform(self.rot_angle[0], self.rot_angle[1])
        M = cv2.getRotationMatrix2D((112, 112), angle, scale=1.0)

        for i, fid in enumerate(frame_id):
            frame_id = "%06d.jpg"%(fid)
            kpt = lmk_2d[i].numpy()
            
            img_path = os.path.join(img_folder, f"{motion_id}.{frame_id}.{self.cam_id}.jpg")
            if not os.path.exists(img_path):
                frame = np.zeros((self.original_image_size[0], self.original_image_size[1], 3)).astype(np.uint8)
            else:
                frame = cv2.imread(img_path)
                if frame is None:
                    frame = np.zeros((self.original_image_size[0], self.original_image_size[1], 3)).astype(np.uint8)

            # apply random rotation to both lmks and frame
            frame = cv2.warpAffine(frame, M, (self.original_image_size[1], self.original_image_size[0]))
            kpt_homo = np.concatenate([kpt[...,:2], np.ones((68, 1))], axis=-1)
            kpt = M.dot(kpt_homo.T).T

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tform = self.crop_face(kpt, scale) 
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T
            cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1
            cropped_image = warp(frame, tform.inverse, output_shape=(self.image_size, self.image_size))

            images_list.append(cropped_image.transpose(2,0,1)) # (3, 224, 224)
            kpt_list.append(cropped_kpt)

        image_array = torch.from_numpy(np.stack(images_list)).type(dtype = torch.float32) 
        kpt_array = torch.from_numpy(np.stack(kpt_list)).type(dtype = torch.float32) 

        image_array = self.image_augment(image_array)

        # mask_array = torch.ones((images_array.shape[0], 224, 224))
        img_mask = self.get_occlusion_mask(kpt_array.shape[0], with_audio)
        lmk_mask = self.get_lmk_mask(kpt_array, img_mask)

        return {
            'image': image_array, # (n, 3, 224, 224)
            'lmk_2d': kpt_array, # (n, 68, 3)
            'img_mask': img_mask.float(), # (n, 224, 224)
            'lmk_mask': lmk_mask.float(),   # (n, 68)
            'audio_emb': audio_emb.float(),
            'shape': shape.float(),
            'target': target.float()
        }

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

def get_split(dataset_path, dataset, split):
    """Return the subject id and motion id for the selected split
    """
    if dataset == 'FaMoS':
        split_id = dataset_setting.FaMoS_split[split]
        selected_motions = [dataset_setting.FaMoS_motion_id[i] for i in split_id]
    elif dataset == 'vocaset':
        split_id = dataset_setting.vocaset_split[split]
        selected_motions = [dataset_setting.vocaset_motion_id[i] for i in split_id]
    else:
        raise ValueError(f"Dataset name not supported!, Should be in [FaMoS, vocaset]")
    processed_folder = os.path.join(dataset_path, dataset, 'processed')
    img_folder = os.path.join(dataset_path, dataset, 'image')
    img_paths = []
    processed_paths = []
    for subject in os.scandir(processed_folder):
        for motion in os.scandir(subject):
            motion_id = motion.name[:-3]
            if motion_id in selected_motions:
                motion_data = torch.load(motion.path)

                if len(motion_data['frame_id']) < 45:
                    continue
                processed_paths.append(motion.path)
                motion_image_path = os.path.join(img_folder, subject.name, motion_id)
                img_paths.append(motion_image_path)
    
    split_data = {
        'img_folders': img_paths,
        'processed_paths': processed_paths
    }
    return split_data

def load_data(datasets, dataset_path, split):
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
    split_data_all = defaultdict(list)
    for dataset in datasets:
        split_path = os.path.join('./processed', dataset, split+'.npy')
        if os.path.exists(split_path):
            split_data = np.load(split_path, allow_pickle=True)[()]
            
        else:
            folder_path = os.path.join('./processed', dataset)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            split_data = get_split(dataset_path, dataset, split)

            np.save(split_path, split_data)
        
        for key in split_data:
            split_data_all[key].extend(split_data[key])

    return  split_data_all

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
