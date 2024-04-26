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
from utils.famos_camera import batch_perspective_project, load_mpi_camera
from utils import utils_transform

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
        dataset_names, # list of dataset names
        split_data,
        input_motion_length=120,
        train_dataset_repeat_times=1,
        no_normalization=True,
        occlusion_mask_prob=0.5,
        fps=30
    ):
        self.split_data = split_data
        self.fps = fps
        self.dataset_info = defaultdict(dict)
        for dataset_name in dataset_names:
            data_fps_original = 60 if dataset_name != 'multiface' else 30
            self.dataset_info[dataset_name]['original_image_size'] = dataset_setting.image_size[dataset_name]
            self.dataset_info[dataset_name]['skip_frames'] = int(data_fps_original / self.fps) 
            self.dataset_info[dataset_name]['input_motion_length'] = input_motion_length * self.dataset_info[dataset_name]['skip_frames']

        # image process
        self.image_size = 224 
        self.scale = 1.5
        self.trans = 0
        self.cam_id = '26_C'
        self.rot_angle = [-10, 10]  # random rotation
        
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization
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

def random_occlusion(occlusion_type, mask_array):
    occlusion_types = [
        'downsample_frame',
        'bottom_right',
        'bottom_left',
        'top_left',
        # 'top_right',
        'right_half',
        'left_half',
        'top_half' ,
        'bottom_half',
        'all_occ',
        'missing_middle'
    ]
    num_occ_types = len(occlusion_types)

    if occlusion_type == 'random_occlusion':
        occlusion_id = torch.randint(low=0, high=num_occ_types, size=(1,))[0]
        occlusion_type = occlusion_types[occlusion_id]
    
    bs, h, w = mask_array.shape
    ch = h // 2
    cw = w // 2

    if occlusion_type == 'downsample_frame':
        mask_array[::3] = 0
    elif occlusion_type == 'bottom_right':
        mask_array[:,ch:,cw:] = 0
    elif occlusion_type == 'missing_middle':
        mask_array[5:15,:,:] = 0
    elif occlusion_type == 'bottom_left':
        mask_array[:,ch:,:cw] = 0
    elif occlusion_type == 'top_left':
        mask_array[:,:ch,:cw] = 0
    elif occlusion_type == 'top_right':
        mask_array[:,:ch,cw:] = 0
    elif occlusion_type == 'right_half':
        mask_array[:,:,cw:] = 0
    elif occlusion_type == 'left_half':
        mask_array[:,:,:cw] = 0
    elif occlusion_type == 'top_half':
        mask_array[:,:ch,:] = 0
    elif occlusion_type == 'bottom_half':
        mask_array[:,ch:,:] = 0
    elif occlusion_type == 'all_occ':
        mask_array[:,:,:] = 0
    else:
        raise ValueError(f"Occlusion type not supported!")
    return mask_array

class TestOneMotion:
    def __init__(
        self,
        dataset_name,
        split_data,
        input_motion_length=120,
        occlusion_mask_prob=0.5,
        fps=30,
        occlusion_type=None
    ):
        self.split_data = split_data    # path dict for one test motion sequence
        self.dataset_name = dataset_name
        self.data_fps_original = 60 if dataset_name != 'multiface' else 30
        self.original_image_size = dataset_setting.image_size[self.dataset_name]
        self.occlusion_type = occlusion_type

        # image process
        self.image_size = 224 
        self.scale = 1.5
        self.trans = 0
        self.cam_id = '26_C'
        
        self.input_motion_length = input_motion_length
        self.occlusion_mask_prob = occlusion_mask_prob
        self.fps = fps
        self.skip_frames = int(self.data_fps_original / self.fps) 

        self.calib = load_mpi_camera(
            calib_fname=self.split_data['calib_path'],
            resize_factor=4.0
        )

        processed_data = torch.load(self.split_data['processed_path'])
        self.img_folder = self.split_data['img_folder']
        self.motion_id = self.split_data['motion_id']
        self.subject_id = self.split_data['subject_id']
        self.audio_path = self.split_data['audio_path']

        # get processed data
        self.frame_id = processed_data['frame_id'][::self.skip_frames]
        self.num_frames = len(self.frame_id)
        self.lmk_2d = processed_data['lmk_2d'][::self.skip_frames]  # (n, 68, 2)
        self.shape = processed_data['shape'][::self.skip_frames]
        if 'audio_emb' in processed_data:
            self.audio_emb = processed_data['audio_emb'][::self.skip_frames]
            self.with_audio = True
        else:
            self.audio_emb = torch.zeros((self.target.shape[0], 768))
            self.with_audio = False
        
        # get target flame params for 2d aligment
        flame_params = np.load(self.split_data['flame_param_path'], allow_pickle=True)[()]
        
        self.expression = torch.Tensor(flame_params["flame_expr"])[::self.skip_frames,:50]
        self.rot_aa = torch.Tensor(flame_params["flame_pose"])[::self.skip_frames,:3*3] # full poses exluding eye poses (root, neck, jaw, left_eyeball, right_eyeball)
        self.trans = torch.Tensor(flame_params['flame_trans'])[::self.skip_frames].unsqueeze(1) 
        # get random mask
        self.img_mask = self.get_occlusion_mask(self.num_frames, self.occlusion_type)
    
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

    def get_occlusion_mask(self, num_frames, occlusion_type='downsample_frame'):
        # add random occlusion mask
        add_mask = torch.bernoulli(torch.ones(1) * self.occlusion_mask_prob)[0]
        mask_array = torch.ones(num_frames, self.original_image_size[0], self.original_image_size[1])
        if not add_mask:
            print(f"non occluded mask returned")
            return mask_array
        return random_occlusion(occlusion_type, mask_array)

    def get_lmk_mask(self, lmk2d, img_mask):
        lmk_mask = torch.ones(lmk2d.shape[0], lmk2d.shape[1]) # [B, V]
        kpt = (lmk2d[...,:2].clone() * 112 + 112).long()
        for i in range(lmk2d.shape[0]):
            for j in range(lmk2d.shape[1]):
                x, y = kpt[i, j]
                if x < 0 or y < 0 or \
                    x >= self.image_size or y >= self.image_size or \
                    img_mask[i, y, x] < 1e-8:
                    lmk_mask[i, j] = 0
        return lmk_mask

    def image_augment(self, image):
         # image augmentation
        transf_order, b, c, s, h = transforms.ColorJitter.get_params(
            brightness=(1, 2),
            contrast=(1, 1.5),
            saturation=(1, 1),
            hue=(-0.1,0.1))
        
        return random_color_jitter_to_video(image, b, c, s, h, transf_order)

    def prepare_chunk_motion(self, start_id):

        # read images as input 
        images_list = []
        kpt_list = []
        original_images_list = []
        mask_list = []
        lmk_2d_gt = self.lmk_2d.clone()[start_id:start_id+self.input_motion_length]
        mask_gt = self.img_mask.clone()[start_id:start_id+self.input_motion_length]
        mask_gt = (mask_gt * 255).numpy().astype(np.uint8)
        
        for i, fid in enumerate(self.frame_id[start_id:start_id+self.input_motion_length]):
            frame_id = "%06d"%(fid)
            kpt = lmk_2d_gt[i].numpy()
            
            img_path = os.path.join(self.img_folder, f"{self.motion_id}.{frame_id}.{self.cam_id}.jpg")
            if not os.path.exists(img_path):
                frame = np.zeros((self.original_image_size[0], self.original_image_size[1], 3)).astype(np.uint8)
            else:
                frame = cv2.imread(img_path)
                if frame is None:
                    frame = np.zeros((self.original_image_size[0], self.original_image_size[1], 3)).astype(np.uint8)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tform = self.crop_face(kpt, self.scale) 
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T
            cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1
            cropped_image = warp(frame, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_mask = warp(mask_gt[i], tform.inverse, output_shape=(self.image_size, self.image_size))
            images_list.append(cropped_image.transpose(2,0,1)) # (3, 224, 224)
            mask_list.append(cropped_mask)
            kpt_list.append(cropped_kpt)
            original_images_list.append(frame.transpose(2,0,1) / 255.)

        image_array = torch.from_numpy(np.stack(images_list)).type(dtype = torch.float32) 
        mask_array = torch.from_numpy(np.stack(mask_list)).type(dtype = torch.float32) 
        kpt_array = torch.from_numpy(np.stack(kpt_list)).type(dtype = torch.float32) 
        original_img = torch.from_numpy(np.stack(original_images_list)).type(dtype = torch.float32) 

        lmk_mask = self.get_lmk_mask(kpt_array, mask_array)

        return {
            'image': image_array, # (n, 3, 224, 224)
            'lmk_2d': kpt_array, # (n, 68, 3)
            'img_mask': mask_array.float(), # (n, 224, 224)
            'lmk_mask': lmk_mask.float(),   # (n, 68)
            'audio_emb': self.audio_emb[start_id:start_id+self.input_motion_length].float(),
            'original_img': original_img
        }


def load_split_for_subject(dataset, dataset_path, subject_id, split='test', motion_id=None, cam_id = '26_C'):
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
    audio_folder = os.path.join(dataset_path, dataset, 'audio')
    calib_folder = os.path.join(dataset_path, dataset, 'calib')
    flame_param_folder = os.path.join(dataset_path, dataset, 'flame_params')
    img_paths = []
    processed_paths = []
    calib_paths = []
    flame_param_paths = []
    subjects = []
    motion_ids = []
    audios = []
    if motion_id == None:
        for motion in selected_motions:
            img_paths.append(os.path.join(img_folder, subject_id, motion))
            processed_paths.append(os.path.join(processed_folder, subject_id, f"{motion}.pt"))
            calib_paths.append(os.path.join(calib_folder, subject_id, motion, f"{cam_id}.tka"))
            subjects.append(subject_id)
            motion_ids.append(motion)
            flame_param_paths.append(os.path.join(flame_param_folder, subject_id, f"{motion}.npy"))
            audios.append(os.path.join(audio_folder, subject_id, f"{motion}.wav"))
    else:
        img_paths.append(os.path.join(img_folder, subject_id, motion_id))
        processed_paths.append(os.path.join(processed_folder, subject_id, f"{motion_id}.pt"))
        calib_paths.append(os.path.join(calib_folder, subject_id, motion_id, f"{cam_id}.tka"))
        subjects.append(subject_id)
        motion_ids.append(motion_id)
        flame_param_paths.append(os.path.join(flame_param_folder, subject_id, f"{motion_id}.npy"))
        audios.append(os.path.join(audio_folder, subject_id, f"{motion_id}.wav"))

    split_data = {
        'img_folder': img_paths,
        'processed_path': processed_paths,
        'calib_path': calib_paths,
        'subject_id': subjects,
        'motion_id': motion_ids,
        'flame_param_path': flame_param_paths,
        'audio_path': audios
    }
    return split_data

def get_path_voca(dataset_path, dataset, split):
    """Return the subject id and motion id for the selected split
    Args:
        dataset: only suport dataset of vocaset & FaMoS
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

                if len(motion_data['lmk_2d']) < 40:
                    continue
                processed_paths.append(motion.path)
                motion_image_path = os.path.join(img_folder, subject.name, motion_id)
                img_paths.append(motion_image_path)
    
    split_data = {
        'img_folders': img_paths,
        'processed_paths': processed_paths,
        'dataset': [dataset] * len(img_paths)
    }
    return split_data

def get_path_multiface(dataset_path, dataset, split):
    if dataset == 'multiface':
        split_id = dataset_setting.multiface_split[split]
        selected_motions = [dataset_setting.multiface_motion_id[i] for i in split_id]
    else:
        raise ValueError(f"Dataset name not supported!, Should be in [multiface]")
    subjects = [subject.path for subject in os.scandir(os.path.join(dataset_path, dataset)) if subject.is_dir()]
    processed_path = []
    for subject in subjects:
        for motion_path in glob.glob(os.path.join(subject, 'processed_data', '*.pt')):
            motion_name = os.path.split(motion_path)[1][:-3]
            if motion_name in selected_motions:
                if len(glob.glob(os.path.join(subject, 'images', f'{motion_name}/*/*.png'))) < 20:
                    continue
                processed_path.append(motion_path)
    
    split_data = {
        'img_folders': [None] * len(processed_path),
        'processed_paths': processed_path,
        'dataset': [dataset] * len(processed_path)
    }
    
    return split_data

def get_path(dataset_path, dataset, split):
    if dataset in ['FaMoS', 'vocaset']:
        return get_path_voca(dataset_path, dataset, split)
    elif dataset == 'multiface':
        return get_path_multiface(dataset_path, dataset, split)
    else:
        raise ValueError(f"Dataset name not supported!, Should be in [vocaset, FaMoS, multiface]")
        
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
            split_data = get_path(dataset_path, dataset, split)

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
