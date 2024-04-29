import argparse
import os
import numpy as np
import torch
from tqdm import tqdm 
import glob
from utils import utils_transform
import pickle
from model.FLAME import FLAME

from utils.famos_camera import batch_perspective_project, load_mpi_camera
from utils.data_util import batch_crop_lmks, crop_np, batch_normalize_lmk_3d
import cv2 
from skimage.transform import estimate_transform, warp, resize, rescale
from configs.config import get_cfg_defaults
from model.wav2vec import Wav2Vec2Model
from transformers import Wav2Vec2Processor
import librosa
from mmseg.apis import inference_model, init_model

def prepare_one_motion_for_test(
        dataset, split, subject_id, motion_id, flame_model_path, flame_lmk_embedding_path,
        n_shape=100, n_exp=50, fps=30):
    skip_frames = 2 if fps == 30 else 1
    flame_params_folder = os.path.join('dataset', dataset, "flame_params")
    camera_calibration_folder = os.path.join('dataset', dataset, "calibrations")
    camera_name = "26_C" # name of the selected camera view
    img_dir = os.path.join('dataset', dataset, 'downsampled_images_4')
    flame_params_file = os.path.join(flame_params_folder, subject_id, f'{motion_id}.npy')
    calib_fname = os.path.join(camera_calibration_folder, subject_id, motion_id, f"{camera_name}.tka")
    motion = np.load(flame_params_file, allow_pickle=True)[()]
    flame = FLAME(flame_model_path, flame_lmk_embedding_path)   # original flame with full params

    calibration = load_mpi_camera(calib_fname, resize_factor=4)

    shape = torch.Tensor(motion["flame_shape"][::skip_frames])
    expression = torch.Tensor(motion["flame_expr"][::skip_frames])
    rot_aa = torch.Tensor(motion["flame_pose"][::skip_frames]) # full poses exluding eye poses (root, neck, jaw, left_eyeball, right_eyeball)
    trans = torch.Tensor(motion['flame_trans'][::skip_frames])
    frame_id = torch.LongTensor(motion['frame_id'][::skip_frames])
    
    n_frames = expression.shape[0]

    # get 2d landmarks from gt mesh
    _, lmk_3d = flame(shape, expression, rot_aa, trans) # (nframes, V, 3)
    calibration = load_mpi_camera(calib_fname, resize_factor=4)
    lmk_2d = batch_3d_to_2d(calibration, lmk_3d)

    # get cropped images and landmarks
    img_mask, arcface_imgs, cropped_imgs, cropped_lmk_2d = get_images_input_for_test(
        img_dir, subject_id, motion_id, camera_name, frame_id, lmk_2d)
    
    # change the root rotation of flame in cam coords and zeros the translation to get the lmk3d
    R_f = utils_transform.aa2matrot(rot_aa[:, :3].reshape(-1, 3)) # (nframes, 3, 3)
    R_C = torch.from_numpy(calibration['extrinsics'][:, :3]).expand(n_frames,-1,-1).float()
    R_m2c = torch.bmm(R_C, R_f)
    root_rot_aa = utils_transform.matrot2aa(R_m2c)
    rot_aa[:, :3] = root_rot_aa
    _, lmk_3d_cam_local = flame(shape, expression, rot_aa) # (nframes, V, 3)
    
    # normalize the 3d lmk (rotated to be in cam coord system), rooted at idx=30
    lmk_3d_normed = batch_normalize_lmk_3d(lmk_3d_cam_local)
    
    # get 6d pose representation
    rot_6d = utils_transform.aa2sixd(rot_aa.reshape(-1, 3)).reshape(n_frames, -1) # (nframes, 5*6)
    
    # get the camT from saved files
    motion_processed = torch.load(os.path.join('processed_data', dataset, split, f'{subject_id[len(dataset)+1:]}_{motion_id}.pt'))
    camT = motion_processed['target'][::skip_frames, -3:]
    target = torch.cat([shape[:,:n_shape], expression[:, :n_exp], rot_6d, camT], dim=1)
    output = {
        "lmk_2d": cropped_lmk_2d, # (n, 68, 2)
        "lmk_3d_normed": lmk_3d_normed, # (n, 68, 3)
        "target": target,  # (n, 180)
        "frame_id": frame_id, # (n)
        "img_mask": img_mask,   # (n)
        "arcface_input": arcface_imgs, # (n_imgs, 3, 112, 112)
        "cropped_imgs": cropped_imgs,   # (n_imgs, 3, 224, 224)
    }
    
    return output

def batch_3d_to_2d(calibration, lmk_3d):
    
    # all in tensor
    bs = lmk_3d.shape[0]
    camera_intrinsics = torch.from_numpy(calibration["intrinsics"]).expand(bs,-1,-1).float()
    camera_extrinsics = torch.from_numpy(calibration["extrinsics"]).expand(bs,-1,-1).float()
    radial_distortion = torch.from_numpy(calibration["radial_distortion"]).expand(bs,-1).float()
    
    lmk_2d = batch_perspective_project(lmk_3d, camera_intrinsics, camera_extrinsics, radial_distortion)

    return lmk_2d

def create_training_data(args, dataset, device='cuda'):
    print("processing dataset ", dataset)
    root_dir = './dataset'
    
    # # init flame
    # flame = FLAME(args)

    # # init wav2vec
    # audio_processor = Wav2Vec2Processor.from_pretrained(
    #     "facebook/wav2vec2-base-960h")  # HuBERT uses the processor of Wav2Vec 2.0
    # wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    # wav2vec.feature_extractor._freeze_parameters()
    # wav2vec.to(device)
    
    # init facial segmentator
    config_file = 'face_segmentation/deeplabv3plus_r101_512x512_face-occlusion.py'
    checkpoint_file = 'face_segmentation/deeplabv3plus_r101_512x512_face-occlusion-93ec6695.pth'
    face_segment = init_model(config_file, checkpoint_file, device=device)
    
    # flame_params_folder = os.path.join(root_dir, dataset, "flame_params")
    # calib_folder = os.path.join(root_dir, dataset, "calib")
    image_folder = os.path.join(root_dir, dataset, "image")
    # audio_folder = os.path.join(root_dir, dataset, "audio")
    out_folder = os.path.join(root_dir, dataset, "processed")
    # camera_name = "26_C" # name of the selected camera view
    print_shape = True

    for sbj in os.scandir(image_folder):
        print(f"Process {sbj.name}")
        
        out_folder_sbj = os.path.join(out_folder, sbj.name)
        if not os.path.exists(out_folder_sbj):
            os.makedirs(out_folder_sbj)
        num_motion = 0
        for motion in os.scandir(sbj.path):
            print(f"Process No.{num_motion}_{motion.name}")
            num_motion += 1
            out_fname = os.path.join(out_folder_sbj, f"{motion.name}.pt")
            if not os.path.exists(out_fname):
                continue
            output = torch.load(out_fname)
            if 'img_mask' in output:
                print(f"{motion.name} already processed!")
                continue
            mask_list = []
            skip_frame = 5
            fid = 0
            prev_mask = None
            for img_path in sorted(glob.glob(os.path.join(motion.path, '*.jpg'))):
                if fid % skip_frame == 0:
                    frame = cv2.imread(img_path)
                    seg_result = inference_model(face_segment, frame)
                    seg_mask = np.asanyarray(seg_result.pred_sem_seg.values()[0].to('cpu'))[0]
                    prev_mask = seg_mask
                else:
                    assert prev_mask is not None
                    seg_mask = prev_mask
                mask_list.append(seg_mask)
                fid += 1
            
            mask_list = np.stack(mask_list)
            
            output['img_mask'] = torch.from_numpy(mask_list).float()
            if print_shape:
                print(output['img_mask'].shape)
                print_shape = False

            # flame_params_path = os.path.join(flame_params_folder, sbj.name, f"{motion.name}.npy")
            # flame_params = np.load(flame_params_path, allow_pickle=True)[()]
   
            # num_frames = len(flame_params["flame_verts"])
            # print(f"processing {motion.name} with {num_frames} frames")
            # if num_frames < 10:
            #     print(f"{sbj.name} {motion.name} is nulll")
            #     continue
            # calib_fname = os.path.join(calib_folder, sbj.name, motion.name, f"{camera_name}.tka")
            # audio_path = os.path.join(audio_folder, sbj.name, f"{motion.name}.wav")

            # # get the gt flame params and projected 2d lmks
            # shape = torch.Tensor(flame_params["flame_shape"])[:,:100]
            # expression = torch.Tensor(flame_params["flame_expr"])[:,:50]
            # rot_aa = torch.Tensor(flame_params["flame_pose"])[:,:3*3] # full poses exluding eye poses (root, neck, jaw, left_eyeball, right_eyeball)
            # trans = torch.Tensor(flame_params['flame_trans']).unsqueeze(1)
            # frame_id = torch.Tensor(flame_params['frame_id']).long()

            # # get 2d landmarks from gt mesh
            # _, lmk_3d, _ = flame(shape, expression, rot_aa) # (nframes, V, 3)
            # lmk_3d += trans
            # calibration = load_mpi_camera(calib_fname, resize_factor=4.0)
            # lmk_2d = batch_3d_to_2d(calibration, lmk_3d)

            # # get 6d pose representation of jaw pose
            # rot_jaw_6d = utils_transform.aa2sixd(rot_aa[...,6:9]) # (nframes, 6)
            # target = torch.cat([rot_jaw_6d, expression], dim=1) 

            # get audio input
            # speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
            # audio_input = audio_processor(
            #     speech_array, 
            #     return_tensors='pt', 
            #     padding="longest", 
            #     sampling_rate=sampling_rate).input_values
            
            # output['audio_input'] = audio_input.squeeze()
            # if print_shape:
            #     print(output['audio_input'].shape)
            #     print_shape = False

            # audio_input = np.squeeze(audio_processor(speech_array, return_tensors=None, padding="longest",
            #                             sampling_rate=sampling_rate).input_values)
            # with torch.no_grad():
            #     audio_input = torch.from_numpy(audio_input).float().unsqueeze(0).to(device)
            #     audio_emb = wav2vec(audio_input, frame_num = num_frames).last_hidden_state.squeeze(0).cpu()
            
            # output = {
            #     "lmk_2d": lmk_2d, 
            #     "target": target, 
            #     "shape": shape,
            #     "audio_emb": audio_emb,
            #     "frame_id": frame_id
            # }
            # for k in output:
            #     assert output[k] is not None
            
            torch.save(output, out_fname)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='get vocaset training data.')

    parser.add_argument('--device', type=str, default='cpu', help='device')

    args = parser.parse_args()

    model_cfg = get_cfg_defaults().model
    dataset = 'vocaset'
    device = args.device

    create_training_data(model_cfg, dataset, device)