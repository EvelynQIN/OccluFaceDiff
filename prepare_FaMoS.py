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
from utils.image_process import get_arcface_input, batch_crop_lmks
import cv2 

def batch_3d_to_2d(calibration, lmk_3d):
    
    # all in tensor
    bs = lmk_3d.shape[0]
    camera_intrinsics = torch.from_numpy(calibration["intrinsics"]).expand(bs,-1,-1).float()
    camera_extrinsics = torch.from_numpy(calibration["extrinsics"]).expand(bs,-1,-1).float()
    radial_distortion = torch.from_numpy(calibration["radial_distortion"]).expand(bs,-1).float()
    
    lmk_2d = batch_perspective_project(lmk_3d, camera_intrinsics, camera_extrinsics, radial_distortion)

    return lmk_2d

def batch_normalize_lmk_3d(lmk_3d):
    """
    normalize 3d landmarks s.t. the len between no.30 and no.27 = 1
    set the root to be the no.30 lmk (nose_tip)
    Args:
        lmk_3d: tensor (bs, n, 3)
    Returns:
        lmk_3d_normed: tensor (bs, n, 3)
    """
    root_idx = 30
    pivot_idx = 27
    bs, num_lmk, _ = lmk_3d.shape
    root_node = lmk_3d[:, root_idx] # (bs, 3)
    nose_len = torch.norm(lmk_3d[:, pivot_idx]-root_node, 2, -1)    # (bs, )
    lmk_3d_normed = lmk_3d - root_node.unsqueeze(1).expand(-1, num_lmk, -1)
    lmk_3d_normed = torch.divide(lmk_3d_normed, nose_len.reshape(-1, 1, 1).expand(-1, num_lmk, 3))
    return lmk_3d_normed

def get_images_input_for_arcface(img_dir, subject_id, motion_id, cam_name, frame_ids, lmk2d):

    motion_dir = os.path.join(img_dir, subject_id, motion_id)
    img_frame_ids = sorted(np.asarray(os.listdir(motion_dir), dtype=int)) # to be 0 indexed
    img_frame_ids = torch.LongTensor(img_frame_ids)
    idxs = torch.nonzero(frame_ids[..., None] == img_frame_ids)[:,0]
    n_frames = lmk2d.shape[0]
    img_arr = []
    mask = torch.zeros(n_frames)
    lmk_5 = lmk2d[:, [37, 44, 30, 60, 64], :]  # left eye, right eye, nose, left mouth, right mouth
    lmk_5[:, 0, :] = lmk2d[:, [38, 41], :].mean(1)  # center of left eye
    lmk_5[:, 1, :] = lmk2d[:, [44, 47], :].mean(1)  # center of right eye
    for idx in idxs:
        frame_id = f"{frame_ids[idx]:06d}"
        img_path = os.path.join(motion_dir, frame_id, f"{motion_id}.{frame_id}.{cam_name}.png")
        if not os.path.isfile(img_path):
            print(f"{img_path} not existed, skipped!")
            continue
        img = cv2.imread(img_path)
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            print(f"{img_path} cannot be read, skipped!")
            continue
        mask[idx] = 1
        arcface_input = get_arcface_input(lmk_5[idx].numpy(), img)
        img_arr.append(arcface_input)
    img_arr = np.stack(img_arr)
    
    return mask.bool(), torch.from_numpy(img_arr).float()

def get_training_data(motion, flame, calib_fname):
    # zero the global translation and pose (rigid transformation)
    shape = torch.Tensor(motion["flame_shape"])
    expression = torch.Tensor(motion["flame_expr"])
    rot_aa = torch.Tensor(motion["flame_pose"]) # full poses exluding eye poses (root, neck, jaw, left_eyeball, right_eyeball)
    trans = torch.Tensor(motion['flame_trans'])
    frame_id = torch.LongTensor(motion['frame_id'])
    
    n_frames = expression.shape[0]
    # get 2d landmarks from gt mesh
    verts_3d, lmk_3d = flame(shape, expression, rot_aa, trans) # (nframes, V, 3)
    calibration = load_mpi_camera(calib_fname, resize_factor=4)
    lmk_2d = batch_3d_to_2d(calibration, lmk_3d)
    verts_2d = batch_3d_to_2d(calibration, verts_3d)
    lmk_2d_cropped, verts_2d_cropped = batch_crop_lmks(
        lmk_2d, verts_2d, trans_scale=0, scale=1.5, image_size=224)

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
    target = torch.cat([shape, expression, rot_6d], dim=1)
    output = {
        "lmk_2d": lmk_2d_cropped, 
        "verts_2d_cropped": verts_2d_cropped,
        "lmk_3d_normed": lmk_3d_normed,
        "lmk_3d_cam": lmk_3d_cam_local,
        "target": target, 
        "frame_id": frame_id,
    }
    return output, lmk_2d

def main(args):
    dataset = 'FaMoS'
    print("processing dataset ", dataset)

    flame = FLAME(args.flame_model_path, args.flame_lmk_embedding_path)
    
    flame_params_folder = os.path.join(args.root_dir, dataset, "flame_params")
    camera_calibration_folder = os.path.join(args.root_dir, dataset, "calibrations")
    camera_name = "26_C" # name of the selected camera view
    img_dir = 'dataset/FaMoS/downsampled_images_4'

    # split the dataset by subject id and motion id
    splits = {
        "train": {
            "subjects" : ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010',
                          '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', 
                          '021', '022', '023', '024', '025', '026', '027', '028', '029', '030',
                          '031', '032', '033', '034', '035', '036', '037', '038', '039', '040',
                          '041', '042', '043', '044', '045', '046', '047', '048', '049', '050',
                          '051', '052', '053', '054', '055', '056', '057', '058', '059', '060',
                          '061', '062', '063', '064', '065', '066', '067', '068', '069', '070',],
            "motions" : ['anger', 'bareteeth', 'blow_cheeks', 'eyebrow', 
                         'fear', 'happiness', 'head_rotation_left_right', 'head_rotation_up_down', 
                         'jaw', 'kissing', 'lip_corners_down', 'lips_back', 'lips_up', 
                         'mouth_down', 'mouth_extreme', 'mouth_middle', 'mouth_open', 
                         'mouth_up', 'rolling_lips', 'sadness', 'sentence', 'smile_closed', 'surprise', 'wrinkle_nose']
        },
        "val": {
            "subjects" : ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010',
                          '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', 
                          '021', '022', '023', '024', '025', '026', '027', '028', '029', '030',
                          '031', '032', '033', '034', '035', '036', '037', '038', '039', '040',
                          '041', '042', '043', '044', '045', '046', '047', '048', '049', '050',
                          '051', '052', '053', '054', '055', '056', '057', '058', '059', '060',
                          '061', '062', '063', '064', '065', '066', '067', '068', '069', '070',],
            "motions" : ["disgust", "high_smile", "mouth_side", "cheeks_in"]
            
        },
        "test" : {
            "subjects" : ['071', '072', '073', '074', '075', '076', '077', '078', '079', '080',
                          '081', '082', '083', '084', '085', '086', '087', '088', '089', '090',
                          '091', '092', '093'],
            "motions" : ['anger', 'bareteeth', 'blow_cheeks', 'cheeks_in', 'disgust', 'eyebrow', 'fear', 
                         'happiness', 'head_rotation_left_right', 'head_rotation_up_down', 'high_smile', 
                         'jaw', 'kissing', 'lip_corners_down', 'lips_back', 'lips_up', 'mouth_down', 
                         'mouth_extreme', 'mouth_middle', 'mouth_open', 'mouth_side', 'mouth_up', 
                         'rolling_lips', 'sadness', 'sentence', 'smile_closed', 'surprise', 'wrinkle_nose']
        }
    }

    phases = ["train", "val", "test"]
    
    for phase in phases:
        print(f"processing {phase} data")
        savedir = os.path.join(args.save_dir, dataset, phase)   # save training data as subjectid_expression.pt
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        n_sequences = 0
        for subject in os.listdir(flame_params_folder):
            sid = subject.split('_')[-1]
            if sid not in splits[phase]['subjects']:
                continue 
            files = glob.glob(os.path.join(flame_params_folder, subject, "*.npy"))
            for file in files:
                motion_id = os.path.split(file)[-1].split('.')[0]
                output_path = os.path.join(savedir, f"subject_{sid}_{motion_id}.pt")
                if os.path.exists(output_path):
                    continue
                if motion_id not in splits[phase]['motions']:
                    continue 
                data = np.load(file, allow_pickle=True)[()]
                num_frames = len(data["flame_verts"])
                print(f"processing {file} with {num_frames} frames")
                if num_frames == 0:
                    print(f"{subject}_{motion_id} is nulll")
                    continue
                calib_fname = os.path.join(camera_calibration_folder, subject, motion_id, f"{camera_name}.tka")
                output, lmk_2d = get_training_data(data, flame, calib_fname)
                mask, img_arr = get_images_input_for_arcface(img_dir, subject, motion_id, camera_name, output["frame_id"], lmk_2d)
                if torch.sum(mask) == 0:
                    print(f"{subject}_{motion_id} images is nulll")
                    continue
                n_sequences += 1
                output['img_mask'] = mask
                output['arcface_input'] = img_arr
                for k in output:
                    assert output[k] is not None
                torch.save(output, output_path)
        print(f"num of motion sequences in {phase} = {n_sequences}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="=dir where you want to save your generated data",
    )
    parser.add_argument(
        "--root_dir", type=str, default=None, help="=dir where you put your dataset"
    )
    parser.add_argument(
        "--flame_model_path", default='flame_2020/generic_model.pkl', type=str, help="the path to the flame model"
    )
    parser.add_argument(
        "--flame_lmk_embedding_path", default='flame_2020/dense_lmk_embedding.npy', type=str, help="the path to the flame landmark embeddings"
    )

    args = parser.parse_args()

    main(args)