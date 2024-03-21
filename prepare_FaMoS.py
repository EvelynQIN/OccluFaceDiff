import argparse
import os
import numpy as np
import torch
from tqdm import tqdm 
import glob
from utils import utils_transform
import pickle
from model.FLAME import FLAME
from model.mica import MICA
from utils.famos_camera import batch_perspective_project, load_mpi_camera
from utils.image_process import get_arcface_input, batch_crop_lmks, crop_np, batch_normalize_lmk_3d
import cv2 
from skimage.transform import estimate_transform, warp, resize, rescale
from configs.config import get_cfg_defaults

def batch_3d_to_2d(calibration, lmk_3d):
    
    # all in tensor
    bs = lmk_3d.shape[0]
    camera_intrinsics = torch.from_numpy(calibration["intrinsics"]).expand(bs,-1,-1).float()
    camera_extrinsics = torch.from_numpy(calibration["extrinsics"]).expand(bs,-1,-1).float()
    radial_distortion = torch.from_numpy(calibration["radial_distortion"]).expand(bs,-1).float()
    
    lmk_2d = batch_perspective_project(lmk_3d, camera_intrinsics, camera_extrinsics, radial_distortion)

    return lmk_2d

def get_mica_shape_prediction(img_dir, subject_id, motion_id, cam_name, frame_ids, lmk2d, mica):

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
    img_arr = np.stack(img_arr) # (n_imgs, 3, 112, 112)
    
    with torch.no_grad():
        mica_shape = mica.predict_per_frame_shape(
            torch.from_numpy(img_arr).float().to('cuda'))  # (n_imgs, 300)
    assert mica_shape.shape[1] == 300
    
    return mask.bool(), mica_shape.to('cpu')

def get_images_input_for_test(img_dir, subject_id, motion_id, cam_name, frame_ids, lmk2ds):

    motion_dir = os.path.join(img_dir, subject_id, motion_id)
    img_frame_ids = sorted(np.asarray(os.listdir(motion_dir), dtype=int)) # to be 0 indexed
    img_frame_ids = torch.LongTensor(img_frame_ids)
    idxs = torch.nonzero(frame_ids[..., None] == img_frame_ids)[:,0]
    n_frames = lmk2ds.shape[0]
    arcface_imgs = []
    cropped_imgs = []
    cropped_lmks = []

    mask = torch.zeros(n_frames)
    lmk_5 = lmk2ds[:, [37, 44, 30, 60, 64], :]  # left eye, right eye, nose, left mouth, right mouth
    lmk_5[:, 0, :] = lmk2ds[:, [38, 41], :].mean(1)  # center of left eye
    lmk_5[:, 1, :] = lmk2ds[:, [44, 47], :].mean(1)  # center of right eye
    for i in range(n_frames):
        lmk68 = lmk2ds[i].numpy()
        # crop information
        tform = crop_np(lmk68, trans_scale=0, scale=1.5, image_size=224)
        cropped_lmk68 = np.dot(tform.params, np.hstack([lmk68, np.ones([lmk68.shape[0],1])]).T).T 
        cropped_lmks.append(cropped_lmk68[:, :2])

        # if image available
        if i in idxs:
            frame_id = f"{frame_ids[i]:06d}"
            img_path = os.path.join(motion_dir, frame_id, f"{motion_id}.{frame_id}.{cam_name}.png")
            if not os.path.isfile(img_path):
                print(f"{img_path} not existed, skipped!")
                continue
            img = cv2.imread(img_path) # in BGR format
            if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                print(f"{img_path} cannot be read, skipped!")
                continue
            mask[i] = 1
            arcface_input = get_arcface_input(lmk_5[i].numpy(), img)
            arcface_imgs.append(arcface_input)

            img = img.astype(float) / 255.0
            cropped_image = warp(img, tform.inverse, output_shape=(224, 224)) 
            cropped_imgs.append(cropped_image.transpose(2,0,1)[[2, 1, 0], :, :]) # (3, 224, 224) in RGB

    arcface_imgs = np.stack(arcface_imgs)
    cropped_imgs = np.stack(cropped_imgs)
    cropped_lmks = np.stack(cropped_lmks)
    
    return mask.bool(), torch.from_numpy(arcface_imgs).float(), torch.from_numpy(cropped_imgs).float(), torch.from_numpy(cropped_lmks).float()

def get_training_data(motion, flame, calib_fname):
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
        "target": target, 
        "frame_id": frame_id,
    }
    return output, lmk_2d

def prepare_one_motion_for_test(
        dataset, subject_id, motion_id, flame_model_path, flame_lmk_embedding_path,
        n_shape=100, n_exp=50):
    flame_params_folder = os.path.join('dataset', dataset, "flame_params")
    camera_calibration_folder = os.path.join('dataset', dataset, "calibrations")
    camera_name = "26_C" # name of the selected camera view
    img_dir = os.path.join('dataset', dataset, 'downsampled_images_4')
    flame_params_file = os.path.join(flame_params_folder, subject_id, f'{motion_id}.npy')
    calib_fname = os.path.join(camera_calibration_folder, subject_id, motion_id, f"{camera_name}.tka")
    motion = np.load(flame_params_file, allow_pickle=True)[()]
    flame = FLAME(flame_model_path, flame_lmk_embedding_path)   # original flame with full params

    calibration = load_mpi_camera(calib_fname, resize_factor=4)

    shape = torch.Tensor(motion["flame_shape"])
    expression = torch.Tensor(motion["flame_expr"])
    rot_aa = torch.Tensor(motion["flame_pose"]) # full poses exluding eye poses (root, neck, jaw, left_eyeball, right_eyeball)
    trans = torch.Tensor(motion['flame_trans'])
    frame_id = torch.LongTensor(motion['frame_id'])
    
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
    target = torch.cat([shape[:,:n_shape], expression[:, :n_exp], rot_6d], dim=1)
    output = {
        "lmk_2d": cropped_lmk_2d, # (n, 68, 2)
        "lmk_3d_normed": lmk_3d_normed, # (n, 68, 3)
        "target": target,  # (n, 180)
        "frame_id": frame_id, # (n)
        "img_mask": img_mask,   # (n)
        "arcface_imgs": arcface_imgs, # (n_imgs, 3, 112, 112)
        "cropped_imgs": cropped_imgs,   # (n_imgs, 3, 224, 224)
    }
    
    return output


def main(args):
    dataset = 'FaMoS'
    print("processing dataset ", dataset)

    flame = FLAME(args.flame_model_path, args.flame_lmk_embedding_path)
    
    # load mica pretrained
    pretrained_args = get_cfg_defaults()
    mica = MICA(pretrained_args.mica)
    mica.load_model('cpu')
    mica.to('cuda')
    mica.eval()
    
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
                if num_frames < 10:
                    print(f"{subject}_{motion_id} is nulll")
                    continue
                calib_fname = os.path.join(camera_calibration_folder, subject, motion_id, f"{camera_name}.tka")
                output, lmk_2d = get_training_data(data, flame, calib_fname)
                mask, mica_shape = get_mica_shape_prediction(img_dir, subject, motion_id, camera_name, output["frame_id"], lmk_2d, mica)
                if torch.sum(mask) == 0:
                    print(f"{subject}_{motion_id} images is nulll")
                    continue
                n_sequences += 1
                output['img_mask'] = mask
                output['mica_shape'] = mica_shape
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