import os  
from glob import glob 
import numpy as np
import face_alignment 
import cv2 
from tqdm import tqdm
from utils.data_util import landmarks_interpolate
from loguru import logger

# corresponding to lmk68 [49, 50, 51, 52, 53, 61, 62, 63]
upper_mouth_lmk_ids = [293, 1065, 3996, 4002, 4106, 1069, 4053, 4059] 

# corresponding to lmk68 [59, 58, 57, 56, 55, 67, 66, 65]
lower_mouth_lmk_ids = [2077, 2078, 3362, 4030, 4027, 296, 3363, 4099]

# corresponding to lmk68 [37, 38, 43, 44]
upper_eyelid_lmk_ids = [2351, 2339, 6406, 6403]

# corresponding to lmk68 [41, 40, 47, 46]
lower_eyelid_lmk_ids = [2330, 2329, 6381, 6384]

scale_facter = 1 / 1000.0 # convert mm to m


# compute the landmarks and the distance between two points for each image 
def create_processed_data(path_to_dataset, device):
    
    # init face alignment detector
    face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)
    subjects = [x.path for x in os.scandir(path_to_dataset) if x.is_dir()]
    logger.add('multiface_lmk_detection.log')
    
    for sbj in subjects:
        print("Processing {}".format(sbj))
        to_folder = os.path.join(sbj, "processed_data")
        if not os.path.exists(to_folder):
            os.makedirs(to_folder)

        for motion in os.scandir(os.path.join(sbj, "images")):
            # compute 3d mouth closure and eye closure
            out_fname = os.path.join(to_folder, motion.name + ".npy")
            if os.path.exists(out_fname): 
                continue
            
            lmk68 = []
            frame_ids = []
            null_face_cnt = 0
            for image_path in sorted(glob(os.path.join(motion.path, '*/*.png'))):
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                lmks = face_detector.get_landmarks_from_image(
                    img_rgb, return_bboxes=False, return_landmark_score=False)
                if lmks is None or len(lmks) == 0:
                    lmk_frame = None
                    null_face_cnt += 1
                else: 
                    lmk_frame = lmks[0]
                lmk68.append(lmk_frame)
                frame_ids.append(int(os.path.split(image_path)[-1][:-4]))
            
            # linear interpolate the missing landmarks
            lmk68 = landmarks_interpolate(lmk68)
            lmk68 = np.stack(lmk68)
            frame_ids = np.stack(frame_ids)
            
            mouth_closure_3d = []
            eye_closure_3d = []
            for fid in frame_ids:
                mesh_path = os.path.join(sbj, "tracked_mesh", motion.name, "%06d"%(fid) + ".bin")
                mesh_verts = np.fromfile(mesh_path, dtype=np.float32).reshape(-1, 3) * scale_facter 
                mouth_dist = np.linalg.norm(mesh_verts[upper_mouth_lmk_ids] - mesh_verts[lower_mouth_lmk_ids], axis=-1)
                eye_dist = np.linalg.norm(mesh_verts[upper_eyelid_lmk_ids] - mesh_verts[lower_eyelid_lmk_ids], axis=-1)
                mouth_closure_3d.append(mouth_dist)
                eye_closure_3d.append(eye_dist)
            mouth_closure_3d = np.stack(mouth_closure_3d)
            eye_closure_3d = np.stack(eye_closure_3d)
            
            assert mouth_closure_3d.shape[0] == lmk68.shape[0], f"{motion.name}: {mouth_closure_3d.shape}, {lmk68.shape}"
            processed = {
                'lmk68': lmk68, # (n, 68, 2)
                'mouth_closure_3d': mouth_closure_3d,   # (n, 8)
                'eye_closure_3d': eye_closure_3d,   # (n, 4)
                'frame_id': frame_ids   # (n,)
            }
            logger.info(f"Motion: {motion.name} has {null_face_cnt} / {len(frame_ids)} null faces.")
            
            np.save(out_fname, processed)

if __name__ == '__main__':
    path_to_dataset = 'dataset/multiface'
    device = 'cuda'
    create_processed_data(path_to_dataset, device)