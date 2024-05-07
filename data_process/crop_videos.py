import torch 
import cv2 
import numpy as np
import os 
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import pickle
import h5py
from skimage.transform import estimate_transform, warp
from tqdm import tqdm
from pathlib import Path
from time import time

def get_video_list_for_subject(out_folder, subject):
    view = 'front'
    video_list = []
    subject_path = os.path.join(out_folder, subject)
    for emotion in os.scandir(os.path.join(subject_path, 'video', view)):
        for level in os.scandir(emotion.path):
            for sent in os.scandir(level.path):
                video_id = '/'.join([subject, view, emotion.name, level.name, sent.name[:-4]])
                video_list.append(video_id)
    video_list.sort()
    print(f"there are {len(video_list)} videos resampled for {subject} successfully!")
    return video_list

### same processing as EMOCA
def point2bbox(center, size):
    size2 = size / 2

    src_pts = np.array(
        [[center[0] - size2, center[1] - size2], [center[0] - size2, center[1] + size2],
         [center[0] + size2, center[1] - size2]])
    return src_pts

def point2transform(center, size, target_size_height, target_size_width):
    src_pts = point2bbox(center, size)
    dst_pts = np.array([[0, 0], [0, target_size_width - 1], [target_size_height - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)
    return tform

def linear_interpolate(landmarks, start_idx, stop_idx):
    """linear_interpolate.

    :param landmarks: ndarray, input landmarks to be interpolated.
    :param start_idx: int, the start index for linear interpolation.
    :param stop_idx: int, the stop for linear interpolation.
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

def linear_interpolate_landmarks(landmarks):
    """landmarks_interpolate.

    :param landmarks: List, the raw landmark (in-place)

    """
    valid_frames_idx = []
    for idx, res in enumerate(landmarks):
        if res:
            valid_frames_idx.append(idx)
            landmarks[idx] = landmarks[idx][0][:,:2]

    if not valid_frames_idx:
        return valid_frames_idx, landmarks

    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            print(f'linear interpolate {idx}')
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    # -- Corner case: keep frames at the beginning or at the end failed to be detected. (pad the two end with the side detection)
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])

    return valid_frames_idx, landmarks

class VideoProcessor:
    
    def __init__(
        self,
        scale=1.35,
        bb_center_shift_x=0., 
        bb_center_shift_y=0,
        image_size=224,
        lmk_folder=None, # path to the landmark folder for image cropping
        video_folder=None, # path to the video folder (resampled to 25fps)
        to_folder=None, # path to the result folder
        device='cuda:0',
        save_images=False,
    ):

        # image cropping setting
        self.scale = scale
        self.bb_center_shift_x = bb_center_shift_x
        self.bb_center_shift_y = bb_center_shift_y
        self.image_size = image_size
        self.lmk_folder = lmk_folder
        self.video_folder = video_folder
        self.to_folder = to_folder
        self.save_images=save_images

        if self.save_images:
            # init image segmentator
            config_file = 'face_segmentation/deeplabv3plus_r101_512x512_face-occlusion.py'
            checkpoint_file = 'face_segmentation/deeplabv3plus_r101_512x512_face-occlusion-93ec6695.pth'
            self.seg_model = init_model(config_file, checkpoint_file, device=device)

    def warp_image_from_lmk(self, landmarks, img):    
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0 
        center_y = bottom - (bottom - top) / 2.0
        center = np.array([center_x, center_y])

        center[0] += self.bb_center_shift_x*abs(right-left)
        center[1] += self.bb_center_shift_y*abs(bottom-top)

        size = int(old_size * self.scale)

        tform = point2transform(center, size, self.image_size, self.image_size)
        output_shape = (self.image_size, self.image_size)
        dst_image = warp(img, tform.inverse, output_shape=output_shape, order=3)
        dst_landmarks = tform(landmarks[:, :2])

        return dst_image, dst_landmarks

    def load_landmarks(self, video_id):
        # load the original mediapipe 478 landmarks (w.r.t original img size)
        # landmark_folder = 'dataset/mead_25fps/processed/landmarks_original/mediapipe'
        lmk_fname = os.path.join(self.lmk_folder, video_id, 'landmarks_original.pkl')
        if not os.path.exists(lmk_fname):
            return None
        with open(lmk_fname, 'rb') as f:
            landamrks_original = pickle.load(f)
        return landamrks_original

    def process_one_video(self, motion_id):
        """ Process one video, it should be already resampled to 25 fps
        Args:
            video_id: str, in the form of "subject/view/emotion/level/sent"
        """
        video_id = motion_id[:5] + "video/" + motion_id[5:]
        print(video_id)
        to_fname = os.path.join(self.to_folder, motion_id, 'cropped_frames.hdf5')
        if os.path.exists(to_fname):
            return True 
        # load the detected landmark
        landmarks_original = self.load_landmarks(motion_id)

        if landmarks_original is None:
            return True

        Path(to_fname).parent.mkdir(exist_ok=True, parents=True)
        video_path = os.path.join(self.video_folder, f"{video_id}.mp4")
        video = cv2.VideoCapture()

        # linearly interpolate the landamrk with non-detected frames
        valid_frames_idx, landmarks_original = linear_interpolate_landmarks(landmarks_original)
        if not video.open(video_path):
            print(f"Cannot open {video_path}!")
            return False
        frame_id = 0
        if self.save_images:
            frame_list = []
            seg_list = []
        lmk_list = []
        
        start_time = time()
        while True:
            _, frame = video.read()
            if frame is None:
                break
            dst_image, dst_landmarks = self.warp_image_from_lmk(landmarks_original[frame_id], frame)
            # normalize landmarks to [-1, 1]
            dst_landmarks = dst_landmarks / self.image_size * 2 - 1
            lmk_list.append(dst_landmarks)  # (V, 2)

            if self.save_images:
                dst_image = (dst_image * 255.).astype(np.uint8) # (h, w, 3)
                # perform image segmentation to the warped image
                result = inference_model(self.seg_model, dst_image)
                seg_mask = np.asanyarray(result.pred_sem_seg.values()[0].cpu()).squeeze(0)  # (h, w)
                seg_list.append(seg_mask)

                dst_image = (dst_image[:,:,[2,1,0]] / 255.).transpose(2, 0, 1)   # (3, h, w) in RGB float
                frame_list.append(dst_image)
            
            frame_id += 1
        video.release()
        print(f"used {time() - start_time} secs")
        if self.save_images:
            frame_list = np.stack(frame_list)   # (n, 3, 224, 224)
            seg_list = np.stack(seg_list)   # (n, 224, 224)
        lmk_list = np.stack(lmk_list)   # (n, V, 2)
        

        # print(f"shape of frame_list: {frame_list.shape}")
        # print(f"shape of lmk_list: {lmk_list.shape}")
        # print(f"shape of seg_list: {seg_list.shape}")

        f = h5py.File(to_fname, 'w')
        f.create_dataset('lmk_2d', data=lmk_list)
        f.create_dataset('valid_frames_idx', data=np.array(valid_frames_idx))
        if self.save_images:
            f.create_dataset('images', data=frame_list)
            f.create_dataset('img_masks', data=seg_list)
        f.close()

        torch.cuda.empty_cache()
        return True

    def process_all_videos(self, video_list, exp_name):
        # with open(os.path.join(self.video_folder, "video_list.pkl"), 'rb') as f:
        #     video_list = pickle.load(f)
        bad_video_list = []
        for video_id in tqdm(video_list):
            if not self.process_one_video(video_id):
                bad_video_list.append(video_list)   # video cannot be opened
        if bad_video_list:
            with open(f'dataset/mead_25fps/processed/bad_video_list_{exp_name}.pkl', 'wb') as f:
                pickle.dump(bad_video_list, f)

if __name__ == "__main__":
    video_processor = VideoProcessor(
        scale=1.35,
        bb_center_shift_x=0., 
        bb_center_shift_y=0,
        image_size=224,
        lmk_folder="dataset/mead_25fps/processed/landmarks_original/mediapipe", # path to the landmark folder for image cropping
        video_folder="dataset/mead_25fps/processed/videos_25fps", # path to the video folder (resampled to 25fps)
        to_folder="dataset/mead_25fps/processed/images", # path to the result folder
        device='cpu',
        save_images=False
    )
    # mead_subjects = ['M013', 'M019', 'M024', 'M025', 'M028', 'M029', 'M030', 'M033', 'M037', 'M039', 'M041', 'W009', 'W014', 'W015', 'W016', 'W018', 'W019', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029', 'W033', 'W035', 'W036', 'W037', 'W038', 'W040']
    # subject_with_img = ['M003', 'M005', 'M011', 'M012', 'M022', 'M023', 'M026', 'M027', 'M030', 'M031', 'M034', 'M040', 'M041', 'W018', 'W024', 'W029', 'W037']

    # mead subject to process with images
    # mead_subjects_to_do_img = ['W014', 'W029','W037']
    mead_subjects_to_do_wo_img = ['M013', 'M019', 'M024', 'M025', 'M028', 'M029', 'M033', 'M037', 'M039', 'W009', 'W014', 'W015', 'W016', 'W019', 'W023', 'W025', 'W026', 'W028', 'W033', 'W035', 'W036', 'W038', 'W040']

    process_subjects = mead_subjects_to_do_wo_img[10:]

    video_list = []
    print(f"process subjects: {process_subjects}")
    for subject in process_subjects:
        video_list.extend(get_video_list_for_subject("dataset/mead_25fps/processed/videos_25fps", subject))
    video_list.sort()
    video_processor.process_all_videos(video_list, f"batch_5_woimg")