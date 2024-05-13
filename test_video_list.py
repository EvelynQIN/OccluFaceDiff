import pickle
import h5py
import os

with open('dataset/mead_25fps/processed/video_list_wimg.pkl', 'rb') as f:
    video_list = pickle.load(f)
image_folder = 'dataset/mead_25fps/processed/images'
for video_id, num_frames in video_list:
    img_path = os.path.join(image_folder, video_id, 'cropped_frames.hdf5')
    with h5py.File(img_path, 'r') as f:
        if 'images' not in f:
            print(video_id)
        f['images'][:3]