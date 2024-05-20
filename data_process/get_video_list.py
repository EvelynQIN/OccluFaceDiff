import os 
import pickle
import h5py

def get_video_list_from_images(images_folder, audio_input_folder):
    video_list = {} # dict of video_id (sbj/view/emotion/level/sent) : num_frames
    view = 'front'
    print_id = True
    for subject in os.scandir(images_folder):
        for emotion in os.scandir(os.path.join(subject.path, view)):
            for level in os.scandir(emotion.path):
                for sent in os.scandir(level.path):
                    audio_path = os.path.join(audio_input_folder, subject.name, emotion.name, level.name, f"{sent.name}.pt")
                    if os.path.exists(audio_path):
                        video_id = '/'.join([subject.name, view, emotion.name, level.name, sent.name])
                        if print_id:
                            print(video_id)
                            print_id = False
                        # get num of frames
                        processed_fname = os.path.join(sent.path, 'cropped_frames.hdf5')
                        if not os.path.exists(processed_fname):
                            continue
                        with h5py.File(processed_fname,'r') as f:
                            if 'images' not in f:
                                continue
                            num_frames = f['lmk_2d'].shape[0]
                        video_list[video_id] = num_frames
    print(f"there are {len(video_list)} sequences")
    with open('dataset/mead_25fps/processed/video_list_wimg.pkl', 'wb') as f:
        pickle.dump(sorted(video_list.items()), f)

def get_video_list_from_lmks(lmks_folder, audio_input_folder):
    video_list = {} # dict of video_id (sbj/view/emotion/level/sent) : num_frames
    view = 'front'
    print_id = True
    for subject in os.scandir(lmks_folder):
        for emotion in os.scandir(os.path.join(subject.path, view)):
            for level in os.scandir(emotion.path):
                for sent in os.scandir(level.path):
                    audio_path = os.path.join(audio_input_folder, subject.name, emotion.name, level.name, f"{sent.name}.pt")
                    if os.path.exists(audio_path):
                        video_id = '/'.join([subject.name, view, emotion.name, level.name, sent.name])
                        if print_id:
                            print(video_id)
                            print_id = False
                        # get num of frames
                        processed_fname = os.path.join(sent.path, 'landmarks_mediapipe.hdf5')
                        if not os.path.exists(processed_fname):
                            continue
                        with h5py.File(processed_fname,'r') as f:
                            num_frames = f['lmk_2d'].shape[0]
                        video_list[video_id] = num_frames
    print(f"there are {len(video_list)} sequences")
    with open('dataset/mead_25fps/processed/video_list_woimg.pkl', 'wb') as f:
        pickle.dump(sorted(video_list.items()), f)

if __name__ == '__main__':
    lmks_folder = 'dataset/mead_25fps/processed/cropped_landmarks'
    audio_input_folder = 'dataset/mead_25fps/processed/audio_inputs'
    get_video_list_from_lmks(lmks_folder, audio_input_folder)