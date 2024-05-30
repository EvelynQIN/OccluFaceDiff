import os 
import pickle
import h5py
import librosa
import torch

def read_audio(audio_input_folder):
    for subject in os.scandir(audio_input_folder):
        for emotion in os.scandir(os.path.join(subject.path)):
            for level in os.scandir(emotion.path):
                for audio in os.scandir(level.path):
                    # read data once
                    audio_path = os.path.join('dataset/MEAD', subject.name, 'audio', emotion.name, level.name, f"{audio.name[:-3]}.m4a")
                    if not os.path.exists(audio_path):
                        print(f"{audio_path} not existed!")
                    else:
                        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
                    audio_input = torch.load(audio.path)

def read_cropped_landmarks(landmark_folder):
    for subject in os.scandir(landmark_folder):
        for emotion in os.scandir(os.path.join(subject.path, 'front')):
            for level in os.scandir(emotion.path):
                for sent in os.scandir(level.path):
                    processed_fname = os.path.join(sent.path, 'landmarks_mediapipe.hdf5')
                    if not os.path.exists(processed_fname):
                        print(f"{processed_fname} not existed!")
                    with h5py.File(processed_fname,'r') as f:
                        num_frames = f['lmk_2d'].shape[0]
def read_reconstructions(rec_folder):
    for subject in os.scandir(rec_folder):
        if subject.name == 'W021':
            path = os.path.join(subject.path, '1/front')
        else:
            path = os.path.join(subject.path, 'front')
        for emotion in os.scandir(path):
            for level in os.scandir(emotion.path):
                for sent in os.scandir(level.path):
                    processed_fname = os.path.join(sent.path, 'appearance.hdf5')
                    if not os.path.exists(processed_fname):
                        print(f"{processed_fname} not existed!")
                    with h5py.File(processed_fname,'r') as f:
                        num_frames_1 = f['light'].shape[0]

                    processed_fname = os.path.join(sent.path, 'shape_pose_cam.hdf5')
                    if not os.path.exists(processed_fname):
                        print(f"{processed_fname} not existed!")
                    with h5py.File(processed_fname,'r') as f:
                        num_frames_2 = f['shape'].shape[0]
                    
                    assert num_frames_1 == num_frames_2
                    

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
    # with open('dataset/mead_25fps/processed/video_list_wimg.pkl', 'wb') as f:
    #     pickle.dump(sorted(video_list.items()), f)

if __name__ == '__main__':
    images_folder = 'dataset/mead_25fps/processed/images'
    audio_input_folder = 'dataset/mead_25fps/processed/audio_inputs'
    cropped_landmark_folder = 'dataset/mead_25fps/processed/cropped_landmarks'
    rec_folder = 'dataset/mead_25fps/processed/reconstructions/EMICA-MEAD_flame2020'

    # read image
    # get_video_list_from_images(images_folder, audio_input_folder)

    # read audio
    # read_audio(audio_input_folder)

    # read landmarks
    # read_cropped_landmarks(cropped_landmark_folder)

    # read recon
    read_reconstructions(rec_folder)
    