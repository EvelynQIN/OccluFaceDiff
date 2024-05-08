import os 
import pickle
import h5py

def extract_lmk_from_image(images_folder, audio_input_folder, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    view = 'front'
    for subject in os.scandir(images_folder):
        for emotion in os.scandir(os.path.join(subject.path, view)):
            for level in os.scandir(emotion.path):
                for sent in os.scandir(level.path):
                    audio_path = os.path.join(audio_input_folder, subject.name, emotion.name, level.name, f"{sent.name}.pt")
                    if os.path.exists(audio_path):
                        # get num of frames
                        processed_fname = os.path.join(sent.path, 'cropped_frames.hdf5')
                        if not os.path.exists(processed_fname):
                            continue

                        with h5py.File(processed_fname,'r') as f:
                            lmk_2d = f['lmk_2d'][:]
                            valid_frames_idx = f['valid_frames_idx'][:]
                        out_subfolder = os.path.join(out_folder, subject.name, view, emotion.name, level.name, sent.name)
                        os.makedirs(out_subfolder)
                        out_fname = os.path.join(out_subfolder, 'landmarks_mediapipe.hdf5')
                        out_file = h5py.File(out_fname, 'w')
                        out_file.create_dataset('lmk_2d', data=lmk_2d)
                        out_file.create_dataset('valid_frames_idx', data=valid_frames_idx)
                        out_file.close()

if __name__ == '__main__':
    images_folder = 'dataset/mead_25fps/processed/images'
    audio_input_folder = 'dataset/mead_25fps/processed/audio_inputs'
    out_folder = 'dataset/mead_25fps/processed/cropped_landmarks'
    extract_lmk_from_image(images_folder, audio_input_folder, out_folder)