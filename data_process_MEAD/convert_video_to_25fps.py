import os 
import glob 
from tqdm import tqdm
import pickle
import argparse

def convert_to_25fps(MEAD_path, out_folder, subject_list, fps=25):
    """convert videos from MEAD into 25 fps
    """
    view = 'front'  # currently only focus on the front view
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for subject in subject_list:
        print(f"process subject {subject}")
        if subject == 'W021':
            video_path = os.path.join(subject, 'video/1', view)
        else:
            video_path = os.path.join(subject, 'video', view)
        if not os.path.exists(os.path.join(MEAD_path, video_path)):
            print(f"{subject} does not any videos.")
            continue
        emotions = [emotion for emotion in os.scandir(os.path.join(MEAD_path, video_path))]
        for emotion in tqdm(emotions):
            for level in os.scandir(emotion.path):
                level_folder = os.path.join(out_folder, video_path, emotion.name, level.name)
                if not os.path.exists(level_folder):
                    os.makedirs(level_folder)
                for sent in os.scandir(level.path):
                    # resample video to 25fps
                    to_path = os.path.join(level_folder, sent.name)
                    if os.path.exists(to_path):
                        continue
                    os.system(
                        f"ffmpeg -i {sent.path} -r {fps} {to_path}"
                    )
                    os.system(f"rm {sent.path}")

def get_video_list(out_folder):
    view = 'front'
    video_list = []
    for subject in os.scandir(out_folder):
        for emotion in os.scandir(os.path.join(subject.path, 'video', view)):
            for level in os.scandir(emotion.path):
                for sent in os.scandir(level.path):
                    video_id = '/'.join(subject.name, view, emotion.name, level.name, sent.name[:-4])
                    video_list.append(video_id)
    video_list.sort()
    print(f"there are {len(video_list)} videos resampled successfully!")
    with open('dataset/mead_25fps/processed/video_list.pkl', 'wb') as f:
        pickle.dump(video_list, f)

if __name__ =="__main__":
    MEAD_path = 'dataset/mead_25fps/original_data'
    out_folder = 'dataset/mead_25fps/processed/videos_25fps'

    mead_subjects = os.listdir(MEAD_path)
    convert_to_25fps(MEAD_path, out_folder, mead_subjects)

    # get_video_list(out_folder)