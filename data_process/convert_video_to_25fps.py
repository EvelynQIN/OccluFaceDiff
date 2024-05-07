import os 
import glob 
from tqdm import tqdm
import pickle
import argparse

def convert_to_25fps(MEAD_path, out_folder, subject_list, fps=25):
    # args = ['ffmpeg', '-r', '1', '-i', str(video_file), '-r', '1', str(out_format)]
    # out_format = ' -r 1 -i %s ' % str(video_file) + ' "' + "$frame.%03d.png" + '"'
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
    MEAD_path = 'dataset/MEAD'
    out_folder = 'dataset/mead_25fps/processed/videos_25fps'

    parser = argparse.ArgumentParser(description='resample video to 25 fps')

    parser.add_argument('--subject_id', default=0, type=int, help='subject_id to process')
    parser.add_argument('--num_subjects', default=0, type=int, help='number of subjects to process together')

    args = parser.parse_args()

    mead_subjects = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 'M030', 'M031', 'M032', 'M033', 'M034', 'M035', 'M037', 'M039', 'M040', 'M041', 'M042', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018', 'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029', 'W033', 'W035', 'W036', 'W037', 'W038', 'W040']

    subject_list = mead_subjects[args.subject_id:args.subject_id+args.num_subjects]
    convert_to_25fps(MEAD_path, out_folder, subject_list)

    get_video_list(out_folder)