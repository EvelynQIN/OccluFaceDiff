import os 
from glob import glob
dataset_path = 'dataset/RAVDESS'
video_path = os.path.join(dataset_path, 'video')
audio_path = os.path.join(dataset_path, 'audio')

zip_path = 'dataset/RAVDESS/video/zip'

# Filename identifiers 

# Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
# Vocal channel (01 = speech, 02 = song).
# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
# Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
# Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
# Repetition (01 = 1st repetition, 02 = 2nd repetition).
# Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

# for zip_file in os.scandir(zip_path):
#     os.system(f"unzip {zip_file.path} -d {video_path}")
#     os.system(f"rm {zip_file.path}")

# filter out unneeded file (only take repetition1, and remove actor folders)

# # filter audio
# for actor in os.scandir(audio_path):
#     for audio in os.scandir(actor.path):
#         modality, vocal, emotion, intensity, sent, rep, sbj = audio.name[:-4].split('-')
#         if rep == '02':
#             os.system(f"rm {audio.path}")

# # filter video
# for actor in os.scandir(video_path):
#     for video in os.scandir(actor.path):
#         modality, vocal, emotion, intensity, sent, rep, sbj = video.name[:-4].split('-')
#         if rep == '02':
#             os.system(f"rm {video.path}")

# convert video into 25 fps
fps = 25 
video_25fps_folder_path = os.path.join(dataset_path, 'video_25fps')
if not os.path.exists(video_25fps_folder_path):
    os.makedirs(video_25fps_folder_path)

for actor in os.scandir(video_path):
    for video in os.scandir(actor.path):
        modality, vocal, emotion, intensity, sent, rep, sbj = video.name[:-4].split('-')
        to_path = os.path.join(video_25fps_folder_path, video.name)
        if os.path.exists(to_path):
            continue
        os.system(
            f"ffmpeg -i {video.path} -r {fps} {to_path}"
        )

# # move audio file into audio folder 
# for actor in os.scandir(audio_path):
#     for audio in os.scandir(actor.path):
#         modality, vocal, emotion, intensity, sent, rep, sbj = audio.name[:-4].split('-')
#         os.system(f"mv {audio.path} {audio_path}/")
#     os.system(f"rm -rf {actor.path}")

print(f"len audio {len(os.listdir(audio_path))}")
print(f"len video {len(os.listdir(video_25fps_folder_path))}")

# filter out 01 modality
for video in os.scandir(video_25fps_folder_path):
    modality, vocal, emotion, intensity, sent, rep, sbj = video.name[:-4].split('-')
    if modality == '01':
        os.system(
            f"rm {video.path}"
        )
print(f"len video {len(os.listdir(video_25fps_folder_path))}")

