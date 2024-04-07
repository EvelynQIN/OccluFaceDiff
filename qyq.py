import torch 
from glob import glob 
import os 
from tqdm import tqdm
import ffmpeg
import cv2




if __name__ == "__main__":
        video_path = "test_audio_align.mp4"
        audio_path = "dataset/multiface/m--20171024--0000--002757580--GHS/audio/SEN_are_you_looking_for_employment.wav"
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (333, 512))
        img_folder = 'dataset/multiface/m--20171024--0000--002757580--GHS/images/SEN_are_you_looking_for_employment/400016'
        for frame_path in sorted(glob(f"{img_folder}/*.png")):
                frame = cv2.imread(frame_path)
                video.write(frame)
        video.release()

        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(audio_path)

        ffmpeg.concat(input_video, input_audio, v=1, a=1).output('with_audio.mp4').run()