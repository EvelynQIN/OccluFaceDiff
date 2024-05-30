from transformers import Wav2Vec2Processor
import librosa
import numpy as np  
import pickle
import torch
import warnings
warnings.filterwarnings('ignore')
import os 
from glob import glob 
from tqdm import tqdm

video_id_to_sent_id = np.load(os.path.join('dataset/mead_25fps/' 'processed/video_id_to_sent_id.npy'), allow_pickle=True)[()]
mead_subjects = ['M007', 'M009', 'W011']

def preoprocess_audio(datafolder, to_folder):
    if not os.path.exists(to_folder):
        os.makedirs(to_folder)
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h") 
    n_subj = 0
    empty_audio_list = []
    view = 'front'
    for subject in os.scandir(datafolder):
        if subject.name not in mead_subjects:
            continue
        n_subj += 1
        print(f"process No. {n_subj}__{subject.name}")
        
        if subject.name == 'meta':
            continue
        audio_folder = os.path.join(subject.path, 'audio')
        for emotion in os.scandir(audio_folder):
            for level in os.scandir(emotion.path):
                for audio in os.scandir(level.path):
                    sent = audio.name[:-4]
                    
                    # only extract test audio 
                    video_id = '/'.join([subject.name, view, emotion.name, level.name, sent])
                    sent_id = video_id_to_sent_id[video_id]
                    if sent_id >= 10:
                        continue

                    audio_folder = os.path.join(to_folder, subject.name, emotion.name, level.name)
                    if not os.path.exists(audio_folder):
                        os.makedirs(audio_folder)
                    to_path = os.path.join(audio_folder, f'{audio.name[:-4]}.pt')
                    if os.path.exists(to_path):
                        continue
                    try:
                        speech_array, sampling_rate = librosa.load(audio.path, sr=16000)
                    except:
                        empty_audio_list.append((subject.name, emotion.name, level.name, audio.name[:-4]))
                        continue
                    audio_values = audio_processor(
                        speech_array, 
                        return_tensors='pt', 
                        padding="longest",
                        sampling_rate=sampling_rate).input_values
                    
                    torch.save(audio_values, to_path)
    
    anomalous_path = os.path.join(os.path.split(to_folder)[0], 'bad_audios.pkl')
    print(len(empty_audio_list))
    with open(anomalous_path, 'wb') as f:
        pickle.dump(empty_audio_list, f)

if __name__ == "__main__":

    # extract audio_input and save it as .pt file
    datafolder = 'dataset/mead_25fps/original_data'
    to_folder = 'dataset/mead_25fps/processed/audio_inputs'
    preoprocess_audio(datafolder, to_folder)