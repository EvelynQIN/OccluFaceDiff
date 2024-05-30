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

audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h") 
dataset_path = 'dataset/RAVDESS'
audio_folder_path = os.path.join(dataset_path, 'audio')
processed_folder = os.path.join(dataset_path, 'processed')
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)
audio_input_folder = os.path.join(processed_folder, 'audio_inputs')
os.makedirs(audio_input_folder)

for audio in os.scandir(audio_folder_path):
    to_path = os.path.join(audio_input_folder, f"{audio.name[:-4]}.pt")
    speech_array, sampling_rate = librosa.load(audio.path, sr=16000)
    audio_values = audio_processor(
        speech_array, 
        return_tensors='pt', 
        padding="longest",
        sampling_rate=sampling_rate).input_values
                    
    torch.save(audio_values, to_path)