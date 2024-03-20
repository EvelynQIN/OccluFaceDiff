import torch 
from glob import glob 
import os 
from tqdm import tqdm

folder = 'processed_data/FaMoS'

phases = ['test', 'val', 'train']

for phase in phases:
        files = glob(os.path.join(folder, f"{phase}/*"))
        for f in tqdm(files):
                motion = torch.load(f)
                motion['mica_shape'] = motion['mica_shape'].to('cpu')
                torch.save(motion, f)