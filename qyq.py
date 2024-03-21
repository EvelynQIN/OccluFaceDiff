import torch 
from glob import glob 
import os 
from tqdm import tqdm

folder = 'processed_data/FaMoS'

shape_gt = []
mica_shape = []
files = glob(os.path.join(folder, f"train/*"))
for f in tqdm(files):
        motion = torch.load(f)
        mica_shape.append(motion['mica_shape'][0, :100])
        shape_gt.append(motion['target'][0, :100])
        
mica_shape = torch.stack(mica_shape)
print(mica_shape.shape)
shape_gt = torch.stack(shape_gt) 
print(shape_gt.shape)

avg_error = torch.mean(
        torch.norm(mica_shape - shape_gt, 2, -1)
)
print('Avgerage error: ', avg_error)

mean_shape_gt = torch.mean(shape_gt, dim=0)
std_shape_gt = torch.std(shape_gt, dim=0)

mean_shape_mica = torch.mean(mica_shape, dim=0)
std_shape_mica = torch.std(mica_shape, dim=0)

print(f'mean_gt   = {mean_shape_gt}')
print(f'mean_mica = {mean_shape_mica}')
print(f'std_gt    = {std_shape_gt}')
print(f'std mica  = {std_shape_mica}')