import torch 
import numpy as np 
from tqdm import tqdm
import glob 
import os 

def check(motion_paths, skip_frame, input_motion_length):
    motion_path_list = []
    # discard the motion sequence shorter than the specified length for train / val
    cnt = 0
    for motion_path in tqdm(motion_paths):
        motion = torch.load(motion_path)
        nframes = motion['target'].shape[0]
        if nframes < input_motion_length*skip_frame:
            continue
        
        # check null 
        flag = False
        for k in motion:
            if motion[k].isnan().any():
                flag = True
                print(f"{motion_path} {k} has nan")
        if flag:
            cnt += 1
        else:
            motion_path_list.append(motion_path)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(cnt)

    return motion_path_list

def main():
    skip_frame = 1
    input_motion_length = 120
    data_dir = 'processed_data/FaMoS'
    split = 'val'
    motion_paths = glob.glob(os.path.join(data_dir, split, f"*.pt"))
    val_paths = check(motion_paths, skip_frame, input_motion_length)
    # np.save(os.path.join(data_dir, 'valid_motion_paths', f'{split}.npy'), val_paths)

main()