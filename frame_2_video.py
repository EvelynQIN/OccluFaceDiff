import imageio
import cv2 
import numpy as np
import h5py

img_path = 'dataset/mead_25fps/processed/images/M003/front/contempt/level_3/003/cropped_frames.hdf5'
gif_fname = 'frame2video.gif'

writer = imageio.get_writer(gif_fname, mode='I')

with h5py.File(img_path, "r") as f:
    image = f['images'][:]

print(image.shape)  # (n, 3, 224, 224)
image = (image.transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
print(image.shape)
for frame in image:
    writer.append_data(frame)
