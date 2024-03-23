import cv2 
import numpy as np
from skimage import transform as trans
from skimage.transform import estimate_transform, warp, resize, rescale
import torch
import tqdm 
import os 
from glob import glob

input_mean = 127.5
input_std = 127.5

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def get_arcface_input(lmk, img):
    """get the arcface norm_cropped image input

    Args:
        lmk: 5 2d landmarks, (5, 2)
        img: BGR image input

    Returns:
        arcface_input: [3, 112, 112]
    """
    aimg = norm_crop(img, landmark=lmk)
    blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
    return blob[0]

def crop(lmk, trans_scale, scale, image_size=224):
    left = torch.min(lmk[:,0]); right = torch.max(lmk[:,0]); 
    top = torch.min(lmk[:,1]); bottom = torch.max(lmk[:,1])

    old_size = (right - left + bottom - top)/2
    center = torch.FloatTensor([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    # translate center
    trans_scale = (torch.rand(2)*2 -1) * trans_scale
    center = center + trans_scale*old_size # 0.5

    # scale = torch.rand(1) * (self.scale[1] - self.scale[0]) + self.scale[0]
    size = int(old_size*scale)

    # crop image
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    DST_PTS = np.array([[0,0], [0,image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    
    return torch.from_numpy(tform.params).float()

def crop_np(lmk, trans_scale, scale, image_size=224):
    left = np.min(lmk[:,0]); right = np.max(lmk[:,0]); 
    top = np.min(lmk[:,1]); bottom = np.max(lmk[:,1])

    old_size = (right - left + bottom - top)/2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    # translate center
    trans_scale = (np.random.rand(2)*2 -1) * trans_scale
    center = center + trans_scale*old_size # 0.5

    # scale = torch.rand(1) * (self.scale[1] - self.scale[0]) + self.scale[0]
    size = int(old_size*scale)

    # crop image
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    DST_PTS = np.array([[0,0], [0,image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    
    return tform

def batch_crop_lmks(lmks, verts_2d, trans_scale, scale, image_size=224):
    """
    Args:
        lmks: (bs, V, 2)
        verts_2d; (bs, flameV, 2)
    Returns:
        lmks_cropped: (bs, V, 2)
    """
    lmks_cropped = torch.zeros_like(lmks)
    verts_cropped = torch.zeros_like(verts_2d)
    for i in range(lmks.shape[0]):
        lmk = lmks[i]
        verts = verts_2d[i]
        # crop information
        tform = crop(lmk, trans_scale, scale, image_size)
        ## crop 
        cropped_lmk = torch.matmul(tform, torch.hstack([lmk, torch.ones([lmk.shape[0],1])]).transpose(0, 1)).transpose(0, 1)[:,:2] 
        # normalized kpt
        lmks_cropped[i] = cropped_lmk/image_size * 2  - 1

        ## crop 
        cropped_verts = torch.matmul(tform, torch.hstack([verts, torch.ones([verts.shape[0],1])]).transpose(0, 1)).transpose(0, 1)[:,:2] 
        # normalized kpt
        verts_cropped[i] = cropped_verts/image_size * 2  - 1

    return lmks_cropped, verts_cropped

def batch_normalize_lmk_3d(lmk_3d):
    """
    normalize 3d landmarks s.t. the len between no.30 and no.27 = 1
    set the root to be the no.30 lmk (nose_tip)
    Args:
        lmk_3d: tensor (bs, n, 3)
    Returns:
        lmk_3d_normed: tensor (bs, n, 3)
    """
    root_idx = 30
    pivot_idx = 27
    bs, num_lmk, _ = lmk_3d.shape
    root_node = lmk_3d[:, root_idx] # (bs, 3)
    nose_len = torch.norm(lmk_3d[:, pivot_idx]-root_node, 2, -1)    # (bs, )
    lmk_3d_normed = lmk_3d - root_node.unsqueeze(1).expand(-1, num_lmk, -1)
    lmk_3d_normed = torch.divide(lmk_3d_normed, nose_len.reshape(-1, 1, 1).expand(-1, num_lmk, 3))
    return lmk_3d_normed

