import cv2 
import numpy as np
from skimage import transform as trans
from skimage.transform import estimate_transform, warp, resize, rescale
import torch
import tqdm 
import os 
from glob import glob
import torch.nn.functional as F
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


## from deca
def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn

def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d
    class C(object):
        pass
    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o

# process/generate vertices, normals, faces, borrowed from https://github.com/filby89/spectre/blob/master/src/utils/util.py
def generate_triangles(h, w, margin_x=2, margin_y=5, mask = None):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    #.
    # w*h
    triangles = []
    for x in range(margin_x, w-1-margin_x):
        for y in range(margin_y, h-1-margin_y):
            triangle0 = [y*w + x, y*w + x + 1, (y+1)*w + x]
            triangle1 = [y*w + x + 1, (y+1)*w + x + 1, (y+1)*w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:,[0,2,1]]
    return triangles

# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(), 
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(), 
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals
