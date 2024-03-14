import numpy as np 
import open3d as o3d 
import torch       
import yaml 
import pickle
import sys 
sys.path.append('../') # to access util
from utils.utils_visualize import mesh_sequence_to_video 
from utils.model_util import find_model_using_name
from model.FLAME import FLAME

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
# check shape 100 and expression 50 
flame_0 = FLAME(flame_model_path='../flame_2020/generic_model.pkl', 
              flame_lmk_embedding_path='../flame_2020/dense_lmk_embedding.npy',
              n_shape=100, n_exp=50)

flame_1 = FLAME(flame_model_path='../flame_2020/generic_model.pkl', 
              flame_lmk_embedding_path='../flame_2020/dense_lmk_embedding.npy',
              n_shape=300, n_exp=100)

motion_path = '../dataset/FaMoS/flame_params/FaMoS_subject_001/bareteeth.npy'
motion = np.load(motion_path, allow_pickle=True)[()]

shape = torch.Tensor(motion["flame_shape"])
expression = torch.Tensor(motion["flame_expr"])
rot_aa = torch.Tensor(motion["flame_pose"]) # full poses exluding eye poses (root, neck, jaw, left_eyeball, right_eyeball)
trans = torch.Tensor(motion['flame_trans'])

verts_reduced, _ = flame_0(shape[:,:100], expression[:,:50], rot_aa, trans)
verts_gt, _ = flame_1(shape, expression, rot_aa, trans)

error = torch.mean(torch.norm(verts_gt-verts_reduced, p=2, dim=2)) 

print("Error is ", error)

# mesh_sequence_to_video(mesh_verts = verts_reduced, faces=flame_0.faces_tensor, video_path='shape_100_exp_100.mp4', fps=60)
# mesh_sequence_to_video(mesh_verts = verts_gt, faces=flame_0.faces_tensor, video_path='shape_300_exp_100.mp4', fps=60)

# check mica's shape prediction 
device = 'cuda:0'
cfg_path = '../configs/mica.yaml'
with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)
args = Struct(**cfg)
mica = find_model_using_name(model_dir='model', model_name=args.name)(args)
mica.load_model(device)
mica.to(device)
mica.eval()

# load image input
motion_dict_path = '../processed_data/FaMoS/train/subject_001_bareteeth.pt'
motion_processed = torch.load(motion_dict_path)
arcface_input = motion_processed['arcface_input'].to(device)
img_mask = motion_processed['img_mask']

identity_codes = mica.encode(arcface_input.unsqueeze(0))
print(identity_codes.shape)
pred_shape = mica.decode(identity_codes).detach().cpu()
print(pred_shape.shape)

shape_error = torch.mean(torch.norm(pred_shape[0, :100]-shape[0, :100], p=2, dim=-1))
print(f"Shape Error of mica for single images is {shape_error}") 

flame_vmask_path = "../flame_2020/FLAME_masks.pkl"
with open(flame_vmask_path, 'rb') as f:
    flame_v_mask = pickle.load(f, encoding="latin1")
face_mask = torch.from_numpy(flame_v_mask['face']).long()

verts_mica, _ = flame_0(pred_shape.repeat(shape.shape[0], 1)[:, :100], expression[:,:50], rot_aa, trans)

verts_error = torch.mean(torch.norm(verts_mica-verts_reduced, p=2, dim=-1))
print("mica verts error is on reduced flame params: ", verts_error)

verts_error = torch.mean(torch.norm(verts_mica-verts_gt, p=2, dim=-1))
print("mica verts error is on gt flame params: ", verts_error)
# mesh_sequence_to_video(mesh_verts = verts_mica, faces=flame_0.faces_tensor, video_path='mica_shape100_exp50.mp4', fps=60)

# vertice error on the front face region
verts_error = torch.mean(torch.norm(verts_mica[:,face_mask]-verts_reduced[:,face_mask], p=2, dim=-1))
print("[face] mica verts error is on reduced flame params: ", verts_error)

verts_error = torch.mean(torch.norm(verts_mica[:, face_mask]-verts_gt[:,face_mask], p=2, dim=-1))
print("[face] mica verts error is on gt flame params: ", verts_error)

