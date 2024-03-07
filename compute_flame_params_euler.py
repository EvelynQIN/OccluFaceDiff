'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about VOCA is available at http://voca.is.tue.mpg.de.
For comments or questions, please email us at voca@tue.mpg.de
'''

import os
import glob
import argparse
from tqdm import tqdm
import numpy as np
import chumpy as ch
# from scipy.sparse.linalg import cg
import scipy.sparse as sp
import trimesh
from smpl_webuser.serialization import load_model

from time import time
import json

def load_embedding( file_path ):
    """ funciton: load landmark embedding, in terms of face indices and barycentric coordinates for corresponding landmarks
    note: the included example is corresponding to CMU IntraFace 49-point landmark format.
    """
    lmk_indexes_dict = np.load(file_path, allow_pickle=True, encoding='latin1')[()]
    lmk_face_idx = lmk_indexes_dict['static_lmk_faces_idx'].astype( np.uint32 )
    lmk_b_coords = lmk_indexes_dict[ 'static_lmk_bary_coords']
    return lmk_face_idx, lmk_b_coords

def mesh_points_by_barycentric_coordinates( mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords ):
    """ function: evaluation 3d points given mesh and landmark embedding
    """
    dif1 = ch.vstack([(mesh_verts[mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=1),
                    (mesh_verts[mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=1),
                    (mesh_verts[mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=1)]).T
    return dif1


def landmark_error_3d( mesh_verts, mesh_faces, gt_verts, lmk_face_idx, lmk_b_coords, weight=1.0 ):
    """ function: 3d landmark error objective
    """
    # groudtruch landmarks
    lmk_3d = mesh_points_by_barycentric_coordinates( gt_verts, mesh_faces, lmk_face_idx, lmk_b_coords )
    
    # select corresponding vertices
    v_selected = mesh_points_by_barycentric_coordinates( mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords )
    lmk_num  = lmk_face_idx.shape[0]

    # an index to select which landmark to use
    lmk_selection = np.arange(0,lmk_num).ravel() # use all

    # residual vectors
    lmk3d_obj = weight * ( v_selected[lmk_selection] - lmk_3d[lmk_selection] )

    return lmk3d_obj

def compute_FLAME_params_for_one_mesh(source_path, model, weights, lmk_face_idx, lmk_b_coords,  fix_shape=None, show_fitting=False):
    '''
    Fit a flame model for one mesh with FLAME toplogy, and compute the relative flame params
    :param source_path:         path of the registered mesh file
    :param model:   the loaded flame model
    '''
    
    # load the target mesh in FLAME topology
    target_mesh = trimesh.load_mesh(source_path, process=False)
    frame_vertices = target_mesh.vertices
    # print(frame_vertices.shape)

    # use previous frame fitting result as the starting points
    # model.trans[:] = 0.0
    # model.betas[:] = np.random.rand( model.betas.size ) * 0.0  # initialized to zero
    # model.pose[:]  = np.random.rand( model.pose.size ) * 0.0   # initialized to zero
    
    if fix_shape is not None:
        model.betas[:300] = fix_shape
        free_variables = [model.trans, model.pose, model.betas[300:]]
    else:
        free_variables = [model.trans, model.pose, model.betas]
        
    # objectives
    lmk_err = landmark_error_3d(mesh_verts=model, mesh_faces=model.f,  gt_verts = frame_vertices, lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords)        
    mesh_dist = frame_vertices - model
    pose_reg = weights['pose'] * model.pose[3:]
    shape_reg = weights['shape'] * model.betas[:300]
    exp_reg = weights['expr'] * model.betas[300:]
    
    objectives = {
        'mesh_dist': weights['mesh_dist'] * mesh_dist, 
        'lmk_3d': weights['lmk_3d'] * lmk_err,
        'shape': shape_reg, 
        'expr': exp_reg, 
        'pose': pose_reg
        # 'neck_pose': neck_pose_reg, 
        # 'jaw_pose': jaw_pose_reg,
        # 'eyeballs': eyeballs_pose_reg
    } 
    
    opt_options = {}
    opt_options['disp']    = 1
    opt_options['delta_0'] = 0.1
    opt_options['e_3']     = 1e-4
    opt_options['maxiter'] = 2000
    sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
    opt_options['sparse_solver'] = sparse_solver

    # rigid alignment -> only optimize for global orientation
    # timer_start = time()
    # print("step1: start rigid fitting ...")
    ch.minimize( fun      = mesh_dist,
                 x0       = [ model.trans, model.pose[:3] ],
                 method   = 'dogleg',
                 options  = opt_options )
    # timer_end = time()
    # print("step 1: fitting done, in %f sec\n" % ( timer_end - timer_start ))

    
    # non-rigid alignment
    # timer_start = time()
        
    ch.minimize( fun      = objectives,
                 x0       = free_variables,
                 method   = 'dogleg',
                 options  = opt_options )
    # timer_end = time()
    # print("step 2: fitting done, in %f sec\n" % ( timer_end - timer_start ))
    
    
    if show_fitting:
        from psbody.mesh.meshviewer import MeshViewer
        from psbody.mesh import Mesh
        import six
        # Visualize fitting
        mv = MeshViewer()
        fitting_mesh = Mesh(model.r, model.f)
        fitting_mesh.set_vertex_colors('light sky blue')

        mv.set_static_meshes([target_mesh, fitting_mesh])
        six.moves.input('Press key to continue')
    
    return model

def compute_frame_list_for_FaMoS(image_paths):
    frame_list = {}
    subjects = sorted(os.listdir(image_paths))
    for subject in subjects:
        frame_list[subject] = {}
        expr_path = os.path.join(image_paths, subject)
        expressions = sorted(os.listdir(expr_path))
        for expr in expressions:
            frame_list[subject][expr] = sorted(os.listdir(os.path.join(expr_path, expr)))
    np.save("dataset/FaMoS/frame_list.npy", frame_list)

def compute_flame_params_for_dataset(source_path, to_folder, flame_model_path, weights, lmk_face_idx, lmk_b_coords, subject_selected, subject_genders = None):
    """
    Compute the flame parameters and the flame verts position for all registered meshes under the folder path

    Args:
        source_path: path to the folder that contains the registered meshes
        to_folder: where to save the computed flame parameters
        flame_model_path: the path to the flame model folder
    """

    if subject_genders is None:
        flame_model_fname = os.path.join(flame_model_path, "generic_model.pkl")
    else:
        flame_model_fname = os.path.join(flame_model_path, "generic_model.pkl") # TODO
    
    print("processing subjects for list ", subject_selected)
    for subject in subject_selected:
        print("Processing ", subject) 
        expressions = sorted(os.listdir(os.path.join(source_path, subject)))
        out_path = os.path.join(to_folder, subject)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        fit_shape = True
        
        for expr in tqdm(expressions):
            try:
                params_out_fname = os.path.join(out_path, f'{expr}.npy')
                if os.path.isfile(params_out_fname):
                    fitted_res = np.load(params_out_fname, allow_pickle=True)[()]
                    num_frames = len(fitted_res["flame_verts"])
                    fit_shape = False
                    prev_motion = np.load(os.path.join(out_path, 'anger.npy'), allow_pickle=True)[()]
                    fix_shape = prev_motion["flame_shape"][0]
                    continue
                    
                print("processing ", expr)
                sequence_fnames = sorted(glob.glob(os.path.join(source_path, subject, expr, '*.ply')))
                num_frames = len(sequence_fnames)
                output = {}
                output['flame_trans'] =  np.zeros((num_frames, 3))
                output['flame_pose'] = np.zeros((num_frames, 15))
                output['flame_shape'] = np.zeros((num_frames, 300))
                output['flame_expr'] = np.zeros((num_frames, 100))
                output['flame_verts'] = np.zeros((num_frames, 5023, 3))
                output['frame_id'] = np.zeros((num_frames))
                
                # load the flame model
                model = load_model(flame_model_fname) 
                for i, frame in enumerate(sequence_fnames):
                    output['frame_id'][i] = os.path.split(frame)[-1].split('.')[1]
                    if fit_shape:
                        fitting_model = compute_FLAME_params_for_one_mesh(frame, model, weights, lmk_face_idx, lmk_b_coords)
                        fix_shape = fitting_model.betas.r[:300].copy()
                        fit_shape = False
                    else:
                        fitting_model = compute_FLAME_params_for_one_mesh(frame, model, weights, lmk_face_idx, lmk_b_coords, fix_shape)
                    output['flame_trans'][i] =  fitting_model.trans.r.copy()
                    output['flame_pose'][i] = fitting_model.pose.r.copy()
                    output['flame_shape'][i] = fitting_model.betas.r[:300].copy()
                    output['flame_expr'][i] = fitting_model.betas.r[300:].copy()
                    output['flame_verts'][i] = fitting_model.r.copy()
                np.save(params_out_fname, output)
            except:
                print(f"skipping {expr} for subject {subject}")
                continue

        
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get flame params out of 3d meshes')

    parser.add_argument('--source_path', default='', help='input sequence path')
    parser.add_argument('--to_folder', default='', help='path to folder where flame params can be saved')
    parser.add_argument('--flame_model_path', default='./flame_2020', help='path to the FLAME model')
    parser.add_argument('--start_sid', type=int, default=0, help='subject id to start fitting')
    # parser.add_argument('--num_subjects', type=int, default=1, help='number of subjects to fit')

    args = parser.parse_args()
    source_path = args.source_path
    flame_model_path = args.flame_model_path
    to_folder = args.to_folder
    
    # landmark embedding
    lmk_emb_path = 'flame_2020/landmark_embedding.npy' 
    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
    print("loaded lmk embedding")
    
    # weight of the loss
    weights = {}
    weights['mesh_dist'] = 2.0
    weights['shape'] = 1e-4
    weights['expr']  = 1e-4   
    weights['pose'] = 1e-3    
    weights['lmk_3d'] = 1e-2
    
    sid = args.start_sid-1
    # num_subjects = args.num_subjects
    
    subject_selected = [sorted(os.listdir("dataset/FaMoS/registrations"))[sid]]
    
    compute_flame_params_for_dataset(source_path, to_folder, flame_model_path, weights, lmk_face_idx, lmk_b_coords, subject_selected)
    
    # img_folder = 'dataset/FaMoS/downsampled_images_4'
    # compute_frame_list_for_FaMoS(img_folder)