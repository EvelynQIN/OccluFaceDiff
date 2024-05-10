# Copied from EMOCA https://github.com/radekd91/inferno/blob/75f8f76352ad4fe9ee401c6e845228810eb7f459/inferno/utils/MediaPipeLandmarkLists.py#L5

from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_FACE_OVAL
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_IRISES
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYEBROW
# from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_IRIS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYEBROW
# from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_IRIS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
import numpy as np


def unpack_mediapipe_set(edge_set): 
    vertex_set = set()
    for i in edge_set: 
        vertex_set.add(i[0])
        vertex_set.add(i[1])
    return vertex_set

def face_center_landmark_indices(sorted=True):
    face_center = list(unpack_mediapipe_set(FACEMESH_LIPS) \
        .union(unpack_mediapipe_set(FACEMESH_LEFT_EYE)) \
        .union(unpack_mediapipe_set(FACEMESH_RIGHT_EYE))) 
    if sorted: 
        face_center.sort()
    face_center = np.array(face_center, dtype=np.int32)
    return face_center

def left_eye_eyebrow_landmark_indices(sorted=True): 
    left_eye = list(unpack_mediapipe_set(FACEMESH_LEFT_EYE) \
        # .union(unpack_mediapipe_set(FACEMESH_LEFT_IRIS)) \
        .union(unpack_mediapipe_set(FACEMESH_LEFT_EYEBROW)))
    if sorted: 
        left_eye.sort()
    left_eye = np.array(left_eye, dtype=np.int32)
    return left_eye

def right_eye_eyebrow_landmark_indices(sorted=True): 
    right_eye = list(unpack_mediapipe_set(FACEMESH_RIGHT_EYE) \
        # .union(unpack_mediapipe_set(FACEMESH_RIGHT_IRIS)) \
        .union(unpack_mediapipe_set(FACEMESH_RIGHT_EYEBROW)))
    if sorted: 
        right_eye.sort()
    right_eye = np.array(right_eye, dtype=np.int32)
    return right_eye

def left_eye_landmark_indices(sorted=True): 
    left_eye = list(unpack_mediapipe_set(FACEMESH_LEFT_EYE))
    if sorted: 
        left_eye.sort()
    left_eye = np.array(left_eye, dtype=np.int32)
    return left_eye

def right_eye_landmark_indices(sorted=True): 
    right_eye = list(unpack_mediapipe_set(FACEMESH_RIGHT_EYE))
    if sorted:
        right_eye.sort()
    right_eye = np.array(right_eye, dtype=np.int32)
    return right_eye

def mouth_landmark_indices(sorted=True): 
    mouth = list(unpack_mediapipe_set(FACEMESH_LIPS)) 
    if sorted: 
        mouth.sort()
    mouth = np.array(mouth, dtype=np.int32)
    return mouth

def face_oval_landmark_indices(sorted=True): 
    face_oval = list(unpack_mediapipe_set(FACEMESH_FACE_OVAL))
    if sorted: 
        face_oval.sort()
    face_oval = np.array(face_oval, dtype=np.int32)
    return face_oval

def all_face_landmark_indices(sorted=True): 
    face_all = list(unpack_mediapipe_set(FACEMESH_TESSELATION))
    if sorted: 
        face_all.sort()
    face_all = np.array(face_all, dtype=np.int32)
    return face_all

# mediapipe landmark embedding indices
EMBEDDING_INDICES = [
    276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,
    55,  63,  65,  66,  70, 105, 107, 249, 263, 362, 373, 374, 380,
    381, 382, 384, 385, 386, 387, 388, 390, 398, 466, 7,  33, 133,
    144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
    168,   6, 197, 195,   5,   4, 129,  98,  97,   2, 326, 327, 358,
    0,  13,  14,  17,  37,  39,  40,  61,  78,  80,  81,  82,  84,
    87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
    308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
    415]

sorter = np.argsort(EMBEDDING_INDICES)

# LIP
# upper inner + outer + left corder (inner + outer)
UPPER_LIP_IDX = [191,  80, 81 , 82 , 13 , 312, 311, 310, 415] + [185,  40,  39,  37, 0, 267, 269, 270, 409] + [308, 291]

# lowr inner + outer + right corder (inner + outer)
LOWER_LIP_IDX = [ 95,  88, 178, 87 , 14 , 317, 402, 318, 324] + [146,  91, 181,  84, 17, 314, 405, 321, 375] + [78, 61]

# LEFT EYE
# perspective of the landmarked person
LEFT_EYE_LEFT_CORNER = 263
LEFT_EYE_RIGHT_CORNER = 362 
# the upper and lower eyelid points are in correspondences, ordered from right to left (perspective of the landmarked person)
LEFT_UPPER_EYELID_INDICES = [398, 384, 385, 386, 387, 388, 466]
LEFT_LOWER_EYELID_INDICES = [382, 381, 380, 374, 373, 390, 249]

LEFT_UPPER_EYEBROW_INDICES = [336, 296, 334, 293, 300]
LEFT_LOWER_EYEBROW_INDICES = [285, 295, 282, 283, 276]

# RIGHT EYE
# perspective of the landmarked person
RIGHT_EYE_LEFT_CORNER = 133
RIGHT_EYE_RIGHT_CORNER = 33 
# the upper and lower eyelid points are in correspondences, ordered from right to left (perspective of the landmarked person)
RIGHT_UPPER_EYELID_INDICES = [246, 161, 160, 159, 158, 157, 173]
RIGHT_LOWER_EYELID_INDICES = [7  , 163, 144, 145, 153, 154, 155]

RIGHT_UPPER_EYEBROW_INDICES = [ 70,  63, 105,  66, 107]
RIGHT_LOWER_EYEBROW_INDICES = [ 46,  53,  52,  65,  55]

UPPER_EYELIDS = np.array(LEFT_UPPER_EYELID_INDICES + RIGHT_UPPER_EYELID_INDICES, dtype=np.int64)
LOWER_EYELIDS = np.array(LEFT_LOWER_EYELID_INDICES + RIGHT_LOWER_EYELID_INDICES, dtype=np.int64) 

# em correspondances
UPPER_LIP_EM = sorter[np.searchsorted(EMBEDDING_INDICES, UPPER_LIP_IDX, sorter=sorter)]
LOWER_LIP_EM = sorter[np.searchsorted(EMBEDDING_INDICES, LOWER_LIP_IDX, sorter=sorter)]
UPPER_EYELIDS_EM = sorter[np.searchsorted(EMBEDDING_INDICES, UPPER_EYELIDS, sorter=sorter)]
LOWER_EYELIDS_EM = sorter[np.searchsorted(EMBEDDING_INDICES, LOWER_EYELIDS, sorter=sorter)]

LIP_EM = UPPER_LIP_EM + LOWER_LIP_EM
