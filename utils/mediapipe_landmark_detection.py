import cv2 
import numpy as np 

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp

# wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def mediapipe2np(landmarks): 
    # d = protobuf_to_dict(landmarks)
    array = np.zeros(shape=(len(landmarks), 3))
    for i in range(len(landmarks)):
        array[i, 0] = landmarks[i].x
        array[i, 1] = landmarks[i].y
        array[i, 2] = landmarks[i].z
    return array

class MediapipeDetector:
    def __init__(self, video_based=False, max_faces=1, threshold=0.1):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=not video_based,
            refine_landmarks=True,
            max_num_faces=max_faces,
            min_detection_confidence=threshold)

    def detect(self, image):
        """
        Args:
            image: rgb in uint8, (h, w, 3)
        """
        results = self.face_mesh.process(image)
        h, w, _ = image.shape

        if not results.multi_face_landmarks: 
            # this is a really weird thing, but somehow (especially when switching from one video to another) nothing will get picked up on the 
            # first run but it will be after the second run.
            results = self.face_mesh.process(image) 
        
        if not results.multi_face_landmarks:
            print("no face detected by mediapipe")
            return None
        
        for face_landmarks in results.multi_face_landmarks:
            landmarks = mediapipe2np(face_landmarks.landmark)
            
            # scale landmarks to image size
            landmarks = landmarks * np.array([w, h, 1]) # (478, 3)
        # print(landmarks.shape)
        return landmarks