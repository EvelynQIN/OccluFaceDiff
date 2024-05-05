from utils.MediaPipeLandmarkLists import *
import torch


class MediaPipeFaceOccluder(object):

    def __init__(self) -> None:
        # self.left_eye = left_eye_eyebrow_landmark_indices()
        # self.right_eye = right_eye_eyebrow_landmark_indices()
        # self.mouth = mouth_landmark_indices()
        # self.face_oval = face_oval_landmark_indices()
        # self.face_all = all_face_landmark_indices()
        self.face_center = torch.LongTensor(face_center_landmark_indices())
        self.occlusion_regions_prob = {
            'contour': 0.7,
            'all': 0.1,
            'left_eye': 0.2,
            'right_eye': 0.2,
            'mouth': 0.3,
            'random': 0.15,
        }
        print(f"[Face Occluder] Init occluder with probability: {self.occlusion_regions_prob}")
    
    def occlude_lmk_batch(self, lmk_2d, lmk_mask, region, frame_id):
        """ Randomly mask 2d landmarks
        Args:
            lmk_2d: (n, V, 2) in [-1, 1]
            region: masking region,
            frame_id: frame ids to apply occlusion mask
            lmk_mask: (n, V)
        Returns:
            lmk_mask: (n, V), 0-masked, 1-non-masked
        """
        n, v = lmk_2d.shape[:2]
        
        if region == "all":
            lmk_mask[frame_id,:] = 0
        elif region == "left_eye": 
            left_eye_center = lmk_2d[frame_id, 27].unsqueeze(1) # (nc, 1, 2)
            dw, dh = 0.15 + 0.1 * torch.rand(2) # ~uniform(0.15, 0.25)
            dist_to_center = (lmk_2d[frame_id] - left_eye_center).abs() # (nc, V, 2)
            mask = (dist_to_center[...,0] < dw) & (dist_to_center[...,1] < dh)  # (nc, V)
            whole_mask = torch.zeros(n, v).bool()
            whole_mask[frame_id] = mask
            lmk_mask[whole_mask] = 0
        elif region == "right_eye": 
            right_eye_center = lmk_2d[frame_id, 257].unsqueeze(1) # (nc, 1, 2)
            dw, dh = 0.15 + 0.1 * torch.rand(2) # ~uniform(0.15, 0.25)
            dist_to_center = (lmk_2d[frame_id] - right_eye_center).abs() # (nc, V, 2)
            mask = (dist_to_center[...,0] < dw) & (dist_to_center[...,1] < dh)  # (nc, V)
            whole_mask = torch.zeros(n, v).bool()
            whole_mask[frame_id] = mask
            lmk_mask[whole_mask] = 0
        elif region == "mouth": 
            mouth_center = torch.mean(lmk_2d[frame_id, 13:15], dim=1).unsqueeze(1) # (nc, 1, 2)
            dw, dh = 0.15 + 0.1 * torch.rand(2) # ~uniform(0.15, 0.25)
            dist_to_center = (lmk_2d[frame_id] - mouth_center).abs() # (nc, V, 2)
            mask = (dist_to_center[...,0] < dw) & (dist_to_center[...,1] < dh)  # (nc, V)
            whole_mask = torch.zeros(n, v).bool()
            whole_mask[frame_id] = mask
            lmk_mask[whole_mask] = 0
        elif region == "random": 
            center_lmk_id = torch.randint(low=0, high=468, size=(1,))[0]
            random_center = lmk_2d[frame_id, center_lmk_id].unsqueeze(1)    # (nc, 1, 2)
            dw, dh = 0.1 + 0.5 * torch.rand(2)  # ~uniform(0.1, 0.6)
            dist_to_center = (lmk_2d[frame_id] - random_center).abs() # (nc, V, 2)
            mask = (dist_to_center[...,0] < dw) & (dist_to_center[...,1] < dh)  # (nc, V)
            whole_mask = torch.zeros(n, v).bool()
            whole_mask[frame_id] = mask
            lmk_mask[whole_mask] = 0
        elif region == "contour": 
            center_lmks = lmk_2d[frame_id][:,self.face_center] # (nc, n_center, 2)
            left = torch.min(center_lmks[...,0],dim=1).values   # (nc,)
            right = torch.max(center_lmks[...,0],dim=1).values
            top = torch.min(center_lmks[...,1],dim=1).values
            bottom = torch.max(center_lmks[...,1],dim=1).values

            bbox_center_x = (left + right) / 2  # (nc,)
            bbox_center_y = (top + bottom) / 2 # (nc,)
            bbox_center = torch.hstack([bbox_center_x[:,None], bbox_center_y[:,None]]).unsqueeze(1) # (nc, 1, 2)

            scale = np.random.rand() * 0.6 + 1 
            dw = (right - left) / 2 * scale # (nc)
            dh = (bottom - top) / 2 * scale # (nc)

            dist_to_bbox_center = (lmk_2d[frame_id] - bbox_center).abs() # (nc, V, 2)
            center_mask = (dist_to_bbox_center[...,0] < dw[:,None]) & (dist_to_bbox_center[...,1] < dh[:,None])  # (nc, V)
            whole_mask = torch.zeros(n, v).bool()
            whole_mask[frame_id] = ~center_mask
            lmk_mask[whole_mask] = 0.

        else: 
            raise ValueError(f"Invalid region {region}")
        return lmk_mask

    # def bounding_box(self, landmarks, region): 
    #     landmarks = landmarks[:, :2]
    #     if region == "all":
    #         left = np.min(landmarks[:, 0])
    #         right = np.max(landmarks[:, 0])
    #         top = np.min(landmarks[:, 1])
    #         bottom = np.max(landmarks[:, 1])
    #     elif region == "left_eye": 
    #         left = np.min(landmarks[self.left_eye, 0])
    #         right = np.max(landmarks[self.left_eye, 0])
    #         top = np.min(landmarks[self.left_eye, 1])
    #         bottom = np.max(landmarks[self.left_eye, 1])
    #     elif region == "right_eye": 
    #         left = np.min(landmarks[self.right_eye, 0])
    #         right = np.max(landmarks[self.right_eye, 0])
    #         top = np.min(landmarks[self.right_eye, 1])
    #         bottom = np.max(landmarks[self.right_eye, 1])
    #     elif region == "mouth": 
    #         left = np.min(landmarks[self.mouth, 0])
    #         right = np.max(landmarks[self.mouth, 0])
    #         top = np.min(landmarks[self.mouth, 1])
    #         bottom = np.max(landmarks[self.mouth, 1])
    #     else: 
    #         raise ValueError(f"Invalid region {region}")

    #     width = right - left
    #     height = bottom - top
    #     center_x = left + width / 2
    #     center_y = top + height / 2
        
    #     center = np.stack([center_x, center_y], axis=1).round().astype(np.int32)
    #     size = np.stack([width, height], axis=1).round().astype(np.int32)

    #     bb = np.array([left, right, top, bottom], dtype = np.int32)
    #     sizes = np.concatenate([center, size])
    #     return bb, sizes
    
    # def bounding_box_batch(self, landmarks, region): 
    #     assert landmarks.ndim == 3
    #     landmarks = landmarks[:, :, :2]
    #     if region == "all":
    #         left = np.min(landmarks[:,:, 0], axis=1)
    #         right = np.max(landmarks[:,:, 0], axis=1)
    #         top = np.min(landmarks[:,:, 1], axis=1)
    #         bottom = np.max(landmarks[:,:, 1], axis=1)
    #     elif region == "left_eye": 
    #         left = np.min(landmarks[:,self.left_eye, 0], axis=1)
    #         right = np.max(landmarks[:,self.left_eye, 0], axis=1)
    #         top = np.min(landmarks[:,self.left_eye, 1], axis=1)
    #         bottom = np.max(landmarks[:,self.left_eye, 1], axis=1)
    #     elif region == "right_eye": 
    #         left = np.min(landmarks[:,self.right_eye, 0], axis=1)
    #         right = np.max(landmarks[:,self.right_eye, 0], axis=1)
    #         top = np.min(landmarks[:,self.right_eye, 1], axis=1)
    #         bottom = np.max(landmarks[:,self.right_eye, 1], axis=1)
    #     elif region == "mouth": 
    #         left = np.min(landmarks[:,self.mouth, 0], axis=1)
    #         right = np.max(landmarks[:,self.mouth, 0], axis=1)
    #         top = np.min(landmarks[:,self.mouth, 1], axis=1)
    #         bottom = np.max(landmarks[:,self.mouth, 1], axis=1)
    #     else: 
    #         raise ValueError(f"Invalid region {region}")
        
    #     width = right - left
    #     height = bottom - top
    #     centers_x = left + width / 2
    #     centers_y = top + height / 2
    #     bb = np.stack([left, right, top, bottom], axis=1).round().astype(np.int32)
    #     sizes = np.stack([centers_x, centers_y, width, height], axis=1).round().astype(np.int32)
    #     return bb, sizes

    # def occlude(self, image, region, landmarks=None, bounding_box=None):
    #     assert landmarks is not None and bounding_box is not None, "Specify either landmarks or bounding_box"
    #     if landmarks is not None: 
    #         bounding_box = self.bounding_box(landmarks, region) 
        
    #     image[bounding_box[2]:bounding_box[3], bounding_box[0]:bounding_box[1], ...] = 0 
    #     return image

    # def occlude_batch(self, image, region, landmarks=None, bounding_box_batch=None
    #         , start_frame=None, end_frame=None, validity=None): 
    #     assert not(landmarks is not None and bounding_box_batch is not None), "Specify either landmarks or bounding_box"
    #     start_frame = start_frame or 0
    #     end_frame = end_frame or image.shape[0]
    #     assert end_frame <= image.shape[0]
    #     if landmarks is not None:
    #         bounding_box_batch, sizes_batch = self.bounding_box_batch(landmarks, region) 
    #     for i in range(start_frame, end_frame): 
    #         if validity is not None and not validity[i]: # if the bounding box is not valid, occlude nothing
    #             continue
    #         image[i, bounding_box_batch[i, 2]:bounding_box_batch[i, 3], bounding_box_batch[i, 0]:bounding_box_batch[i, 1], ...] = 0
        
    #     # # do the above without a for loop 
    #     # image[:, bounding_box_batch[:, 2]:bounding_box_batch[:, 3], bounding_box_batch[:, 0]:bounding_box_batch[:, 1], ...] = 0
    #     return image


def sizes_to_bb(sizes): 
    left = sizes[0] - sizes[2]
    right = sizes[0] + sizes[2]
    top = sizes[1] - sizes[3]
    bottom = sizes[1] + sizes[3]
    return np.array([left, right, top, bottom], dtype=np.int32)    


def sizes_to_bb_batch(sizes):
    left = sizes[:, 0] - sizes[:, 2]
    right = sizes[:, 0] + sizes[:, 2]
    top = sizes[:, 1] - sizes[:, 3]
    bottom = sizes[:, 1] + sizes[:, 3]
    return np.stack([left, right, top, bottom], axis=1)