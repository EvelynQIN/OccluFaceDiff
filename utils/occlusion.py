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
            'all': 0.05,
            'eye': 0.01,
            'left': 0.01,
            'right': 0.01,
            'left_eye': 0.05,
            'right_eye': 0.05,
            'mouth': 0.5,
            'random': 0.1,
            'contour': 0.05
        }
        self.mask_all_prob = 0.1
        self.mask_frame_prob = 0.1
        self.image_size = 224
        print(f"[Face Occluder] Init occluder with probability: all - {self.mask_all_prob}; frame - {self.mask_frame_prob}")
        print(f"[Face Occluder] Init occluder with regional probability: {self.occlusion_regions_prob}")
    
    
    def occlude_img_batch(self, lmk_2d, img_mask, occlusion_type, frame_id):
        n, h, w = img_mask.shape

        if occlusion_type == 'all':
            img_mask[frame_id,:,:] = 0
        elif occlusion_type == 'left_eye':
            idx = left_eye_eyebrow_landmark_indices()
            scale_x, scale_y = torch.rand(2) * 3 + 0.9
            left, right, top, bottom = bbox_from_lmk(lmk_2d, idx, scale_x, scale_y)
            img_mask[frame_id,top:bottom, left:right] = 0
        elif occlusion_type == 'right_eye':
            idx = right_eye_eyebrow_landmark_indices()
            scale_x, scale_y = torch.rand(2) * 3 + 0.9
            left, right, top, bottom = bbox_from_lmk(lmk_2d, idx, scale_x, scale_y)
            img_mask[frame_id, top:bottom, left:right] = 0
        elif occlusion_type == 'mouth':
            idx = mouth_landmark_indices()
            scale_x, scale_y = torch.rand(2) * 3 + 0.9
            left, right, top, bottom = bbox_from_lmk(lmk_2d, idx, scale_x, scale_y)
            img_mask[frame_id,top:bottom, left:right] = 0
        elif occlusion_type == 'upper':
            eye_idx = left_eye_eyebrow_landmark_indices() + right_eye_eyebrow_landmark_indices()
            bottom = torch.max(lmk_2d[:,eye_idx, 1])
            img_mask[frame_id,:bottom,:] = 0
        elif occlusion_type == 'bottom':
            img_mask[frame_id, 112:,:] = 0
        elif occlusion_type == 'left':
            img_mask[frame_id,:,:112] = 0
        elif occlusion_type == 'right':
            img_mask[frame_id,:,112:] = 0
        elif occlusion_type == 'random':
            x, y, w, h = torch.randint(low=40, high=180,size = (4,))
            img_mask[frame_id, y:y+h, x:x+w] = 0
        else:
            raise ValueError("Occlusion region not supported!")
        return img_mask
    
    def get_lmk_mask_from_img_mask(self, img_mask, kpts):
        n, v = kpts.shape[:2]
        lmk_mask = torch.ones((n,v))
        for i in range(n):
            for j in range(v):
                x, y = kpts[i,j]
                if x<0 or x >=self.image_size or y<0 or y>=self.image_size or img_mask[i,y,x]==0:
                    lmk_mask[i,j] = 0
        return lmk_mask
    
    def get_lmk_occlusion_mask_from_img(self, lmk_2d, img_mask=None):
        n, v = lmk_2d.shape[:2]
        # occlusion all visual cues:
        p = np.random.rand()
        if p < self.mask_all_prob:
            lmk_mask = torch.zeros(n,v)
            return lmk_mask 
        elif p < self.mask_all_prob + self.mask_frame_prob:
            lmk_mask = torch.ones(n,v)
            frame_id = torch.randint(low=0, high=n, size=(n // 2,))
            lmk_mask[frame_id,:] = 0
            return lmk_mask
        
        # occlude random regions for consecutive frames
        kpts = (lmk_2d.clone() * 112 + 112).long()
        if img_mask is None:
            img_mask = torch.ones((n,self.image_size,self.image_size))

        sid = torch.randint(low=0, high=n-25, size=(1,))[0]
        occ_num_frames = torch.randint(low=25, high=n-sid+1, size=(1,))[0]
        frame_id = torch.arange(sid, sid+occ_num_frames)
        for occ_region, occ_prob in self.occlusion_regions_prob.items():
            prob = np.random.rand()
            if prob < occ_prob:
                img_mask = self.occlude_img_batch(kpts, img_mask, occ_region, frame_id)
                if occ_region == 'all':
                    break
        lmk_mask = self.get_lmk_mask_from_img_mask(img_mask, kpts)
        return lmk_mask
    
    def get_lmk_occlusion_mask(self, lmk_2d):
        n, v = lmk_2d.shape[:2]
        # occlusion all visual cues:
        p = np.random.rand()
        if p < self.mask_all_prob:
            lmk_mask = torch.zeros(n,v)
            return lmk_mask 
        elif p < self.mask_all_prob + self.mask_frame_prob:
            lmk_mask = torch.ones(n,v)
            frame_id = torch.randint(low=0, high=n, size=(n // 2,))
            lmk_mask[frame_id,:] = 0
            return lmk_mask
        
        # occlude random regions for consecutive frames
        sid = torch.randint(low=0, high=n-30, size=(1,))[0]
        occ_num_frames = torch.randint(low=30, high=n-sid+1, size=(1,))[0]
        frame_id = torch.arange(sid, sid+occ_num_frames)
        lmk_mask = torch.ones(n, v)
        for occ_region, occ_prob in self.occlusion_regions_prob.items():
            prob = np.random.rand()
            if prob < occ_prob:
                lmk_mask = self.occlude_lmk_batch(lmk_2d, lmk_mask, occ_region, frame_id)
                if occ_region in ['all', 'left', 'right', 'eye']:
                    break
        return lmk_mask

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
        elif region == 'eye':
            left_eye_center = lmk_2d[frame_id, 27].unsqueeze(1) # (nc, 1, 2)
            dw, dh = 0.25 + 0.5 * torch.rand(2) # ~uniform(0.15, 0.25)
            dist_to_center = (lmk_2d[frame_id] - left_eye_center).abs() # (nc, V, 2)
            mask = (dist_to_center[...,0] < dw) & (dist_to_center[...,1] < dh)  # (nc, V)
            whole_mask = torch.zeros(n, v).bool()
            whole_mask[frame_id] = mask
            lmk_mask[whole_mask] = 0

            right_eye_center = lmk_2d[frame_id, 257].unsqueeze(1) # (nc, 1, 2)
            dw, dh = 0.25 + 0.5 * torch.rand(2) # ~uniform(0.15, 0.25)
            dist_to_center = (lmk_2d[frame_id] - right_eye_center).abs() # (nc, V, 2)
            mask = (dist_to_center[...,0] < dw) & (dist_to_center[...,1] < dh)  # (nc, V)
            whole_mask = torch.zeros(n, v).bool()
            whole_mask[frame_id] = mask
            lmk_mask[whole_mask] = 0

        elif region == 'left':
            whole_mask = torch.zeros(n, v).bool()
            mask = lmk_2d[frame_id, :, 0] < 0
            whole_mask[frame_id] = mask
            lmk_mask[whole_mask] = 0

        elif region == 'right':
            whole_mask = torch.zeros(n, v).bool()
            mask = lmk_2d[frame_id, :, 0] > 0
            whole_mask[frame_id] = mask
            lmk_mask[whole_mask] = 0

        elif region == "left_eye": 
            left_eye_center = lmk_2d[frame_id, 27].unsqueeze(1) # (nc, 1, 2)
            dw, dh = 0.25 + 0.5 * torch.rand(2) # ~uniform(0.15, 0.25)
            dist_to_center = (lmk_2d[frame_id] - left_eye_center).abs() # (nc, V, 2)
            mask = (dist_to_center[...,0] < dw) & (dist_to_center[...,1] < dh)  # (nc, V)
            whole_mask = torch.zeros(n, v).bool()
            whole_mask[frame_id] = mask
            lmk_mask[whole_mask] = 0
        elif region == "right_eye": 
            right_eye_center = lmk_2d[frame_id, 257].unsqueeze(1) # (nc, 1, 2)
            dw, dh = 0.25 + 0.5 * torch.rand(2) # ~uniform(0.15, 0.25)
            dist_to_center = (lmk_2d[frame_id] - right_eye_center).abs() # (nc, V, 2)
            mask = (dist_to_center[...,0] < dw) & (dist_to_center[...,1] < dh)  # (nc, V)
            whole_mask = torch.zeros(n, v).bool()
            whole_mask[frame_id] = mask
            lmk_mask[whole_mask] = 0
        elif region == "mouth": 
            mouth_center = torch.mean(lmk_2d[frame_id, 13:15], dim=1).unsqueeze(1) # (nc, 1, 2)
            dw, dh = 0.25 + 0.5 * torch.rand(2) # ~uniform(0.2, 0.5)
            dist_to_center = (lmk_2d[frame_id] - mouth_center).abs() # (nc, V, 2)
            mask = (dist_to_center[...,0] < dw) & (dist_to_center[...,1] < dh)  # (nc, V)
            whole_mask = torch.zeros(n, v).bool()
            whole_mask[frame_id] = mask
            lmk_mask[whole_mask] = 0
        elif region == "random": 
            center_lmk_id = torch.randint(low=0, high=468, size=(1,))[0]
            random_center = lmk_2d[frame_id, center_lmk_id].unsqueeze(1)    # (nc, 1, 2)
            dw, dh = 0.25 + 0.5 * torch.rand(2)  # ~uniform(0.1, 0.6)
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

def test_mediapipe_lmk_occlussion(lmk_mask, occlusion_type):
    n, v = lmk_mask.shape
    if occlusion_type == 'non_occ':
        return lmk_mask 
    elif occlusion_type == 'left_eye':
        idx = left_eye_eyebrow_landmark_indices()
        lmk_mask[:,idx] = 0
    elif occlusion_type == 'right_eye':
        idx = right_eye_eyebrow_landmark_indices()
        lmk_mask[:,idx] = 0
    elif occlusion_type == 'mouth':
        idx = mouth_landmark_indices()
        lmk_mask[:,idx] = 0
    return lmk_mask

def bbox_from_lmk(lmk_2d, idx, scale_x=None, scale_y=None):
    left = torch.min(lmk_2d[:,idx, 0]) 
    right = torch.max(lmk_2d[:,idx, 0])
    top = torch.min(lmk_2d[:,idx, 1])
    bottom = torch.max(lmk_2d[:,idx, 1])
    
    if scale_x is None:
        scale_x = np.random.rand() * 0.6 + 0.9
    if scale_y is None:
        scale_y = np.random.rand() * 0.6 + 0.9
    dx = ((right - left ) / 2 * scale_x).long()
    dy = ((bottom - top ) / 2 * scale_y).long()
    centers_x = (left + right) // 2
    centers_y = (top + bottom) // 2

    left = torch.clamp(centers_x - dx, min=0, max=224)
    right = torch.clamp(centers_x + dx, min=0, max=224)
    top = torch.clamp(centers_y - dy, min=0, max=224)
    bottom = torch.clamp(centers_y + dy, min=0, max=224)

    return [left, right, top, bottom]

def get_test_img_occlusion_mask(img_mask, lmk_2d, occlusion_type):
    n, h, w = img_mask.shape
    
    if occlusion_type == 'non_occ':
        return img_mask
    
    if n < 40:
        sid = 0
    sid = torch.randint(low=0, high=n-40, size=(1,))[0] if n > 40 else 0
    occ_num_frames = torch.randint(low=min(40, n-sid), high=n-sid+1, size=(1,))[0]
    frame_id = torch.arange(sid, sid+occ_num_frames)

    if occlusion_type == 'all':
        img_mask[:,:,:] = 0
    elif occlusion_type == 'missing_frames':
        img_mask[frame_id,:,:] = 0
    elif occlusion_type == 'left_eye':
        idx = left_eye_eyebrow_landmark_indices()
        left, right, top, bottom = bbox_from_lmk(lmk_2d, idx)
        img_mask[frame_id,top:bottom, left:right] = 0
    elif occlusion_type == 'right_eye':
        idx = right_eye_eyebrow_landmark_indices()
        left, right, top, bottom = bbox_from_lmk(lmk_2d, idx)
        img_mask[frame_id,top:bottom, left:right] = 0
    elif occlusion_type == 'mouth':
        idx = mouth_landmark_indices()
        left, right, top, bottom = bbox_from_lmk(lmk_2d, idx)
        img_mask[frame_id,top:bottom, left:right] = 0
    elif occlusion_type == 'upper':
        eye_idx = np.concatenate([left_eye_eyebrow_landmark_indices(),right_eye_eyebrow_landmark_indices()])
        eye_idx = torch.from_numpy(eye_idx).long()
        bottom = torch.max(lmk_2d[:,eye_idx, 1])
        img_mask[frame_id,:bottom,:] = 0
    elif occlusion_type == 'bottom':
        img_mask[frame_id,112:,:] = 0
    elif occlusion_type == 'left':
        img_mask[frame_id,:,:112] = 0
    elif occlusion_type == 'right':
        img_mask[frame_id,:,112:] = 0
    elif occlusion_type == 'asym':
        if np.random.rand() < 0.5:
            img_mask[frame_id,:,:112] = 0
        else:
            img_mask[frame_id,:,112:] = 0
    elif occlusion_type == 'downsample_frames':
        img_mask[::3,:,:] = 0
    elif occlusion_type == 'random':
        x, y = torch.randint(low=20, high = 170, size=(2,))
        dx, dy = torch.randint(low=100, high=220, size=(2,))
        img_mask[frame_id, x:x+dx, y:y+dy] = 0
    return img_mask