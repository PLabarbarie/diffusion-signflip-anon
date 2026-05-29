import torch
import numpy as np


class utility_metrics():
    @staticmethod
    def iou_bboxes(boxA: list, boxB: list) -> float:
        """
        Compute the Intersection over Union (IoU) of two bounding boxes.
        Each box is represented as a list of four numbers: [x1, y1, x2, y2].
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

    @staticmethod
    def bbox_l2_distance(boxA: list, boxB: list, unique_face = True) -> float:
        """
        Compute the L2 distance between the centers of two bounding boxes.
        Each box is represented as a list of four numbers: [x1, y1, x2, y2].
        Return the sum of L2 distances for a batch of boxes.
        """
        if unique_face:
            centerA = [(boxA[:,0,0] + boxA[:,0,2]) / 2, (boxA[:,0,1] + boxA[:,0,3]) / 2]
            centerB = [(boxB[:,0,0] + boxB[:,0,2]) / 2, (boxB[:,0,1] + boxB[:,0,3]) / 2]     
            res = np.sqrt((centerA[0] - centerB[0]) ** 2 + (centerA[1] - centerB[1]) ** 2)  
        else:     
            centerA = [(boxA[:,:,0] + boxA[:,:,2]) / 2, (boxA[:,:,1] + boxA[:,:,3]) / 2]
            centerB = [(boxB[:,:,0] + boxB[:,:,2]) / 2, (boxB[:,:,1] + boxB[:,:,3]) / 2]
            res = np.sqrt((centerA[0] - centerB[0]) ** 2 + (centerA[1] - centerB[1]) ** 2).sum(1)
        return res
    
    @staticmethod
    def landmarks_l2_distance(landmarksA: np.ndarray, landmarksB: np.ndarray, unique_face = True) -> float:
        """
        Compute the average L2 distance between two sets of facial landmarks.
        Each set of landmarks is represented as a numpy array of shape (B, N, 2), where N is the number of landmarks.
        """
        if landmarksA.shape != landmarksB.shape:
            raise ValueError("Landmark arrays must have the same shape.")
        if unique_face:
            out = np.sqrt((landmarksA[:,0] - landmarksB[:,0])**2).sum(1)
        else:
            out = np.sqrt((landmarksA - landmarksB)**2).sum(2).sum(1)
        return out

