from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional
import torch
from utils.image_utils import tensor_to_pil
import torch.nn.functional as F
from typing import Tuple
import os 

class FaceDetector(ABC):
    """Abstract base class for face detectors."""

    def __init__(self, name: str, device: torch.device):
        """A detector base class.

        :param name: detector title
        :param cfg: DETECTOR config for detector setting
        :param input_tensor_size: size of the tensor to input the detector. Square (x*x).
        :param device: torch device (cuda or cpu)
        """
        self.name = name
        self.device = device
        self.detector = None

    @abstractmethod
    def __call__(self, image: torch.Tensor, *args, **kwds) -> torch.Tensor:
        """
        Return a cropped face image tensor given an input image tensor.
        1. Detect faces in the input image.
        2. Crop the face region and convert to tensor.
        Args:

        """
        pass
    
    @abstractmethod
    def detect_faces(self, image: np.ndarray, conf_threshold: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (RGB)
            conf_threshold: Confidence threshold for detection
            
        Returns:
            boxes: Array of bounding boxes [x1, y1, x2, y2]
            scores: Array of confidence scores
            landmarks: Array of facial landmarks (if available)
        """
        pass

    @abstractmethod
    def extract_landmarks(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract facial landmarks from detected face boxes.
        
        Args:
            image: Input image as a torch Tensor
        Returns:
            landmarks: Extracted facial landmarks as a torch Tensor
        """
        pass

    @abstractmethod
    def get_hints(self, image: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Generate hint maps for controlnet based on detected landmarks.
        
        Args:
            image: Input image as a torch Tensor
        Returns:
            hints: Generated hint maps as a torch Tensor
        """
        pass

    def eval(self):
        """
        This is for model eval setting: fix the model and boost computing.
        """
        assert self.detector
        self.detector.eval()

    def to(self, device: torch.device):
        """
        Move the detector model to the specified device.
        
        Args:
            device: Target torch device (cuda or cpu)
        """
        assert self.detector
        self.device = device
        self.detector.to(device)




class MTCNNDetector(FaceDetector):
    """MTCNN-based face detector."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu', input_size: int = 256, fr_model = None):
        from .mtcnn import MTCNN
        self.device = device
        mtcnn = MTCNN(
            image_size=input_size,
            margin=14,
            device=device,
            keep_all=False, 
            thresholds=[0.6, 0.7, 0.7],
            select_largest=False, 
            selection_method='center_weighted_size',
            fr_model=fr_model
        )
        self.detector = mtcnn

    def extract_landmarks(self, image: torch.Tensor) -> torch.Tensor:
        landmarks = self.detector.extract_landmarks(image)
        return landmarks
    
    def get_hints(self, images: torch.Tensor, window_size: int = 1) -> torch.Tensor:
        hint = torch.zeros_like(images)
        landmarks = self.detector.extract_landmarks(images)
        for j in range(images.shape[0]):
            for k in range(landmarks[j].shape[0]):
                x, y = int(landmarks[j][k][0]),  int(landmarks[j][k][1])
                hint[j, :, y-window_size:y+window_size+1, x-window_size:x+window_size+1] = 1.0  # Mark landmark on a blank image
        return hint
    
    def __call__(self, image: torch.Tensor, *args, **kwds) -> torch.Tensor:
        image = tensor_to_pil(image)
        detected_faces = self.detector(image)
        detected_face = torch.stack(detected_faces).to(self.device)
        return detected_face
    
    
    def detect_faces(self, image: torch.Tensor): #-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect faces using MTCNN."""
        image = tensor_to_pil(image)
        boxes, probs, landmarks = self.detector.detect_torch(image, landmarks=True)
        return boxes, probs, landmarks


class RetinaFaceDetector(FaceDetector):
    """RetinaFace-based face detector."""
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu', confidence_threshold: float = 0.5,
                 network: str = 'resnet50', image_size: list[int] = [256, 256],  crop_size: int = 256, margin: int = 14):
        from .retina import retinafacedetector
        self.device = device
        model_path = os.path.dirname(__file__)
        if network == 'mobilenet':
            model_path = os.path.join(model_path,'weights', 'retinaface', 'mobilenet0.25_Final.pth')
        elif network == 'resnet50':
            model_path = os.path.join(model_path,'weights', 'retinaface', 'Resnet50_Final.pth')
        else:
            raise ValueError(f"Unsupported network type: {network}. Choose 'mobilenet' or 'resnet50'.")
        self.detector = retinafacedetector(model_path=model_path,
                                           confidence_threshold=confidence_threshold, network=network, image_size=image_size, device=device)
        self.crop_size = crop_size
        self.margin = margin
    
    def detect_faces(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect faces using RetinaFace."""
        return self.detector.forward(image)
    
    def extract_landmarks(self, image: torch.Tensor) -> torch.Tensor:
        _, landmarks, _ = self.detector.forward(image)
        return landmarks
    
    def get_hints(self, images: torch.Tensor, window_size: int = 1) -> torch.Tensor:
        hint = torch.zeros_like(images)
        landmarks = self.extract_landmarks(images)
        landmarks = landmarks[:, 0, :, :]  # Get the landmark coordinates
        for j in range(images.shape[0]):
            for k in range(landmarks[j].shape[0]):
                x, y = int(landmarks[j][k][0]),  int(landmarks[j][k][1])
                hint[j, :, y-window_size:y+window_size+1, x-window_size:x+window_size+1] = 1.0  # Mark landmark on a blank image
        return hint
    
    # First implementation
    # def __call__(self, image: torch.Tensor, *args, **kwds) -> torch.Tensor:
    #     dets, _, _ = self.detector.forward(image)
    #     crop_resized = torch.zeros_like(image)
    #     for i in range(dets.shape[0]):
    #         bbox = dets[i, 0, :]
    #         x1, y1, x2, y2 = map(int, bbox)
    #         y1 = int(max(0, y1 - self.margin/2))
    #         y2 = int(min(image.shape[2], y2 + self.margin/2))
    #         x1 = int(max(0, x1 - self.margin/2))
    #         x2 = int(min(image.shape[3], x2 + self.margin/2))
    #         crop = image[i, :, y1:y2, x1:x2]
    #     crop_resized[i] = F.interpolate(
    #         crop.unsqueeze(0),          # add batch dim
    #         size=(self.crop_size, self.crop_size),
    #         mode='bilinear',
    #         align_corners=False
    #     ).squeeze(0)

    #     return crop_resized

        
    def __call__(self, image: torch.Tensor, *args, **kwds) -> torch.Tensor:
        dets, _, _ = self.detector.forward(image)

        B, C, H, W = image.shape
        crop_resized = torch.zeros((B, C, self.crop_size, self.crop_size), device=image.device)

        for i in range(dets.shape[0]):
            bbox = dets[i, 0, :]
            x1, y1, x2, y2 = map(int, bbox)

            # Apply margin
            y1 = int(max(0, y1 - self.margin / 2))
            y2 = int(min(H, y2 + self.margin / 2))
            x1 = int(max(0, x1 - self.margin / 2))
            x2 = int(min(W, x2 + self.margin / 2))

            crop = image[i, :, y1:y2, x1:x2]

            h = y2 - y1
            w = x2 - x1

            if h == 0 or w == 0:
                continue

            # Scale while preserving aspect ratio
            scale = self.crop_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)

            resized = F.interpolate(
                crop.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # Create square canvas
            canvas = torch.zeros((C, self.crop_size, self.crop_size), device=image.device)

            # Center the resized crop
            y_offset = (self.crop_size - new_h) // 2
            x_offset = (self.crop_size - new_w) // 2

            canvas[:, y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

            crop_resized[i] = canvas

        return crop_resized
        

class RTMFaceDetector(FaceDetector):
    """RTM-based face detector."""
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 crop_size: int = 256, margin: int = 14, n_landmarks: int = 26, **kwargs):
        from .rtmlib.rtmlib import Wholebody
        self.device = device
        backend = 'onnxruntime'  # opencv, onnxruntime, openvino
        self.detector = Wholebody(pose='fr_system/weights/rtmw-dw-x-l-384x288/end2end.onnx',  # download link or local path
                    det_input_size = (640,640),
                    det = 'fr_system/weights/yolo_m_8/end2end.onnx',
                     backend=backend, device=device)
        self.crop_size = crop_size
        self.margin = margin
        self.n_landmarks = n_landmarks 
        if self.n_landmarks == 26:
            self.landmarks_points = np.array([0,1,2,3,4,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,77,71,74,80])
        elif self.n_landmarks == 5:
            self.landmarks_points = np.array([0,1,2,71,77])
        elif self.n_landmarks == 7 :
            self.landmarks_points = np.array([0,1,2,3,4,71,77])
        else:
            raise ValueError("n_landmarks must be 5 or 26")
            
    def detect_faces(self, image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect faces using RTM."""
        image = image.clamp(-1, 1).cpu().permute(0,2,3,1).numpy()  # Convert to numpy HWC
        image = (image + 1) * 127.5  # Convert back to [0, 255]
        image = image.astype(np.uint8)
        bboxes, landmarks, scores = [], [], []
        for i in range(image.shape[0]):
            box, landmark, score = self.detector(image[i])
            if len(box) != 0:
                largest_bbox_ind = ((box[:,2]-box[:,0])*(box[:,3]-box[:,1])).argmax()
            else:
                largest_bbox_ind = 0
            ind = largest_bbox_ind
            box = box[ind,:]
            landmark = landmark[ind,self.landmarks_points, :]
            score = score[ind,self.landmarks_points]
            bboxes.append(box)
            landmarks.append(landmark)
            scores.append(score)
        bboxes = np.array(bboxes) #bboxes can be of variable length
        landmarks = np.array(landmarks)
        scores = np.array(scores)
        return bboxes, landmarks, scores

    def extract_landmarks(self, image: torch.Tensor) -> torch.Tensor:
        image = image.clamp(-1, 1).cpu().permute(0,2,3,1).numpy()  # Convert to numpy HWC
        image = (image + 1) * 127.5  # Convert back to [0, 255]
        image = image.astype(np.uint8)
        landmarks = []
        for i in range(image.shape[0]):
            box, landmark, _ = self.detector(image[i])
            if len(box) != 0:
                largest_bbox_ind = ((box[:,2]-box[:,0])*(box[:,3]-box[:,1])).argmax()
            else:
                largest_bbox_ind = 0
            ind = largest_bbox_ind
            landmark = landmark[ind,self.landmarks_points, :]
            landmarks.append(torch.tensor(landmark).to(self.device))
        landmarks = torch.stack(landmarks)
        return landmarks
    
    def get_hints(self, images: torch.Tensor, window_size: int = 1) -> torch.Tensor:
        hint = torch.zeros_like(images)
        landmarks = self.extract_landmarks(images)
        for j in range(images.shape[0]):
            for k in range(landmarks[j].shape[0]):
                x, y = int(landmarks[j][k][0]),  int(landmarks[j][k][1])
                hint[j, :, y-window_size:y+window_size+1, x-window_size:x+window_size+1] = 1.0  # Mark landmark on a blank image
        return hint
    
    def __call__(self, image: torch.Tensor, *args, **kwds) -> torch.Tensor:
        dets, _, _ = self.detect_faces(image)
        crop_resized = torch.zeros_like(image)
        for i in range(len(image)):
            bbox = dets[i, :]
            x1, y1, x2, y2 = map(int, bbox)
            y1 = int(max(0, y1 - self.margin/2))
            y2 = int(min(image.shape[2], y2 + self.margin/2))
            x1 = int(max(0, x1 - self.margin/2))
            x2 = int(min(image.shape[3], x2 + self.margin/2))
            crop = image[i, :, y1:y2, x1:x2]
        crop_resized[i] = F.interpolate(
            crop.unsqueeze(0),          # add batch dim
            size=(self.crop_size, self.crop_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        return crop_resized
        

def get_face_detector(detector_name: str = 'mtcnn', **kwargs) -> FaceDetector:
    """
    Factory function to get a face detector.
    
    Args:
        detector_name: Type of detector ('mtcnn' or 'retinaface')
        **kwargs: Additional arguments for the detector
        
    Returns:
        FaceDetector instance
    """
    detector_name = detector_name.lower()
    
    if detector_name == 'mtcnn':
        return MTCNNDetector(**kwargs)
    elif detector_name in ['retinaface', 'retina']:
        return RetinaFaceDetector(**kwargs)
    elif detector_name   == 'rtm':
        return RTMFaceDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_name}. Choose 'mtcnn' or 'retinaface' or 'rtm'")