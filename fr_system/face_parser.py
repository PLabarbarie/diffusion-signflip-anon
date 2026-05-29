from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict
import torch


class FaceParser(ABC):
    """Abstract base class for face parsing/segmentation."""
    
    @abstractmethod
    def parse_face(self, image: np.ndarray, bbox: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Parse facial regions in an image.
        
        Args:
            image: Input image as numpy array (RGB)
            bbox: Optional bounding box [x1, y1, x2, y2] to crop face region
            
        Returns:
            Segmentation mask with labeled regions
        """
        pass
    
    @abstractmethod
    def get_label_map(self) -> Dict[int, str]:
        """
        Get mapping of label indices to region names.
        
        Returns:
            Dictionary mapping label index to region name
        """
        pass


class BiSeNetParser(FaceParser):
    """BiSeNet-based face parser."""
    
    # Face parsing labels (based on CelebAMask-HQ)
    LABEL_MAP = {
        0: 'background',
        1: 'skin',
        2: 'nose',
        3: 'eye_g',  # eyeglasses
        4: 'l_eye',
        5: 'r_eye',
        6: 'l_brow',
        7: 'r_brow',
        8: 'l_ear',
        9: 'r_ear',
        10: 'mouth',
        11: 'u_lip',
        12: 'l_lip',
        13: 'hair',
        14: 'hat',
        15: 'ear_r',  # earring
        16: 'neck_l',
        17: 'neck',
        18: 'cloth'
    }
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        from .bisenet import BiSeNet
        self.device = device
        self.model = BiSeNet(n_classes=19)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.to(device)
        self.model.eval()
    
    def parse_face(self, image: np.ndarray, bbox: Optional[np.ndarray] = None) -> np.ndarray:
        """Parse face using BiSeNet."""
        import cv2
        import torch.nn.functional as F
        
        # Crop face if bbox provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            face_img = image[y1:y2, x1:x2]
        else:
            face_img = image
        
        # Preprocess
        face_resized = cv2.resize(face_img, (512, 512))
        face_tensor = torch.from_numpy(face_resized).float()
        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
        face_tensor = (face_tensor - 127.5) / 127.5
        face_tensor = face_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            out = self.model(face_tensor)[0]
            parsing = out.squeeze(0).argmax(0).cpu().numpy()
        
        # Resize back to original face size
        if bbox is not None:
            parsing = cv2.resize(parsing.astype(np.uint8), (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        else:
            parsing = cv2.resize(parsing.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        return parsing
    
    def get_label_map(self) -> Dict[int, str]:
        """Get label mapping."""
        return self.LABEL_MAP.copy()


class RTMLandmarkParser(FaceParser):
    """RTMPose-based face landmark parser (alternative approach)."""
    
    LABEL_MAP = {
        0: 'background',
        1: 'face_region'
    }
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        # This is a placeholder for RTMLib integration
        # You would integrate rtmlib here for landmark-based parsing
        self.device = device
    
    def parse_face(self, image: np.ndarray, bbox: Optional[np.ndarray] = None) -> np.ndarray:
        """Parse face using landmarks (placeholder implementation)."""
        # Placeholder: returns simple binary mask
        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            return mask
        return np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    def get_label_map(self) -> Dict[int, str]:
        """Get label mapping."""
        return self.LABEL_MAP.copy()


def get_face_parser(parser_type: str = 'bisenet', **kwargs) -> FaceParser:
    """
    Factory function to get a face parser.
    
    Args:
        parser_type: Type of parser ('bisenet' or 'rtmlib')
        **kwargs: Additional arguments for the parser
        
    Returns:
        FaceParser instance
    """
    parser_type = parser_type.lower()
    
    if parser_type == 'bisenet':
        return BiSeNetParser(**kwargs)
    elif parser_type in ['rtmlib', 'rtm']:
        return RTMLandmarkParser(**kwargs)
    else:
        raise ValueError(f"Unknown parser type: {parser_type}. Choose 'bisenet' or 'rtmlib'")