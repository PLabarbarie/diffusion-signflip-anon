from .face_detector import get_face_detector
from .fr_models import FR_model
from utils.image_utils import tensor_to_pil
import torch

class FRPipeline(torch.nn.Module):
    def __init__(self, face_detector_name: str, face_recognizer_name: str, device: str = 'cuda', models_dir="models"):
        super(FRPipeline, self).__init__()
        self.face_recognizer = FR_model(model_name=face_recognizer_name, device=device, models_dir=models_dir)
        # For now only MTCNN is supported as face detector
        self.face_detector = get_face_detector(face_detector_name, input_size=self.face_recognizer.input_size,
                                               fr_model=face_recognizer_name, device=device)        
        self.device = device

    def forward(self, images, save_path=None):
        """Expect images to be a batch of images as tensors of shape (N, 3, H, W) in RGB format and float [-1, 1]"""
        # images = tensor_to_pil(images)
        # detected_face = self.face_detector(images)

        # if save_path is not None:
        #     for i, face in enumerate(detected_face):
        #         face = tensor_to_pil(face.unsqueeze(0))[0]
        #         face.save(f"{save_path}/detected_face_{i}.png")

        # detected_face = torch.stack(detected_face).to(self.device)
        detected_face = self.face_detector(images)
        fr_embeddings = self.face_recognizer(detected_face)

        return fr_embeddings
