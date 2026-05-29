from PIL import Image
import torch
import numpy as np
import os 
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a tensor."""
    return torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

def tensor_to_pil(tensor: torch.Tensor) -> list[Image.Image]:
    images = (tensor / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    pil_images = [Image.fromarray((image * 255).round().astype("uint8")) for image in images]
    return pil_images

def save_image(image: Image.Image, path: str) -> None:
    """Save a PIL image to a specified path."""
    # print(path, path.split('/')[0])
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    image.save(path)

def load_image(path: str) -> Image.Image:
    """Load an image from a specified path."""
    return Image.open(path).convert("RGB")

def float_to_uint8(image: torch.Tensor) -> torch.Tensor:
    """
    Convert a float tensor in [-1, 1] to uint8.
    """
    image = image.clamp(-1,1)
    image = (image / 2 + 0.5).clamp(0, 1)
    return (image * 255).clamp(0, 255).round().to(torch.uint8)

def convert_to_PIL(x):
    # Convert tensor to PIL image
    x = x.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
    x = (x * 255).astype(np.uint8)  # Scale to [0, 255]
    return Image.fromarray(x)

def min_max_normalize(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of tensors of size (N, C, H, W) to the range [0, 1] using min-max normalization.
    """
    min_val = image.amin(dim=[2, 3], keepdim=True)
    max_val = image.amax(dim=[2, 3], keepdim=True)
    normalized_tensor = (image - min_val) / (max_val - min_val + 1e-8)
    return normalized_tensor

class Image_metrics():
    def __init__(self, data_range: tuple = (0,255.0), n_type = "-1,1"):
        self.data_range = data_range
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range, reduction = "none")
        self.psnr_metric = PeakSignalNoiseRatio(data_range=data_range, reduction = "none", dim=(1,2,3))
        self.n_type = n_type
    
    def compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        img1 = self.normalize_to_uint8(img1)
        img2 = self.normalize_to_uint8(img2)
        return self.ssim_metric(img1, img2)
    
    def compute_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        img1 = self.normalize_to_uint8(img1)
        img2 = self.normalize_to_uint8(img2)
        return self.psnr_metric(img1, img2)
    
    def normalize_to_uint8(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor to the range [0, 255] and convert to uint8.
        """
        if image.dtype == torch.uint8:
            return image
        elif self.n_type == "minmax" and (image.min() < 0 or image.max() > 1):
            image = (min_max_normalize(image) * 255).clamp(0, 255).round().to(torch.uint8)
        elif self.n_type == "-1,1":
            image = float_to_uint8(image)
        return image

    def iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum((1,2,3))
        union = np.logical_or(mask1, mask2).sum((1,2,3))
        return intersection / (union + 1e-6)

    def dice(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum((1,2,3))
        return (2 * intersection) / (mask1.sum((1,2,3)) + mask2.sum((1,2,3)))


