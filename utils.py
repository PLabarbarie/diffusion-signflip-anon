import torch
import torch.nn.functional as F
from PIL import Image

# ------------------
# MSE
# ------------------
def mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error (MSE).
    img1, img2: tensors in [0,1], shape (N,C,H,W)
    """
    return F.mse_loss(img1, img2, reduction="mean")


# ------------------
# MAE (L1 Loss)
# ------------------
def mae(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error (MAE).
    """
    return F.l1_loss(img1, img2, reduction="mean")


# ------------------
# PSNR
# ------------------
def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio (PSNR).
    """
    mse_val = ((img1 - img2)**2).mean((1,2,3))
    if mse_val == 0:
        return torch.tensor(float("inf"))
    return 20 * torch.log10(255 / torch.sqrt(mse_val))

def torch_to_pil(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    pil_images = [Image.fromarray((image * 255).round().astype("uint8")) for image in images]
    return pil_images

def float_to_uint8(image: torch.Tensor) -> torch.Tensor:
    """
    Convert a float tensor in [-1, 1] to uint8.
    """
    image = (image / 2 + 0.5).clamp(0, 1)
    return (image * 255).clamp(0, 255).round().to(torch.uint8)
