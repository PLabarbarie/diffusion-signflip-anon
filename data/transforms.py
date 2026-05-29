from torchvision import transforms as tfms
from torchvision.transforms import v2 as tfms_v2
import torch

def get_transforms_fr(model):
    # transforms.ToTensor() divides by 255 when the input is a PIL image
    if 'facenet' in model:
        return tfms.Compose([
            tfms.Resize((160, 160)),
            tfms.ToTensor(),
            tfms.Normalize(mean=[0.5], std=[0.5])  # normalize grayscale or RGB
        ])
    elif model in ['arcface_r50', 'arcface_r34', "adaface_vit"]:
        return tfms.Compose([
            tfms.Resize((112, 112)),  # or 64x64
            tfms.ToTensor(),
            tfms.Normalize(mean=[0.5], std=[0.5])  # normalize grayscale or RGB
        ])
    else:
        raise ValueError(f"Unknown model: {model}. Please specify a valid model name.")

def get_base_transforms(DATASET):
    if DATASET == "LFW":
        return tfms.Compose([
            tfms.Resize((256, 256)),
            tfms.CenterCrop((200,200)),
            tfms.Resize((256, 256)),
            tfms.ToTensor(),
            tfms.Normalize(mean=[0.5], std=[0.5])  # normalize grayscale or RGB
        ])
    else :        
        return tfms.Compose([
            tfms.Resize((256, 256)),
            tfms.ToTensor(),
            tfms.Normalize(mean=[0.5], std=[0.5])  # normalize grayscale or RGB
        ])

class JPEG_compression():
    def __init__(self, qf):
        self.tfms_compression = {q : tfms_v2.JPEG(quality=q) for q in qf}
    def compression(self, img, q):
        #expected a [-1,1] tensors
        img = img.clamp(-1,1)
        img = (img / 2 + 0.5).clamp(0, 1)
        img = (img * 255).clamp(0, 255).round().to(torch.uint8)
        c_img = self.tfms_compression[q](img.cpu())
        c_img = c_img.cuda()
        c_img = ((c_img/255.)-0.5)/0.5
        return c_img
