from torchvision import transforms as tfms

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