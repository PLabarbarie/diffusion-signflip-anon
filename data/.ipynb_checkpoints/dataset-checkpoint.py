import os
import random
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms as tfms
import platform
from torchvision.datasets import ImageFolder, VisionDataset
import csv
from collections import namedtuple
from typing import Any, Callable, Optional, Union
import torch 
from pathlib import Path
import numpy as np

CSV = namedtuple("CSV", ["header", "index", "data"])

class LFW_match_or_missmatch_pairs_dataset(Dataset):

    def __init__(self, root, img_dir, match=True, shuffle=True, transform=None, train=False):

        self.match = match # True for match pairs, False for missmatch pairs
        self.train = train
        # a decommenter pour utiliser les csv d'origine
        # if self.match :
        #     df = os.path.join(root, 'matchpairsDev')
        # else:
        #     df = os.path.join(root, 'mismatchpairsDev')
        # if self.train:
        #     df = df + 'Train.csv'
        # else:
        #     df = df + 'Test.csv'

        df = os.path.join(root, 'matchpairs.csv')

            
        df = pd.read_csv(df, dtype=str)
        self.lines = df.values.tolist()
        if len(self.lines) == 0:
            raise ValueError("The dataset is empty. Please check the CSV file.")
        
        img_dir = os.path.join(root, img_dir)
        self.img_dir = img_dir

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples  = len(self.lines)
        self.transform = transform


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        line = self.lines[index]

        if self.match:
            img1_path = os.path.join(self.img_dir, line[0], line[0] + "_" + "0"*(4-len(line[1])) + f"{line[1]}.jpg")
            img2_path = os.path.join(self.img_dir, line[0], line[0] + "_" + "0"*(4-len(line[2])) +  f"{line[2]}.jpg")
        else:
            img1_path = os.path.join(self.img_dir, line[0], line[0] + "_" + "0"*(4-len(line[1])) + f"{line[1]}.jpg")
            img2_path = os.path.join(self.img_dir, line[2], line[2] + "_" + "0"*(4-len(line[3])) + f"{line[3]}.jpg")

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)


        name1 = line[0]
        name2 = line[0] if self.match else line[2]

        return img1, img2, name1, name2
    

class FFHQ(Dataset):

    def __init__(self, root, transform=None, controlnet=False, latent_dir=None):
        self.root = root
        self.img_dir = os.path.join(self.root, "images1024x1024")
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.nSamples = len(self.image_files)
        self.controlnet = controlnet
        self.to_tensor = tfms.ToTensor()
        self.latent_dir = latent_dir

        if self.controlnet:
            self.landmark_dir = os.path.join(self.latent_dir, "segmentation_masks")
            self.latents = torch.load(os.path.join(self.latent_dir, "ffhq_latents.pt"))

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img_path = self.image_files[index]

        if self.controlnet:
            name = torch.tensor([int(img_path.split('.')[0])]).long()
            latents = self.latents[name]
            landmarks = Image.open(os.path.join(self.landmark_dir, img_path)).convert('RGB')
            landmarks = self.to_tensor(landmarks)
            return latents, landmarks, img_path
        else : 
            img = Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)


            return img, img_path
        
        # img_path = torch.Tensor([int(img_path.split('.')[0])])  # Extract filename without extension
        return img, img_path
    
class ImageFolderNoLabels(ImageFolder):
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)  # call parent, ignore label
        return image
    

class CelebA_HQ(Dataset):

    def __init__(self, root, transform=None, clip_image_processor=None):
        self.root = root
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.root) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.nSamples = len(self.image_files)
        self.clip_image_processor = clip_image_processor

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img_path = self.image_files[index]
        raw_image = Image.open(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(raw_image.convert("RGB"))
        if self.clip_image_processor is not None:
            clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values

        # img_path = os.path.join(self.root, img_path)
        # img_path = torch.Tensor([int(img_path.split('.')[0])])  # Extract filename without extension


        if self.clip_image_processor is not None:
            return {"img" : img,
                    "img_path": img_path,
                    "clip_image": clip_image}
        else :
            return img, img_path
    



class CelebA(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (Tensor shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (Tensor shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (Tensor shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

            .. warning::

                To download the dataset `gdown <https://github.com/wkentaro/gdown>`_ is required.
    """
    base_folder = "celeba"
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "all",
        multiple_images: bool = True,
        target_type: Union[list[str], str] = "identity",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")


        split_map = {
                    "train": 0,
                    "valid": 1,
                    "test": 2,
                    "valid_test" : (1, 2),
                    "all": None,
                }
        split_lower = split.lower() if isinstance(split, str) else split
        if split_lower not in split_map:
            raise ValueError(f"split must be one of {tuple(split_map.keys())}, got '{split}'")
        split_ = split_map[split_lower]

        splits = self._load_csv("new_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        id = identity.data[:, 0]
        mask = torch.zeros_like(id, dtype=torch.bool)

        self.multiple_images = multiple_images

        if split_ == None:
            mask = torch.ones_like(id, dtype=torch.bool)
        else:
            for i in range(len(splits.data)):
                if isinstance(split_, tuple) and splits.data[i] in split_:
                    mask[i] = 1
                elif isinstance(split_, int) and splits.data[i] == split_:
                    mask[i] = 1
                else:
                    mask[i] = 0

        
        if self.multiple_images:
            mask_multiple = torch.zeros_like(id, dtype=torch.bool)
            _, id_counts = torch.unique(id, return_counts=True)
            ids_with_multiple_images = torch.where(id_counts > 1)[0] + 1
            for i in ids_with_multiple_images:
                mask_multiple = mask_multiple | (id == i)
        else :
            mask_multiple = torch.ones_like(id, dtype=torch.bool)

        mask = mask & mask_multiple

        self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]  # type: ignore[arg-type]
        self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header


    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba_resized2", self.filename[index].split(".")[0] + "_resized.png"))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target


    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)





















def load_dataset(name, transform=tfms.ToTensor(), clip_image_processor= None, split="all", multiple_images=True, controlnet=False):
    whereIam = platform.node()
    print(whereIam)
    if whereIam == "MSI":
        root = "C:\\Users\\Pol\\Documents\\code"
    else:
        root = "/lustre/fswork/projects/rech/irm/uzq68by/data"

    if name == "FFHQ":
        root = os.path.join(root, "FlickrFace")
        latent_dir = root
        if whereIam == "MSI":
            pass
        else : 
            root = '/lustre/fsmisc/dataset'
            root = os.path.join(root, "FlickrFace")
        return FFHQ(root=root, transform=transform, controlnet=controlnet, latent_dir=latent_dir)
    elif name == "LFW":
        root = os.path.join(root, "LFW")
        # if whereIam == "MSI":
        #     root = root + "\\LFW"
        # else:
        #     root = os.path.join(root, "LFW")
        return LFW_match_or_missmatch_pairs_dataset(root=root, img_dir="imgs", match=True, shuffle=False, transform=transform, train=False)
        # return ImageFolderNoLabels(root=root, transform=transform)
    elif name == "CelebA":
        return CelebA(root, split=split, multiple_images=multiple_images, target_type="identity", transform=transform)
    elif name == "CelebA_HQ":
        root = os.path.join(root, "CelebAMask-HQ", "CelebA-HQ-img")
        # if whereIam == "MSI":
        #     root = root + "\\CelebA-HQ"
        # else:
        #     root = os.path.join(root, "CelebA-HQ")
        return CelebA_HQ(root=root, transform=transform, clip_image_processor = clip_image_processor)
    else:
        raise ValueError(f"Unknown dataset: {name}")



