import time

import cv2

from rtmlib import Body, draw_skeleton, RTMPose, RTMO, Wholebody

import matplotlib.pyplot as plt
import platform
import os
import numpy as np
from PIL import Image
import argparse

def save_image(image: Image.Image, path: str) -> None:
    """Save a PIL image to a specified path."""
    # print(path, path.split('/')[0])
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    image.save(path)

def convert_to_PIL(x):
    # Convert tensor to PIL image
    x = (x * 255).astype(np.uint8)  # Scale to [0, 255]
    return Image.fromarray(x)


device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

cap = cv2.VideoCapture(0)


openpose_skeleton = False  # True for openpose-style, False for mmpose-style

whereIam = platform.node()
if whereIam == "MSI":
    root = "C:\\Users\\Pol\\Documents\\code\\models"
    ffhq_root = "C:\\Users\\Pol\\Documents\\code"
    save_folder = ffhq_root
else:
    root = "/lustre/fswork/projects/rech/irm/uzq68by/code/models"
    ffhq_root = '/lustre/fsmisc/dataset'
    save_folder = "/lustre/fswork/projects/rech/irm/uzq68by/data"


argparse = argparse.ArgumentParser()
argparse.add_argument("--n", type=int, default=0, help="which part of the dataset to process")
argparse = argparse.parse_args()

whole = Wholebody(pose=os.path.join(root, 'rtmw-dw-x-l-384x288/end2end.onnx'),  # download link or local path
                    det_input_size = (640,640),
                    det = os.path.join(root, 'yolo_m_8/end2end.onnx'),
                     backend=backend, device=device)

for i in range(argparse.n * 7000, (argparse.n + 1) * 7000):
    # 00000 to 69999
    # i = 75
    image = cv2.imread(os.path.join(ffhq_root ,f"FlickrFace/images1024x1024/{i:05d}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    # plt.imshow(image)
    # plt.show()
    # break 
    bboxes, keypoints, scores = whole(image)
    # d_eyes = ((keypoints[:,1] - keypoints[:,2])**2).s
    # um(1)**0.5
    # print(d_eyes)
    if len(bboxes) != 0:
        largest_bbox_ind = ((bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])).argmax()
    else:
        largest_bbox_ind = 0
    ind = largest_bbox_ind
    keypoints = keypoints[ind].reshape(1, -1, 2)
    image_keypoints = np.zeros_like(image)
    print(f"Image {i} done.")
    for j in range(len(keypoints[0])):
        if j in [0,1,2,3,4,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,77,71,74,80]:
            x, y = int(keypoints[0][j][0]),  int(keypoints[0][j][1])
            image_keypoints[y-1:y+2, x-1:x+2, :] = 1
    
    # plt.imshow(image_keypoints * 255)
    img_landmarks = convert_to_PIL(image_keypoints)
    save_path = os.path.join(save_folder, f"FlickrFace/landmarks_26/{i:05d}.png")
    save_image(img_landmarks, save_path)
    # break