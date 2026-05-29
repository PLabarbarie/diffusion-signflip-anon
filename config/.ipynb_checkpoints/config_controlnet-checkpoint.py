from easydict import EasyDict as edict
import time

day = time.strftime("%d", time.localtime())
month = time.strftime("%m", time.localtime())
 

config = edict({
    "seed": 1111,
    "task_name": "ffhq",
    "batch_size": 16,
    "save_path": "weights/ffhq_controlnet/segmentation_masks",
    "controlnet_epochs": 15,
    # "num_samples": 2,
    # "num_grid_rows": 2,
    # "autoencoder_lr": 0.00001,
    "controlnet_lr": 0.00001,
    "controlnet_lr_steps": [10],
    "controlnet_save_interval": 5,
})

