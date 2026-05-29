from easydict import EasyDict as edict
import os
 
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_code_root = os.path.dirname(os.path.dirname(_repo_root))

config = edict()

config.root = _repo_root
config.data_root = os.path.join(_code_root, "data")
config.weights_root = os.path.join(config.root, "weights")
config.models_root = os.path.join(config.weights_root, "models")

config.dataset_roots = edict()
config.dataset_roots.DATASET = "FFHQ" # "LFW" "CelebA_HQ", "FFHQ"
config.dataset_roots.FFHQ = os.path.join(config.data_root, "FlickrFace")
config.controlnet_conditioning = "segmentation_masks" # "landmarks_5", "landmarks_26", "segmentation_masks"
config.dataset_roots.FFHQ_latents = os.path.join(config.dataset_roots.FFHQ, config.controlnet_conditioning)

config.seed = 1111
config.task_name = "ffhq"
config.batch_size = 16
config.save_path = os.path.join(config.weights_root, "ffhq_controlnet", config.controlnet_conditioning)
config.controlnet_epochs = 15
# config.num_samples = 2
# config.num_grid_rows = 2
# config.autoencoder_lr = 0.00001
config.controlnet_lr = 0.00001
config.controlnet_lr_steps = [10]
config.controlnet_save_interval = 5

