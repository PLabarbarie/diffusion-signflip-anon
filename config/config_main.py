from easydict import EasyDict as edict
import os
 

config = edict()

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_code_root = os.path.dirname(os.path.dirname(_repo_root))

config.root = _repo_root
config.data_root = os.path.join(_code_root, "data")
config.weights_root = os.path.join(config.root, "weights")
config.models_root = os.path.join(config.weights_root, "models")

config.dataset_roots = edict()
config.dataset_roots.FFHQ = os.path.join(config.data_root, "FlickrFace")
config.dataset_roots.FFHQ_latents = config.dataset_roots.FFHQ
config.dataset_roots.LFW = os.path.join(config.data_root, "LFW")
config.dataset_roots.CelebA_HQ = os.path.join(config.data_root, "CelebAMask-HQ", "CelebA-HQ-img")

config.DATASET = "CelebA_HQ" # "LFW" "CelebA_HQ", "FFHQ"
config.batch_size = 5

config.diffusion_name = "controlnet_segmentation_masks" # controlnet_landmarks_5, controlnet_landmarks_26, controlnet_segmentation_masks, ffhq, multi_controlnet
config.guidance = False # default = False
config.guidance_type = "segmentation" # "landmarks", "segmentation"
config.controlnet = True # default = True
config.ddim_inversion = False # default = False
config.evaluation = "diversity" # "standard", "diversity" "diff_cryptanalysis" "jpeg_compression" "test" "ai_detection"

if config.guidance == True:
    config.guidance_scale = 5
else : 
    config.guidance_scale = None

config.fr_models = ["facenet_vggface2", "facenet_casia", "arcface_r50"] # "facenet_vggface2", "facenet_casia", "arcface_r50", "arcface_r34"

config.mask = "anonymized_face" # "original_face", "anonymized_face"
config.seeds = "identity"

if config.ddim_inversion:
    prefix = "ddim_inversion"
else :
    prefix = "controlnet_inversion"

if config.guidance == True:
    config.save_path_suffix = f"conditionned_{config.guidance_type}{config.guidance_scale}_mask_{config.mask}_{prefix}"
    # config.save_path_suffix = f"test"
elif config.controlnet == True:
    config.save_path_suffix = f"{config.diffusion_name}_mask_{config.mask}_{prefix}"
else :
    config.save_path_suffix = f"unconditionned_mask_{config.mask}_{prefix}"

if config.evaluation in ["diversity", "diff_cryptanalysis", "jpeg_compression", "test_noise", "ai_detection"]:
    if config.evaluation == "test_noise":
        config.noise_level = 0.5
        config.evaluation = config.evaluation + f"_{config.noise_level}"
    config.save_path_suffix = os.path.join(config.evaluation, config.save_path_suffix)
    if "test_noise" in config.evaluation:
        config.evaluation = "test_noise"
    config.n_size = 1000
elif config.evaluation == "standard":
    config.n_size = 2500

