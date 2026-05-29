from easydict import EasyDict as edict
import os
 

config = edict()
config.DATASET = "CelebA_HQ" # "LFW" "CelebA_HQ", "FFHQ"
config.batch_size = 5

config.diffusion_name = "ffhq" # controlnet_landmarks_5, controlnet_landmarks_26, controlnet_segmentation_masks, ffhq, multi_controlnet
config.guidance = False # default = False
config.guidance_type = "segmentation" # "landmarks", "segmentation"
config.controlnet = False # default = True
config.ddim_inversion = True # default = False
config.evaluation = "diversity" # "standard", "diversity" "diff_cryptanalysis"

if config.guidance == True:
    config.guidance_scale = 5
else : 
    config.guidance_scale = None

config.fr_models = ["facenet_vggface2", "facenet_casia"]

config.mask = "original_face" # "original_face", "anonymized_face"
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

if config.evaluation == "diversity":
    config.save_path_suffix = os.path.join("diversity_evaluation", config.save_path_suffix)
    config.n_size = 1000
elif config.evaluation == "standard":
    config.n_size = 2500
elif config.evaluation == "diff_cryptanalysis":
    config.save_path_suffix = os.path.join("diff_cryptanalysis", config.save_path_suffix)
    config.n_size = 1000