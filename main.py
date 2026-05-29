from anonymization.diffusion import DiffusionModel
from fr_system.bisenet import BiSeNet
from fr_system.mtcnn import MTCNN
from fr_system.fr_pipeline import FRPipeline
from fr_system.face_detector import get_face_detector
from data.dataset import load_dataset
from data.transforms import get_base_transforms, JPEG_compression
from anonymization.flip_anonymizer import FlipAnonymizer, generate_keys
from utils.image_utils import save_image, tensor_to_pil, float_to_uint8, min_max_normalize, Image_metrics
from utils.utility_metrics import utility_metrics
import torch
from torch.utils.data import DataLoader
import tqdm
from torchvision.utils import make_grid
import os
from torchvision import transforms
import numpy as np
from PIL import Image
import argparse
# from fr_models import FR_model
from torchmetrics.image.fid import FrechetInceptionDistance
from config.config_main import config



argparse = argparse.ArgumentParser()
argparse.add_argument("--n", type=int, default=0, help="which part of the dataset to process")
argparse = argparse.parse_args()

def convert_to_PIL(x):
    # Convert tensor to PIL image
    x = x.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
    x = (x * 255).astype(np.uint8)  # Scale to [0, 255]
    return Image.fromarray(x)

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = config.batch_size
    DATASET = config.DATASET
    root = config.root


    # Load the diffusion model and anonymizer
    diffusion_name = config.diffusion_name
    diffusion_model = DiffusionModel(name=diffusion_name,
                                    inverse_scheduler="hugging_face", torch_device=device,
                                    models_root=config.models_root, weights_root=config.weights_root)
    diffusion_model.to_device(device)
    face_parser = BiSeNet(num_classes=19, backbone_name="resnet34")
    face_parser.load_state_dict(torch.load(os.path.join(config.models_root, "bisenet", "resnet34.pt")
                                           ,map_location=device))
    face_parser = face_parser.to(device).eval()

    mtcnn = get_face_detector(detector_name="mtcnn", device="cuda", input_size=256)

    flip_anonymizer = FlipAnonymizer(diffusion_model, face_parser, mtcnn, config.guidance, config.guidance_type,
                                     config.guidance_scale, config.controlnet, config.ddim_inversion)
    del diffusion_model

    # Load dataset (synthetic or real)
    tfms = get_base_transforms(DATASET)
    dataset = load_dataset(DATASET, transform=tfms, split="all", multiple_images=True,
                           data_root=config.data_root, dataset_roots=config.dataset_roots) # or "LFW"
    # process only sample [n: n+1] of the dataset
    if DATASET == "CelebA_HQ":
        if config.evaluation == "standard":
            dataset = torch.utils.data.Subset(dataset, range(config.n_size*argparse.n, config.n_size*(argparse.n+1))) # for lfw 0, 1000
    if config.evaluation in ["diversity", "diff_cryptanalysis", "jpeg_compression", "test_noise", "ai_detection"]:
        dataset = torch.utils.data.Subset(dataset, range(0, config.n_size)) # for diversity we choose the same images for all parts
    print(f"Dataset size: {len(dataset)}")
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if config.evaluation == "jpeg_compression":
        qf = [100,90,75,50]
        jpeg_cp = JPEG_compression(qf)
    elif config.evaluation == "test_noise":
        blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(config.noise_level, config.noise_level))

    # Initialize face recognition pipelines
    face_recognizer_models = config.fr_models
    fr_pipelines = {
        model: FRPipeline(face_detector_name="mtcnn", face_recognizer_name=model, device="cuda", models_dir=config.models_root)
        for model in face_recognizer_models
    }

    # Initialize storage for all models
    templates = {model: {'clean': [], 'anonymized': [], 'reconstructed': []} for model in face_recognizer_models}


    all_keys = []
    all_masks = []

    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize = False, input_img_size=(3, 256, 256)).to("cuda")
    fid_recon = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize = False, input_img_size=(3, 256, 256)).to("cuda")
    img_metrics = Image_metrics(data_range=(0,255.), n_type="-1,1")

    retina_net = get_face_detector(detector_name="retinaface", device="cuda", network="resnet50")
    
    utility_maintenance_metrics = {"anonymization": {"face_detection": [],
                                                      "bbox_distance": [],
                                                      "landmark_distance": [],
                                                      "iou_mask" : [],
                                                      "dice_mask" : []},
                                    "reconstruction": {"face_detection": [],
                                                      "bbox_distance": [],
                                                      "landmark_distance": []}}

    mae = torch.empty(0)
    ssim = torch.empty(0)
    psnr = torch.empty(0)

    if config.evaluation == "standard":
        keys_presaved = torch.load("keys/keys_CelebA_HQ.pt").cuda()
        keys_presaved = keys_presaved[config.n_size*argparse.n : config.n_size*(argparse.n+1)]
    elif config.evaluation in ["diversity", "jpeg_compression", "test_noise", "ai_detection"]:
        keys_presaved = torch.load(f"keys/sub_keys_diversity_{argparse.n}.pt").cuda()
        if config.evaluation == "jpeg_compression":
            for model in face_recognizer_models:
                for q in qf:
                    templates[model][q] = []
    elif config.evaluation == "diff_cryptanalysis":
        for model in face_recognizer_models:
            templates[model]["wrong_lsb"] = []
            templates[model]["wrong_rand"] = []
    for i, images in enumerate(tqdm.tqdm(dataset)):

        if config.evaluation in ["standard", "diversity", "jpeg_compression", "test_noise", "ai_detection"]:
            keys = keys_presaved[i*batch_size : (i+1)*batch_size]
        elif config.evaluation == "diff_cryptanalysis":
            keys = generate_keys(shape=torch.Size([batch_size,3,64,64]), p=0.5, key_seed=42, device=device)
            keys_lsb = generate_keys(shape=torch.Size([batch_size,3,64,64]), p=0.5, key_seed=42 ^1, device=device)
            keys_random = generate_keys(shape=torch.Size([batch_size,3,64,64]), p=0.5, key_seed=100, device=device)


        # print(images)
        if DATASET == "LFW":
            images = images[1]  # for lfw following G2Face benchmark we anonymize the second image of positive pairs
        else :
            images = images[0]


        # images, y = images
        images = images.to(device)
        # y = y.to(device)

        # Compute FID for real imagess
        fid.update(float_to_uint8(images), real=True)
        fid_recon.update(float_to_uint8(images), real=True)

        pil_images = make_grid(images, nrow=batch_size)
        if config.evaluation == "test_noise":
            pil_blurred_images = make_grid(blur(images), nrow=batch_size)

        # keys = flip_anonymizer.generate_keys((batch_size, flip_anonymizer.DM.unet.in_channels, flip_anonymizer.DM.unet.sample_size,
        #                                        flip_anonymizer.DM.unet.sample_size), p=0.5, seeds=None, device=device)

        # x_ano, _ = flip_anonymizer.anonymization_pipeline(images, keys)
        if i < 5 :
            flip_anonymizer.saving_path = os.path.join(root, "results", f"{DATASET}", f"{config.save_path_suffix}", "int_l", f"landmarks_{2500*argparse.n + i}")
        else : 
            flip_anonymizer.save_intermediate_landmarks = False
        p = 0.5
        x_ano, keys, mask, _ = flip_anonymizer.anonymization_pipeline(images, keys=keys)
        if config.evaluation == "test_noise":
            x_ano_blurred, _, mask_blurred, _ = flip_anonymizer.anonymization_pipeline(blur(images), keys=keys)
            pil_blurred_ano = make_grid(x_ano_blurred, nrow=batch_size)
            


        all_keys.append(keys.cpu().detach())
        all_masks.append(mask.cpu().detach())

        pil_mask = make_grid(transforms.Resize((256, 256))(mask), nrow=batch_size)
        pil_ano = make_grid(x_ano, nrow=batch_size)


        if config.mask == "original_face":
            mask_for_recon = mask
        elif config.mask == "anonymized_face":
            mask_for_recon = flip_anonymizer.face_parser.get_face_mask(x_ano, size=(64,64))
            mask_for_recon = (mask_for_recon > 0)*1.0
            pil_mask = make_grid(transforms.Resize((256, 256))((flip_anonymizer.face_parser.get_face_mask(x_ano, size=(256,256))>0)*1.0), nrow=batch_size)




        

        x_rec = flip_anonymizer.de_anonymization_pipeline(x_ano, keys, mask_for_recon)     
        pil_rec = make_grid(x_rec, nrow=batch_size)

        if config.evaluation == "diff_cryptanalysis":
            x_rec_lsb = flip_anonymizer.de_anonymization_pipeline(x_ano, keys_lsb, mask_for_recon)
            x_rec_random = flip_anonymizer.de_anonymization_pipeline(x_ano, keys_random, mask_for_recon)
        elif config.evaluation == "jpeg_compression":
            x_ano_jpeg = [jpeg_cp.compression(x_ano, q) for q in qf]
            x_rec_jpeg = [flip_anonymizer.de_anonymization_pipeline(x_ano_j, keys, mask_for_recon) for x_ano_j in x_ano_jpeg]
        elif config.evaluation == "test_noise":
            x_blurred_rec= flip_anonymizer.de_anonymization_pipeline(x_ano_blurred, keys, mask_blurred)    

        # x_rec_ano = flip_anonymizer.de_anonymization_pipeline(x_ano, keys, mask_ano)
        # pil_reco_ano = make_grid(x_rec_ano, nrow=batch_size)

        if config.evaluation in ["standard", "diversity", "jpeg_compression"]:
            pil = torch.cat([pil_images, pil_mask, pil_ano, pil_rec], dim=1)
        elif config.evaluation == "diff_cryptanalysis":
            pil = torch.cat([pil_images, pil_mask, pil_ano, pil_rec,
                             make_grid(x_rec_lsb, nrow=batch_size),
                             make_grid(x_rec_random, nrow=batch_size)], dim=1)
        elif config.evaluation == "test_noise":
            pil = torch.cat([pil_images, pil_blurred_images, pil_ano, pil_blurred_ano,
                             pil_rec, make_grid(x_blurred_rec, nrow=batch_size)], dim=1)
        if i<=5 and config.evaluation != "ai_detection":
            pil = tensor_to_pil(pil.unsqueeze(0))
            save_image(pil[0], os.path.join(root, "results", f"{DATASET}", f"{config.save_path_suffix}", f"image_{config.n_size*argparse.n + i}.png"))
        if i<=5 and config.evaluation == "jpeg_compression":
            pil_ano_jpeg = [make_grid(x_ano_j, nrow=batch_size) for x_ano_j in x_ano_jpeg]
            pil_rec_jpeg = [make_grid(x_rec_j, nrow=batch_size) for x_rec_j in x_rec_jpeg]
            pil_jpeg = torch.cat([pil_images] +  pil_ano_jpeg +  pil_rec_jpeg, dim=1)
            pil_jpeg = tensor_to_pil(pil_jpeg.unsqueeze(0))
            save_image(pil_jpeg[0], os.path.join(root, "results", f"{DATASET}", f"{config.save_path_suffix}", f"image_jpeg_{config.n_size*argparse.n + i}.png"))
        if i<=5 and config.evaluation == "test_noise":
            pil_mask = torch.cat([pil_mask, make_grid(transforms.Resize((256, 256))((flip_anonymizer.face_parser.get_face_mask(x_ano_blurred, size=(256,256))>0)*1.0), nrow=batch_size)], dim=1)
            pil_mask = tensor_to_pil(pil_mask.unsqueeze(0))
            save_image(pil_mask[0], os.path.join(root, "results", f"{DATASET}", f"{config.save_path_suffix}", f"masks_{config.n_size*argparse.n + i}.png"))
        if config.evaluation == "ai_detection":
            for b in range(batch_size):
                x_ano_b = tensor_to_pil(x_ano[b].unsqueeze(0))
                x_recover_b = tensor_to_pil(x_rec[b].unsqueeze(0))

                save_image(x_ano_b[0], os.path.join(root, "results", f"{DATASET}", f"{config.save_path_suffix}", "ano", f"{i*batch_size+b}.jpg"))
                save_image(x_recover_b[0], os.path.join(root, "results", f"{DATASET}", f"{config.save_path_suffix}", "rec", f"{i*batch_size+b}.jpg"))

        if config.evaluation == "test_noise":
            x_ano = x_ano_blurred.clone()
            x_rec = x_blurred_rec.clone()

        # Compute visual metrics for anonymization and reconstruction
        mae = torch.cat((mae, (images.view(batch_size, -1) - x_rec.clamp(-1,1).view(batch_size, -1)).abs().view(batch_size,-1).mean(1).cpu()))
        ssim = torch.cat((ssim, img_metrics.compute_ssim(images.cpu(), x_rec.cpu())))
        psnr = torch.cat((psnr, img_metrics.compute_psnr(images.cpu(), x_rec.cpu())))
        fid.update(float_to_uint8(x_ano), real=False)
        fid_recon.update(float_to_uint8(x_rec), real=False)

        # Compute image utility metrics
        retinadetections = retina_net.detect_faces(images)
        retinadetections_ano = retina_net.detect_faces(x_ano)
        retinadetections_rec = retina_net.detect_faces(x_rec)

        # print(retinadetections_ano[0].shape)
        # print(retinadetections_ano[1].shape)
        # print(retinadetections_ano[2].shape)
        utility_maintenance_metrics["anonymization"]["face_detection"].append(retinadetections_ano[2] > 0)
        utility_maintenance_metrics["reconstruction"]["face_detection"].append(retinadetections_rec[2] > 0)
        ldms = retinadetections[1]
        ldms_ano = retinadetections_ano[1]
        ldms_rec = retinadetections_rec[1]

        bbox = retinadetections[0]
        bbox_ano = retinadetections_ano[0]
        bbox_rec = retinadetections_rec[0]
        utility_maintenance_metrics["anonymization"]["bbox_distance"].append(
            utility_metrics.bbox_l2_distance(bbox, bbox_ano, unique_face=True))
        utility_maintenance_metrics["reconstruction"]["bbox_distance"].append(
            utility_metrics.bbox_l2_distance(bbox_rec, bbox_ano, unique_face=True))
        utility_maintenance_metrics["anonymization"]["landmark_distance"].append(
            utility_metrics.landmarks_l2_distance(ldms, ldms_ano, unique_face=True))
        utility_maintenance_metrics["reconstruction"]["landmark_distance"].append(
            utility_metrics.landmarks_l2_distance(ldms, ldms_rec, unique_face=True))

        mask_ori = ((flip_anonymizer.face_parser.get_face_mask(images, size=(256,256))>0)*1.0).cpu().detach()
        mask_ano  = ((flip_anonymizer.face_parser.get_face_mask(x_ano, size=(256,256))>0)*1.0).cpu().detach()
        utility_maintenance_metrics["anonymization"]["iou_mask"].append(img_metrics.iou(mask_ori, mask_ano))
        utility_maintenance_metrics["anonymization"]["dice_mask"].append(img_metrics.dice(mask_ori, mask_ano))

        # Compute face recognition templates
        for model_name, fr_pipeline in fr_pipelines.items():
            # Compute templates for clean, anonymized and reconstructed images
            for img_name, face in zip(["clean", "anonymized", "reconstructed"], [images, x_ano, x_rec]):
                with torch.no_grad():
                    face_features = fr_pipeline(face)
                templates[model_name][img_name].append(face_features.cpu())     

            if config.evaluation == "diff_cryptanalysis":
                # Compute templates for wrongly reconstructed images
                for img_name, face in zip(["wrong_lsb", "wrong_rand"], [x_rec_lsb, x_rec_random]):
                    with torch.no_grad():
                        face_features = fr_pipeline(face)
                    templates[model_name][img_name].append(face_features.cpu())       

            if config.evaluation == "jpeg_compression":
                for img_name, face in zip(qf, x_rec_jpeg):
                    with torch.no_grad():
                        face_features = fr_pipeline(face)
                    templates[model_name][img_name].append(face_features.cpu())                      
        # if i>5:
        #     break

    ################################################################################
    ############################## Results saving ##################################
    ################################################################################

    for key, value in utility_maintenance_metrics.items():
        for subkey, subvalue in value.items():
            utility_maintenance_metrics[key][subkey] = torch.from_numpy(np.concatenate(subvalue, axis=0))

    results = {'keys': torch.cat(all_keys, dim=0),
               'mae': mae,
               'ssim': ssim,
               'psnr': psnr,
               'fid': fid.compute().item(),
               'fid_recon': fid_recon.compute().item(),
                'utility_maintenance_metrics': utility_maintenance_metrics
               }
    
    for model_name in templates.keys():
        if templates[model_name]:  # Check if we have any templates
            results[model_name] = {}
            for img_type in templates[model_name].keys():
                templates[model_name][img_type] = torch.cat(templates[model_name][img_type], dim=0)
                print(f"Concatenating templates for model: {model_name}, image type: {img_type}, shape: {templates[model_name][img_type].shape}")
                results[model_name][img_type] = templates[model_name][img_type]

    os.makedirs(f"results/{DATASET}/{config.save_path_suffix}", exist_ok=True)
    torch.save(results, f"results/{DATASET}/{config.save_path_suffix}/results_part_{argparse.n}.pt")
            # torch.save({
            #     'templates': final_templates[model_name],
            #     'templates_ano': final_ano_templates[model_name],
            #     'templates_recon': final_recon_templates[model_name],
            #     'keys': torch.cat(all_keys, dim=0)
            # }, f"results/{DATASET}/{config.save_path_suffix}/{DATASET}_{model_name}_{argparse.n}_templates.pt")

    print("Results saved!")




if __name__ == "__main__":
    main()
