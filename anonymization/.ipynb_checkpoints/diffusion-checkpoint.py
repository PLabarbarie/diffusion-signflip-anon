from diffusers import UNet2DModel, DDIMScheduler, VQModel, DDIMInverseScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from torchvision import transforms as tfms
from PIL import Image
import tqdm
from utils.image_utils import tensor_to_pil
import platform
import os
import matplotlib.pyplot as plt

from anonymization.controlnet import ControlNet
from anonymization.controlnet_v2 import MultiControlNet


whereIam = platform.node()
if whereIam == "MSI":
    root = "C:\\Users\\Pol\\Documents\\code\\models"
else:
    root = "/lustre/fswork/projects/rech/irm/uzq68by/code/models"

class DiffusionModel(torch.nn.Module):
    def __init__(self,name="CompVis/ldm-celebahq-256", inverse_scheduler = "hugging_face", torch_device="cuda"):
        super().__init__()
        self.name = name
        # Load diffusion models
        if name == "CompVis/ldm-celebahq-256":
            self.unet = UNet2DModel.from_pretrained(name, subfolder="unet", local_files_only=True)
            self.vae = VQModel.from_pretrained(name, subfolder="vqvae", local_files_only=True)
            self.scheduler = DDIMScheduler.from_config(name, subfolder="scheduler", local_files_only=True)
        elif name == "CompVis/stable-diffusion-v1-4":
            self.unet = UNet2DConditionModel.from_pretrained(name, subfolder="unet", use_safetensors=True)
            self.vae = AutoencoderKL.from_pretrained(name, subfolder="vae", use_safetensors=True)
            self.text_encoder = CLIPTextModel.from_pretrained(name, subfolder="text_encoder", use_safetensors=True)
            self.tokenizer = CLIPTokenizer.from_pretrained(name, subfolder="tokenizer")
            self.scheduler = DDIMScheduler.from_pretrained(name, subfolder="scheduler", use_safetensors=True)
        elif name == "ffhq":
            self.unet = UNet2DModel.from_pretrained(os.path.join(root,"ffhq-diffusers"), subfolder="unet", local_files_only=True)
            self.vae = VQModel.from_pretrained(os.path.join(root,"ffhq-diffusers"), subfolder="vqvae", local_files_only=True)
            self.scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler",
                                                        local_files_only=True)
            
        elif "controlnet" in name and "multi" not in name:
            self.vae = VQModel.from_pretrained(os.path.join(root,"ffhq-diffusers"), subfolder="vqvae", local_files_only=True)
            self.scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler",
                                                        local_files_only=True)
            unet = UNet2DModel.from_pretrained(os.path.join(root,"ffhq-diffusers"), subfolder="unet", local_files_only=True)
            unet_config = unet.config
            del unet
            self.unet = ControlNet(model_config=unet_config,
                        model_ckpt=os.path.join(root, "ffhq-diffusers"),
                        hint_channels=3,
                        down_sample_factor=4,
                        device=torch_device)
            state_dict = os.path.join(root.split('models')[0], 'anonymization' , "weights", "ffhq_controlnet")
            if "landmarks_5" in name:
                state_dict = os.path.join(state_dict, "landmarks_5", "controlnet_epoch_15.pth")
            elif "landmarks_26" in name:
                state_dict = os.path.join(state_dict, "landmarks_26", "controlnet_epoch_15.pth")
            elif "segmentation_masks" in name:
                state_dict = os.path.join(state_dict, "segmentation_masks", "controlnet_epoch_15.pth")
            self.unet.load_state_dict(torch.load(state_dict, map_location=torch_device))

        elif "multi_controlnet" in name:
            self.vae = VQModel.from_pretrained(os.path.join(root,"ffhq-diffusers"), subfolder="vqvae", local_files_only=True)
            self.scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler",
                                                        local_files_only=True)
            unet = UNet2DModel.from_pretrained(os.path.join(root,"ffhq-diffusers"), subfolder="unet", local_files_only=True)
            unet_config = unet.config
            del unet

            controlnet_segmentation = ControlNet(model_config=unet_config,
                                    model_ckpt=os.path.join(root, "ffhq-diffusers"),
                                    hint_channels=3,
                                    down_sample_factor=4,
                                    device=torch_device)
            state_dict = os.path.join(root.split('models')[0], 'anonymization' , "weights", "ffhq_controlnet")
            state_dict = os.path.join(state_dict, "segmentation_masks", "controlnet_epoch_15.pth")
            controlnet_segmentation.load_state_dict(torch.load(state_dict, map_location=torch_device))
            controlnet_segmentation.to_device(torch_device)
            controlnet_segmentation.eval()

            controlnet_landmarks = ControlNet(model_config=unet_config,
                                    model_ckpt=os.path.join(root, "ffhq-diffusers"),
                                    hint_channels=3,
                                    down_sample_factor=4,
                                    device=torch_device)
            state_dict = os.path.join(root.split('models')[0], 'anonymization' , "weights", "ffhq_controlnet")
            state_dict = os.path.join(state_dict, "landmarks_5", "controlnet_epoch_15.pth")
            controlnet_landmarks.load_state_dict(torch.load(state_dict, map_location=torch_device))
            controlnet_landmarks.to_device(torch_device)
            controlnet_landmarks.eval()

            self.unet = MultiControlNet(
                controlnets=[controlnet_segmentation, controlnet_landmarks],
                control_scales=[0.5, 0.5]  # Scale second control to 70%
            )



        self.torch_device = torch_device
        if inverse_scheduler == "hugging_face":
            self.inverse_scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
            self.ddim_inversion = self.ddim_inversion_diffusers
        elif inverse_scheduler == "custom":
            self.ddim_inversion = self.ddim_inversion_unconditional

    def to_device(self, device):
        self.unet.to(device).eval()
        self.vae.to(device).eval()
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(device).eval()
        pass

    @torch.no_grad()
    def pil_to_latent(self, input_im):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        latent = self.vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(self.torch_device)*2-1) # Note scaling
        # return 0.18215 * latent.latent_dist.sample()
        return latent.latents

    @torch.no_grad()
    def latents_to_pil(self, latents):
        # bath of latents -> list of images
        # latents = (1 / 0.18215) * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):
        """
        Encode image into latent space.
        image: (B, 3, H, W), pixel values in R
        """
        latent = self.vae.encode(image).latents
        # scale latents (important for SD convention)
        # latent = latent * 0.18215
        return latent


    def decode_latent(self, latent: torch.Tensor):
        """
        Decode latent into image space.
        latent: (B, 4, H/8, W/8)
        """
        # latent = latent / 0.18215
        image = self.vae.decode(latent).sample
        return image  # (B, 3, H, W), in [-1, 1]

    def forward_diffusion(self, latent, t):
        """
        Diffuse a latent to timestep t.
        t: torch.Tensor of shape (B,) or int timestep
        """
        noise = torch.randn_like(latent)
        noisy_latent = self.scheduler.add_noise(latent, noise, t)
        return noisy_latent, noise

    @torch.no_grad()
    def backward_diffusion(self, noisy_latent, mask = 1, encoder_hidden_states = None, num_inference_steps=50, eta=0.0):
        # Set timesteps for generation
        self.scheduler.set_timesteps(num_inference_steps)
        with torch.no_grad():
            # Denoising loop
            image = noisy_latent
            for t in tqdm.tqdm(self.scheduler.timesteps):
                image = image * mask
                # Predict noise
                if hasattr(self, "text_encoder"):
                    # for stable diffusion
                    residual = self.unet(image, t, encoder_hidden_states)["sample"]
                else:
                    residual = self.unet(image, t)["sample"]
                # Update noise
                prev_image = self.scheduler.step(residual, t, image, eta=eta)["prev_sample"]
                # x_t-1 -> x_t
                image = prev_image
        return image
    
    @torch.no_grad()
    def masked_backward_diffusion(self, noisy_latents, mask=1, hint=None, num_inference_steps=50, eta=0.0):
          # Set timesteps for generation
        self.scheduler.set_timesteps(num_inference_steps)
        with torch.no_grad():
            # Denoising loop
            it = len(self.scheduler.timesteps)
            xknown = noisy_latents[-1] * (1 - mask)
            xunknown = noisy_latents[-1] * mask
            xt = xknown + xunknown
            # lxt = [xt.clone()]
            lxt = 0
            for t in tqdm.tqdm(self.scheduler.timesteps):
                # Predict noise
                if hint == None:
                    residual = self.unet(xt, t)["sample"]
                else:
                    residual = self.unet(xt, t, hint)
                # Update noise
                prev_image = self.scheduler.step(residual, t, xt, eta=eta)["prev_sample"]
                # x_t-1 -> x_t
                it = it - 1
                # print(it)
                xknown = noisy_latents[it] * (1 - mask)
                xunknown = prev_image * mask
                xt = xknown + xunknown

        return xt, lxt      

    @torch.no_grad()
    def backward_diffusion2(self, noisy_latent, mask = 1, num_inference_steps=50, eta=0.0):
        # Set timesteps for generation
        self.scheduler.set_timesteps(num_inference_steps)
        with torch.no_grad():
            # Denoising loop
            it = len(self.scheduler.timesteps)
            xknown = noisy_latent[-1] * (1 - mask)
            xunknown = noisy_latent[-1] * mask
            xt = xknown + xunknown
            # lxt = [xt.clone()]
            lxt = 0
            for t in tqdm.tqdm(self.scheduler.timesteps):
                # Predict noise
                residual = self.unet(xt, t)["sample"]
                # Update noise
                prev_image = self.scheduler.step(residual, t, xt, eta=eta)["prev_sample"]
                # x_t-1 -> x_t
                it = it - 1
                # print(it)
                xknown = noisy_latent[it] * (1 - mask)
                xunknown = prev_image * mask
                xt = xknown + xunknown

        return xt, lxt

    def conditional_backward_diffusion(self, noisy_latent, landmarks, conditional_model, guidance_scale=1,
    mask = 1, num_inference_steps=50, eta=0.0, save_intermediate_landmarks=True, save_path=None):

        # Set timesteps for generation
        self.scheduler.set_timesteps(num_inference_steps)
            # Denoising loop
        it = len(self.scheduler.timesteps)
        # print("diff = ",(noisy_latent[it] * (1 - mask) - noisy_latent[-1] * (1 - mask)).abs().sum())
        # lxt = [xt.clone()]
        lxt = 0
        for t in tqdm.tqdm(self.scheduler.timesteps):
            xknown = noisy_latent[it] * (1 - mask)
            if it == len(self.scheduler.timesteps):
                xunknown = noisy_latent[it] * mask
            else : 
                xunknown = prev_image * mask
            xunknown = xunknown.detach().requires_grad_(True)
            xt = xknown + xunknown

            # Predict noise
            # xt = xt.detach().requires_grad_(True)
            residual = self.unet(xt, t)["sample"]
            if t>0:
                # predict condition
                z0_hat = self.scheduler.step(residual, t, xt, eta=eta)["pred_original_sample"]
                x0_hat = self.decode_latent(z0_hat)

                condition = conditional_model.extract_landmarks(x0_hat)
                l2 = ((condition - landmarks)**2).sum((1,2))
                l2 = torch.log(l2)
                # print(l2)
                grad = torch.autograd.grad(l2.sum(), xt)[0]
                # xt = xt.detach() - 2 * grad * mask

            # print(condition)
            if save_intermediate_landmarks and  t%100 == 0 :
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.clf()
                plt.imshow((x0_hat[0]/2 + 0.5).detach().cpu().permute(1,2,0).numpy())
                # Plot landmarks if available
                # print("condition", condition.shape)
                if condition is not None and len(condition) > 0:
                    # Get landmarks for the first image
                    face_landmarks = condition[0].detach().cpu().numpy()
                    # print(face_landmarks.shape)
                    # print("face landmkars ", face_landmarks)
                    if face_landmarks is not None:
                        # Plot each landmark point
                        for landmark in face_landmarks:
                            plt.plot(landmark[0], landmark[1], 'ro', markersize=6)
                        for landmark in landmarks[0].detach().cpu().numpy():
                            plt.plot(landmark[0], landmark[1], 'bo', markersize=6)

                plt.savefig(os.path.join(save_path, f'{int(t.item())}.png'))


            # xt.grad = None
            # Update noise
            residual = residual + guidance_scale * torch.sqrt(1-self.scheduler.alphas_cumprod[t]) * grad
            prev_image = self.scheduler.step(residual, t, xt, eta=eta)["prev_sample"]
            # if t > 0 :
            #     xt = prev_image.detach() - 0.8 * self.scheduler.sigmas[t] * grad * mask
            # xt = xt.detach()
            xt = prev_image.detach()

            # if t == 800:
            #     for i in range(100):
            #         xknown = noisy_latent[it] * (1 - mask)
            #         if it == len(self.scheduler.timesteps):
            #             xunknown = noisy_latent[it] * mask
            #         else : 
            #             xunknown = prev_image * mask
            #         xt = xknown + xunknown
            #         xt_copy = xt.clone().detach()
            #         xt_copy.requires_grad = True

            #         # Predict noise
            #         # xt = xt.detach().requires_grad_(True)
            #         residual = self.unet(xt_copy, t)["sample"]
            #         # predict condition
            #         z0_hat = self.scheduler.step(residual, t, xt_copy, eta=eta)["pred_original_sample"]
            #         x0_hat = self.decode_latent(z0_hat)

            #         condition = conditional_model.extract_landmarks(x0_hat)

            #         # print(condition.shape, landmarks.shape)
            #         l2 = ((condition - landmarks)**2).sum((1,2))
            #         print("inside loop", l2)
            #         # l2 = l2.sum()
            #         # print("summed l2", l2)
            #         # l2.backward()

            #         grad = torch.autograd.grad(l2.sum(), xt_copy)[0]

            #         xt = (xt - 2 * grad * mask).detach()

            #         # xt = (xt/xt.norm() - 0.2 * grad/grad.norm() * mask)*xt.norm()
            #         # print(grad * mask)
            #         # xt.grad = None
            #         # Update noise
            #         prev_image = self.scheduler.step(residual, t, xt, eta=eta)["prev_sample"]



            # x_t-1 -> x_t
            it = it - 1
            # print(it)
        
                # print(((xt - noisy_latent[it]) * (1-mask)).abs().sum())
                # lxt.append(xt.clone())

        # print("tutu",((xt - noisy_latent[0]) * (1-mask)).abs().sum())
        # print("tutu",((xt - noisy_latent[1]) * (1-mask)).abs().sum())
        # print("tutu",((xt - noisy_latent[2]) * (1-mask)).abs().sum())
        # print("tutu",((xt - noisy_latent[3]) * (1-mask)).abs().sum())
        return xt, lxt



    @torch.no_grad()
    def backward_diffusion_repaint(
        self,
        noisy_latent,
        mask=1,
        num_inference_steps=50,
        eta=0.0,
        jump_length=10,      # how far back we jump
        jump_n_sample=5      # how many times to resample when we jump
    ):
        """
        RePaint-style backward diffusion with mask, jumping, and resampling.
        noisy_latent: list of latents from the forward (DDIM inverse) process.
        mask: 1 = inpaint region, 0 = keep region.
        """

        # Set timesteps for generation
        self.scheduler.set_timesteps(num_inference_steps)
        self.inverse_scheduler.set_timesteps(num_inference_steps)
        self.inverse_scheduler.config.prediction_type = "epsilon"
        self.inverse_scheduler.config.clip_sample = False
        self.inverse_scheduler.config.timestep_spacing = "leading"

        with torch.no_grad():
            # Denoising loop
            it = len(self.scheduler.timesteps)
            xknown = noisy_latent[-1] * (1 - mask)
            xunknown = noisy_latent[-1] * mask
            xt = xknown + xunknown
            # lxt = [xt.clone()]
            lxt = 0
            for step, t in enumerate(tqdm.tqdm(self.scheduler.timesteps)):
                print(step, t)
                # Predict noise
                residual = self.unet(xt, t)["sample"]
                # Update noise
                prev_image = self.scheduler.step(residual, t, xt, eta=eta)["prev_sample"]
                # x_t-1 -> x_t
                it = it - 1
                # print(it)
                xknown = noisy_latent[it] * (1 - mask)
                xunknown = prev_image * mask
                xt = xknown + xunknown

                if step % jump_length == 0 and step > 0:

                    intermediates_xt = [xt.clone()]
                    for _ in range(jump_n_sample):
                        print("jumping")
                        print(step, jump_length)
                        print(step-jump_length, step)
                        print(reversed(self.inverse_scheduler.timesteps)[step-jump_length: step])
                        for t_resample in reversed(reversed(self.inverse_scheduler.timesteps)[step-jump_length: step]):
                            print("tutututututu")
                            print("t_resample = ", t_resample)
                            noise_pred = self.unet(xt, t_resample).sample
                            out = self.inverse_scheduler.step(noise_pred, t_resample, xt)
                            xt = out.prev_sample

                        for t_resample in reversed(reversed(self.scheduler.timesteps[step-jump_length: step])):
                            print("t_resample = ", t_resample)
                            noise_pred = self.unet(xt, t_resample).sample
                            out = self.scheduler.step(noise_pred, t_resample, xt)
                            xt = out.prev_sample


                xknown = noisy_latent[it] * (1 - mask)
                xunknown = xt * mask
                xt = xknown + xunknown


        return xt, lxt


    @torch.no_grad()
    def ddim_inversion_unconditional(self, z0: torch.Tensor) -> torch.Tensor:
        """
        Deterministic DDIM inversion (η=0) in latent space.
        Input:
        z0: latent (1, C, H, W) scaled by LATENT_SCALE (matching SD convention)
        Returns:
        z_T: latent at final timestep (noisy) on the DDIM trajectory
        """
        # prepare timesteps in *increasing* order so we walk from t=0 -> t=T-1
        inv_timesteps = list(reversed(self.scheduler.timesteps.cpu().numpy().tolist()))  # e.g. [0,1,2,...,49]
        z = z0.clone()

        # helper to fetch alpha_cumprod for a scalar timestep index
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.torch_device)

        for i, t in enumerate(inv_timesteps):
            t_int = int(t)
            # predict eps = epsilon_theta(z_t, t)
            # UNet in SD expects "sample" scaled by scheduler.model_input_scale? In Stable Diffusion it's handled inside UNet via
            # the scheduler's scaling of model input in pipeline. Use the same call pattern: pass z as "sample" and timestep t.
            # UNet signature: unet(sample, timestep, encoder_hidden_states=...) -> model_output
            # For unconditional we pass no encoder_hidden_states (None)
            # The output object has .sample attribute
            # Note: ensure dtype matches model (float32/float16)
            model_output = self.unet(z, torch.tensor([t_int], device=self.torch_device)).sample  # predict noise
            eps = model_output

            alpha_t = alphas_cumprod[t_int]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            # predict x0 from current z (z = z_t)
            x0_pred = (z - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t

            # compute z_{t+1} (next noisier latent) using DDIM forward formula (η=0)
            if i + 1 < len(inv_timesteps):
                t_next = int(inv_timesteps[i + 1])
                alpha_t_next = alphas_cumprod[t_next]
                sqrt_alpha_t_next = torch.sqrt(alpha_t_next)
                sqrt_one_minus_alpha_t_next = torch.sqrt(1 - alpha_t_next)

                # z_next = sqrt(alpha_next) * x0_pred + sqrt(1 - alpha_next) * eps
                z = sqrt_alpha_t_next * x0_pred + sqrt_one_minus_alpha_t_next * eps
            # else we've reached final (no further)
        # z now is approximately z_T (the noisy latent that would produce z0 under deterministic DDIM sampling)
        return z

    # @torch.no_grad()
    # def ddim_inversion_diffusers(self, z0: torch.Tensor, mask = 1, num_inference_steps=50, return_intermediates=False) -> torch.Tensor:
    #     self.inverse_scheduler.set_timesteps(num_inference_steps)
    #     self.inverse_scheduler.config.prediction_type = "epsilon"
    #     self.inverse_scheduler.config.clip_sample = False
    #     self.inverse_scheduler.config.timestep_spacing = "leading"


    #     if return_intermediates:
    #         intermediates = []
    #     z = z0.clone()
    #     for i, t in enumerate(self.inverse_scheduler.timesteps):
    #         z = z * mask
    #         with torch.no_grad():
    #             noise_pred = self.unet(z, t).sample  # unconditional
    #         out = self.inverse_scheduler.step(noise_pred, t, z)
    #         z = out.prev_sample
    #         if return_intermediates:
    #             intermediates.append(z)
    #     if return_intermediates:
    #         return z, intermediates
    #     return z


    @torch.no_grad()
    def ddim_inversion_diffusers(self, z0: torch.Tensor, hint=None, num_inference_steps=50, return_intermediates=False) -> torch.Tensor:
        self.inverse_scheduler.set_timesteps(num_inference_steps)
        self.inverse_scheduler.config.prediction_type = "epsilon"
        self.inverse_scheduler.config.clip_sample = False
        self.inverse_scheduler.config.timestep_spacing = "leading"


        if return_intermediates:
            intermediates = [z0.clone()]
        z = z0.clone()
        for i, t in enumerate(self.inverse_scheduler.timesteps):
            with torch.no_grad():
                if hint == None and self.name == "ffhq":
                    noise_pred = self.unet(z, t).sample  # unconditional
                elif hint == None and "controlnet" in self.name:
                    noise_pred = self.unet.trained_unet(z, t).sample  # unconditional
                else :
                    noise_pred = self.unet(z, t, hint)
            out = self.inverse_scheduler.step(noise_pred, t, z)
            z = out.prev_sample
            if return_intermediates:
                intermediates.append(z)
        if return_intermediates:
            return z, intermediates
        return z