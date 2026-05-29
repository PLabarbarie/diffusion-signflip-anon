from anonymization.diffusion import DiffusionModel
from fr_system.bisenet import BiSeNet
from fr_system.mtcnn import MTCNN
import torch
import numpy as np

class FlipAnonymizer:
    def __init__(self, model: DiffusionModel, face_parser: BiSeNet = None, face_detector: MTCNN = None,
    guidance: bool = True, guidance_scale : float = 1, controlnet : str = None,
    save_intermediate_landmarks : bool = True, saving_path = "trash"):
        self.DM = model
        self.face_parser = face_parser
        self.face_detector = face_detector
        self.guidance = guidance
        self.guidance_scale = guidance_scale

        self.controlnet = controlnet

        self.save_intermediate_landmarks = save_intermediate_landmarks
        self.saving_path = saving_path

    def generate_keys(self, shape: torch.Size, p: float = 0.5, seeds: torch.Tensor = None, device: torch.device = None) -> torch.Tensor:
        """
        Generate anonymization keys for a given shape.
        
        Args:
            shape: torch.Size - Shape of the tensor to generate keys for (B, C, H, W)
            p: float - Probability for Bernoulli distribution (default: 0.5)
            seeds: torch.Tensor or None - Seeds for each example in batch (B,) or single int
            device: torch.device - Device to create keys on
        
        Returns:
            torch.Tensor: Keys tensor of shape (B, C, H, W) with values ±1
        """
        batch_size = shape[0]
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if seeds is None:
            # Use random seeds if none provided
            seeds = torch.randint(0, 2**128, (batch_size,), device=device)
        elif isinstance(seeds, (int, float)):
            # If single seed provided, use it for all examples
            seeds = torch.full((batch_size,), int(seeds), device=device)
        elif len(seeds.shape) == 0:
            # If scalar tensor, expand to batch
            seeds = seeds.unsqueeze(0).expand(batch_size).to(device)
        else:
            # Ensure seeds tensor is on correct device
            seeds = seeds.to(device)
        
        # Generate keys for each example in batch with individual seeds
        keys = torch.zeros(shape, device=device)
        for i in range(batch_size):
            gen = torch.Generator(device=device).manual_seed(int(seeds[i]))
            keys[i] = (2 * torch.bernoulli(p * torch.ones(shape[1:], device=device), generator=gen) - 1)
        
        return keys
    
    def generate_keys_masked(self, mask: torch.Tensor = None, p: float = 0.5, seeds: torch.Tensor = None, device: torch.device = None) -> torch.Tensor:
        """
        Generate anonymization keys for a given shape.
        
        Args:
            shape: torch.Size - Shape of the tensor to generate keys for (B, C, H, W)
            p: float - Probability for Bernoulli distribution (default: 0.5)
            seeds: torch.Tensor or None - Seeds for each example in batch (B,) or single int
            device: torch.device - Device to create keys on
        
        Returns:
            torch.Tensor: Keys tensor of shape (B, C, H, W) with values ±1
        """
        batch_size = mask.shape[0]
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if seeds is None:
            # Use random seeds if none provided
            seeds = torch.randint(0, 2**128, (batch_size,), device=device)
        elif isinstance(seeds, (int, float)):
            # If single seed provided, use it for all examples
            seeds = torch.full((batch_size,), int(seeds), device=device)
        elif len(seeds.shape) == 0:
            # If scalar tensor, expand to batch
            seeds = seeds.unsqueeze(0).expand(batch_size).to(device)
        else:
            # Ensure seeds tensor is on correct device
            seeds = seeds.to(device)
        
        # Generate keys for each example in batch with individual seeds
        keys = torch.zeros_like(mask, device=device)
        for i in range(batch_size):
            gen = torch.Generator(device=device).manual_seed(int(seeds[i]))
            keys[i] = mask[i] * (2 * torch.bernoulli(p * torch.ones_like(mask[i]), generator=gen) - 1)
        
        return keys

    def apply_anonymization(self, latent: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Apply anonymization using pre-generated keys.
        
        Args:
            latent: (B, C, H, W) - Input latent representation
            keys: (B, C, H, W) - Pre-generated anonymization keys
        
        Returns:
            torch.Tensor: Anonymized latent representation
        """
        return latent * keys

    def deanonymize(self, anonymized_latent: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Reverse the anonymization using the stored keys.
        
        Args:
            anonymized_latent: (B, 4, H/8, W/8) - Anonymized latent
            keys: (B, 4, H/8, W/8) - Keys used for anonymization
        
        Returns:
            torch.Tensor: Original latent representation
        """
        return anonymized_latent * keys  # Since keys are ±1, multiplying twice gives original

    def anonymization_pipeline(self, images: torch.Tensor, keys: torch.Tensor, p: float = 0.5, seeds: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full anonymization pipeline for a batch of images.
        
        Args:
            images: (B, 3, H, W) - Input images
            p: float - Probability for Bernoulli distribution
            seeds: torch.Tensor or None - Seeds for each example in batch
        
        Returns:
            tuple: (anonymized_images, keys) - Anonymized images and keys for reversal
        """
        # Encode images to latent space
        with torch.no_grad():
            mask = self.face_parser.get_face_mask(images)
            mask = (mask > 0)*1.0

            if self.controlnet:
                hint = torch.zeros_like(images)
                landmarks = self.face_detector.extract_landmarks(images)
                for j in range(images.shape[0]):
                    for k in range(landmarks[j].shape[0]):
                        x, y = int(landmarks[j][k][0]),  int(landmarks[j][k][1])
                        hint[j, :, y-1:y+2, x-1:x+2] = 1.0  # Mark landmark on a blank image



            # # Visualize the first image with landmarks
            # import matplotlib.pyplot as plt

            # # Convert tensor to numpy for visualization
            # img = images[0].permute(1, 2, 0).cpu().numpy()
            # # Normalize to 0-1 range if needed
            # if img.max() > 1.0:
            #     img = img / 255.0

            # plt.figure(figsize=(10, 10))
            # plt.imshow(img)

            # # Plot landmarks if available
            # if landmarks is not None and len(landmarks) > 0:
            #     # Get landmarks for the first image
            #     face_landmarks = landmarks[0,0]
            #     if face_landmarks is not None:
            #         # Plot each landmark point
            #         for landmark in face_landmarks:
            #             plt.plot(landmark[0], landmark[1], 'ro', markersize=6)

            # plt.title('Image with Facial Landmarks')
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()
            # exit()

            z0 = self.DM.encode_image(images)
            if self.controlnet:
                zT, z_list = self.DM.ddim_inversion(z0, hint, num_inference_steps=50, return_intermediates=True)
            else : 
                zT, z_list = self.DM.ddim_inversion(z0, hint=None, num_inference_steps=50, return_intermediates=True)

            # z0_tilde, _ = self.DM.backward_diffusion2(z_list, mask, num_inference_steps=30, eta=0.0)
            # zT, z_list = self.DM.ddim_inversion(z0_tilde, num_inference_steps=30, return_intermediates=True)


            if keys is None: 
                keys = self.generate_keys(zT.size(), p, seeds)
                keys = mask * keys
            # import matplotlib.pyplot as plt
            # plt.imshow(keys[0,0].cpu().numpy())
            # plt.colorbar()
            # plt.show()
            zT_ano = self.apply_anonymization(zT, keys) * mask + zT * (1 - mask)
            z_list[-1] = zT_ano.clone()
            # z0_ano, _ = self.DM.backward_diffusion_repaint(z_list, mask, num_inference_steps=50, eta=0.0, jump_length=10, jump_n_sample=2)
            # z0_ano, _ = self.DM.backward_diffusion_repaint(z_list, mask, num_inference_steps=30, eta=0.0, jump_length=10, jump_n_sample=10)

        if self.guidance:
            landmarks = self.face_detector.extract_landmarks(images)
            z0_ano, _ = self.DM.conditional_backward_diffusion(z_list, landmarks, self.face_detector, self.guidance_scale, mask,
            num_inference_steps=50, eta=0.0,
            save_intermediate_landmarks=self.save_intermediate_landmarks, save_path= self.saving_path)
        elif self.controlnet :
            z0_ano, _ = self.DM.masked_backward_diffusion(z_list, mask, hint, num_inference_steps=50, eta=0.0)
        else : 
            z0_ano, _ = self.DM.masked_backward_diffusion(z_list, mask, num_inference_steps=50, eta=0.0)


        # print(((z_list[0] - z0) * (1-mask)).abs().sum())
        # print(((z0_ano - z0) * (1-mask)).abs().sum())

        # non_masked_diff = ((z0_ano - z0) * (1-mask)).abs().sum()
        # print(f"Non-masked region difference: {non_masked_diff}")
        # exit()
        with torch.no_grad():
            x_ano = self.DM.decode_latent(z0_ano)
        return x_ano, keys, mask


    @torch.no_grad()
    def de_anonymization_pipeline(self, ano_images: torch.Tensor, keys: torch.Tensor, mask ) -> torch.Tensor:
        """
        Full de-anonymization pipeline for a batch of anonymized images.

        Args:
            ano_images: (B, 3, H, W) - Anonymized images
            keys: (B, 3, H/8, W/8) - Keys used for anonymization
            mask: (B, 3, H/8, W/8) - Mask indicating which areas are anonymized
        Returns:
            torch.Tensor: Original images
        """
        # Encode images to latent space
        if mask is None:
            mask = self.face_parser.get_face_mask(ano_images)
            mask = (mask > 0)*1.0

        if self.controlnet:
            hint = torch.zeros_like(ano_images)
            landmarks = self.face_detector.extract_landmarks(ano_images)
            for j in range(ano_images.shape[0]):
                for k in range(landmarks[j].shape[0]):
                    x, y = int(landmarks[j][k][0]),  int(landmarks[j][k][1])
                    hint[j, :, y-1:y+2, x-1:x+2] = 1.0  # Mark landmark on a blank image

        z0_ano = self.DM.encode_image(ano_images)
        if self.controlnet:
            zT_ano, z_ano_list = self.DM.ddim_inversion(z0_ano, hint, num_inference_steps=50, return_intermediates=True)
        else:
            zT_ano, z_ano_list = self.DM.ddim_inversion(z0_ano, hint=None, num_inference_steps=50, return_intermediates=True)
        zT = self.deanonymize(zT_ano, keys) * mask + zT_ano * (1 - mask)
        z_ano_list[-1] = zT.clone()
        # z0, _ = self.DM.backward_diffusion_repaint(z_ano_list, mask, num_inference_steps=50, eta=0.0, jump_length=10, jump_n_sample=1)
        if self.controlnet :
            z0, _ = self.DM.masked_backward_diffusion(z_ano_list, mask, hint, num_inference_steps=50, eta=0.0)
        else : 
            z0, _ = self.DM.masked_backward_diffusion(z_ano_list, mask, num_inference_steps=50, eta=0.0)

        z0, _ = self.DM.masked_backward_diffusion(z_ano_list, mask, num_inference_steps=50, eta=0.0)
    
        # z0 = self.DM.backward_diffusion(zT, num_inference_steps=50, eta=0.0)
        x = self.DM.decode_latent(z0)
        return x


# Example usage:
# anonymizer = FlipAnonymizer(diffusion_model)

# Option 1: Different seed for each example
# seeds = torch.tensor([42, 123, 456, 789])  # 4 different seeds for batch of 4
# anonymized, keys = anonymizer.anonymize(latents, p=0.5, seeds=seeds)

# Option 2: Same seed for all examples
# anonymized, keys = anonymizer.anonymize(latents, p=0.5, seeds=42)

# Option 3: Random seeds (default)
# anonymized, keys = anonymizer.anonymize(latents, p=0.5)