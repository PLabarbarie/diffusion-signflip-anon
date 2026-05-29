import torch
import torch.nn as nn
from diffusers import UNet2DModel
from typing import List, Optional, Union
from .controlnet import ControlNet


class MultiControlNet(nn.Module):
    r"""
    Multiple ControlNet Module that combines multiple control signals.
    Uses pre-trained ControlNet instances from controlnet.py
    """
    def __init__(self, 
                 controlnets: List[ControlNet],
                 control_scales: Optional[List[float]] = None):
        """
        Args:
            controlnets: List of ControlNet instances from controlnet.py
            control_scales: Optional list of scaling factors for each ControlNet (default: 1.0 for all)
        """
        super().__init__()
        
        if not controlnets:
            raise ValueError("At least one ControlNet must be provided")
        
        self.controlnets = nn.ModuleList(controlnets)
        
        if control_scales is None:
            control_scales = [1.0] * len(controlnets)
        elif len(control_scales) != len(controlnets):
            raise ValueError(f"Number of control_scales ({len(control_scales)}) must match number of controlnets ({len(controlnets)})")
        
        self.control_scales = control_scales
        
        # Use the trained_unet from the first controlnet as the shared decoder
        self.trained_unet = self.controlnets[0].trained_unet
        self.time_proj = self.trained_unet.time_proj
        self.time_embedding = self.trained_unet.time_embedding
        self.dtype = self.trained_unet.dtype
        
        # Store whether the model is locked
        self.model_locked = self.controlnets[0].model_locked

    def to_device(self, device):
        """Move all controlnets to device"""
        for controlnet in self.controlnets:
            controlnet.to_device(device)

    def get_params(self):
        """Get all trainable parameters from all controlnets"""
        params = []
        for controlnet in self.controlnets:
            params += controlnet.get_params()
        # Remove duplicates (shared decoder parameters)
        return list(set(params))

    def forward(self, x, timestep, hints: Union[torch.Tensor, List[torch.Tensor]]):
        """
        Args:
            x: Noisy sample (B, C, H, W)
            timestep: Timestep (B,) or int
            hints: Either a single hint tensor or list of hint tensors (one per ControlNet)
        """
        # Handle single hint vs multiple hints
        if isinstance(hints, torch.Tensor):
            if len(self.controlnets) == 1:
                hints = [hints]
            else:
                raise ValueError(f"Expected {len(self.controlnets)} hints but got a single tensor")
        
        if len(hints) != len(self.controlnets):
            raise ValueError(f"Number of hints ({len(hints)}) must match number of ControlNets ({len(self.controlnets)})")

        # 1. Process timestep
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        timesteps = timesteps * torch.ones(x.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2. Process each controlnet branch to get control signals
        all_control_down_outs = []
        all_control_mid_outs = []
        
        for controlnet, hint, scale in zip(self.controlnets, hints, self.control_scales):
            # Process hint through hint block
            hint_emb = controlnet.control_copy_unet_hint_block(hint)
            
            # Forward pass through control copy UNet
            control_sample = controlnet.control_copy_unet.conv_in(x.clone())
            control_sample = control_sample + hint_emb
            
            # Down blocks
            control_copy_unet_down_outs = (control_sample,)
            for i, down_block in enumerate(controlnet.control_copy_unet.down_blocks):
                control_sample, control_block_skips = down_block(control_sample, emb)
                control_block_skips = list(control_block_skips)
                for j in range(len(down_block.resnets)):
                    control_block_skips[j] = controlnet.control_copy_unet_down_zero_convs[
                        i * len(down_block.resnets) + j
                    ](control_block_skips[j])
                control_block_skips = tuple(control_block_skips)
                control_copy_unet_down_outs += control_block_skips

            # Mid block
            control_sample = controlnet.control_copy_unet.mid_block(control_sample, emb)
            control_mid_out = controlnet.control_copy_unet_mid_zero_conv(control_sample)
            
            # Apply scaling
            scaled_control_down_outs = tuple(scale * out for out in control_copy_unet_down_outs)
            scaled_control_mid_out = scale * control_mid_out
            
            all_control_down_outs.append(scaled_control_down_outs)
            all_control_mid_outs.append(scaled_control_mid_out)

        # 3. Combine control signals (sum)
        num_down_outputs = len(all_control_down_outs[0])
        combined_control_down_outs = []
        
        for i in range(num_down_outputs):
            combined = sum(control_outs[i] for control_outs in all_control_down_outs)
            combined_control_down_outs.append(combined)
        
        combined_control_down_outs = tuple(combined_control_down_outs)
        combined_control_mid_out = sum(all_control_mid_outs)

        # 4. Forward through trained UNet (same as ControlNet in controlnet.py)
        with torch.no_grad():
            # Pre-process
            skip_sample = x.clone()
            sample = self.trained_unet.conv_in(x.clone())

            # Down blocks
            down_block_res_samples = (sample,)
            for downsample_block in self.trained_unet.down_blocks:
                if hasattr(downsample_block, "skip_conv"):
                    sample, res_samples, skip_sample = downsample_block(
                        hidden_states=sample, temb=emb, skip_sample=skip_sample
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                down_block_res_samples += res_samples

            # Mid block
            sample = self.trained_unet.mid_block(sample, emb)

        # Add combined control signal to mid block
        train_sample = sample + combined_control_mid_out

        # 5. Up blocks
        skip_sample = None
        for upsample_block in self.trained_unet.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            res_control_samples = combined_control_down_outs[-len(upsample_block.resnets):]
            combined_control_down_outs = combined_control_down_outs[:-len(upsample_block.resnets)]

            # Combine res_samples with combined control signals
            res_samples = [t + c for t, c in zip(res_samples, res_control_samples)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(train_sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(train_sample, res_samples, emb)
            train_sample = sample

        # 6. Output layers
        train_sample = self.trained_unet.conv_norm_out(train_sample)
        train_sample = self.trained_unet.conv_act(train_sample)
        train_sample = self.trained_unet.conv_out(train_sample)
        
        return train_sample