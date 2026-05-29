import torch
import torch.nn as nn
from diffusers import UNet2DModel


def make_zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class ControlNet(nn.Module):
    r"""
    Control Net Module for Unconditional Diffusion Models using diffusers format
    """
    def __init__(self, model_config,
                 model_locked=True,
                 model_ckpt=None,
                 hint_channels=3,
                 down_sample_factor=32,
                 device=None):
        super().__init__()
        # Trained UNet from diffusers
        self.model_locked = model_locked
        
        if model_ckpt is not None:
            print('Loading Trained Diffusion Model')
            self.trained_unet = UNet2DModel.from_pretrained(model_ckpt, subfolder="unet", local_files_only=True)
        else:
            self.trained_unet = UNet2DModel(**model_config)
        
        if device is not None:
            self.trained_unet = self.trained_unet.to(device)

        # ControlNet Copy of Trained UNet (encoder only)
        print('Loading Control Diffusion Model')
        if model_ckpt is not None:
            self.control_copy_unet = UNet2DModel.from_pretrained(model_ckpt, subfolder="unet", local_files_only=True)
        else:
            self.control_copy_unet = UNet2DModel(**model_config)
        
        if device is not None:
            self.control_copy_unet = self.control_copy_unet.to(device)

        # Time embedding

        self.time_proj = self.trained_unet.time_proj
        self.time_embedding = self.trained_unet.time_embedding
        self.dtype = self.trained_unet.dtype
        
        # Get channel dimensions from the UNet config
        block_out_channels = self.trained_unet.config.block_out_channels
        self.block_out_channels = block_out_channels
        layers_per_block = self.trained_unet.config.layers_per_block
        self.layers_per_block = layers_per_block
        
        # Hint Block for ControlNet
        # Stack of Conv activation and zero convolution at the end
        base_hint_channel = 16
        curr_down_sample_factor = down_sample_factor
        hint_layers = [nn.Sequential(
                nn.Conv2d(hint_channels,
                          base_hint_channel,
                          kernel_size=3,
                          padding=(1, 1)),
                nn.SiLU())]
        while curr_down_sample_factor > 1:
            hint_layers.append(nn.Sequential(
                nn.Conv2d(base_hint_channel,
                          base_hint_channel*2,
                          kernel_size=3,
                          padding=(1, 1),
                          stride=2),
                nn.SiLU(),
                nn.Conv2d(base_hint_channel*2,
                          base_hint_channel*2,
                          kernel_size=3,
                          padding=(1, 1))
            ))
            base_hint_channel = base_hint_channel * 2
            curr_down_sample_factor = curr_down_sample_factor / 2
        hint_layers.append(nn.Sequential(
            nn.Conv2d(base_hint_channel,
                      self.trained_unet.block_out_channels[0],
                      kernel_size=3,
                      padding=(1, 1)),
            nn.SiLU(),
            make_zero_module(nn.Conv2d(self.trained_unet.block_out_channels[0],
                                       self.trained_unet.block_out_channels[0],
                                       kernel_size=1,
                                       padding=0))
        ))
        self.control_copy_unet_hint_block = nn.Sequential(*hint_layers)

        # Zero Convolution Module for Downblocks
        self.control_copy_unet_down_zero_convs = []
        for i, block in enumerate(self.trained_unet.down_blocks):
            for _ in range(len(block.resnets)):
                self.control_copy_unet_down_zero_convs.append(
                    make_zero_module(
                        nn.Conv2d(block_out_channels[i],
                                  block_out_channels[i],
                                  kernel_size=1,
                                  padding=0)
                    )
                )
        self.control_copy_unet_down_zero_convs = nn.ModuleList(self.control_copy_unet_down_zero_convs)

            
        # self.control_copy_unet_down_zero_convs = nn.ModuleList([
        #     make_zero_module(nn.Conv2d(block_out_channels[i],
        #                                block_out_channels[i],
        #                                kernel_size=1,
        #                                padding=0))
        #     for i in range(len(block_out_channels))
        # ])

        # Zero Convolution Module for MidBlock
        mid_channels = block_out_channels[-1]
        self.control_copy_unet_mid_zero_conv = make_zero_module(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0)
        )

        if device is not None:
            self.control_copy_unet_hint_block = self.control_copy_unet_hint_block.to(device)
            self.control_copy_unet_down_zero_convs = self.control_copy_unet_down_zero_convs.to(device)
            self.control_copy_unet_mid_zero_conv = self.control_copy_unet_mid_zero_conv.to(device)

    def to_device(self, device):
        self.trained_unet = self.trained_unet.to(device)
        self.control_copy_unet = self.control_copy_unet.to(device)
        self.control_copy_unet_hint_block = self.control_copy_unet_hint_block.to(device)
        self.control_copy_unet_down_zero_convs = self.control_copy_unet_down_zero_convs.to(device)
        self.control_copy_unet_mid_zero_conv = self.control_copy_unet_mid_zero_conv.to(device)

    def get_params(self):
        # Add all ControlNet parameters
        params = list(self.control_copy_unet.parameters())
        params += list(self.control_copy_unet_hint_block.parameters())
        params += list(self.control_copy_unet_down_zero_convs.parameters())
        params += list(self.control_copy_unet_mid_zero_conv.parameters())

        # If we desire to not have the decoder layers locked
        if not self.model_locked:
            params += list(self.trained_unet.up_blocks.parameters())
            params += list(self.trained_unet.conv_norm_out.parameters())
            params += list(self.trained_unet.conv_out.parameters())
        return params

    def forward(self, x, timestep, hint):
        """
        Args:
            x: Noisy sample (B, C, H, W)
            t: Timestep (B,) or int
            hint: Control hint image (B, hint_channels, H, W)
        """

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(x.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)


        # Process hint through hint block
        hint_emb = self.control_copy_unet_hint_block(hint)

        with torch.no_grad():

            # 2. pre-process
            skip_sample = x.clone()
            sample = self.trained_unet.conv_in(x.clone())

            # # 3. down
            down_block_res_samples = (sample,)
            for downsample_block in self.trained_unet.down_blocks:
                if hasattr(downsample_block, "skip_conv"):
                    sample, res_samples, skip_sample = downsample_block(
                        hidden_states=sample, temb=emb, skip_sample=skip_sample
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                down_block_res_samples += res_samples

            # # 4. mid
            sample = self.trained_unet.mid_block(sample, emb)
        
        # print(len(down_block_res_samples))
        # print(down_block_res_samples)

        # print("____________________")

        # Forward pass through control copy UNet
        skip_sample = hint.clone()
        control_sample = self.control_copy_unet.conv_in(x.clone())
        control_sample = control_sample + hint_emb
        
        # Down blocks
        control_copy_unet_down_outs = (control_sample,)
        for i, down_block in enumerate(self.control_copy_unet.down_blocks):
            # Apply zero convolution and store
            control_sample, control_block_skips = down_block(control_sample, emb)
            control_block_skips = list(control_block_skips)
            for j in range(len(down_block.resnets)):
                control_block_skips[j] = self.control_copy_unet_down_zero_convs[i * len(down_block.resnets) + j](
                    control_block_skips[j]
                )
            control_block_skips = tuple(control_block_skips)
            control_copy_unet_down_outs += control_block_skips

        # Mid block
        control_sample = self.control_copy_unet.mid_block(control_sample, emb)
        # Add mid block output with zero convolution
        train_sample = sample + self.control_copy_unet_mid_zero_conv(control_sample)
        

        # print(len(down_block_res_samples), len(control_copy_unet_down_outs))
        # print("____________________")


        # # 5. up
        skip_sample = None
        for upsample_block in self.trained_unet.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            res_control_samples = control_copy_unet_down_outs[-len(upsample_block.resnets) :]
            control_copy_unet_down_outs = control_copy_unet_down_outs[: -len(upsample_block.resnets)]

            # Combine res_samples with res_control_samples
            res_samples = [t + c for t, c in zip(res_samples, res_control_samples)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(train_sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(train_sample, res_samples, emb)
            train_sample = sample

        
        
        # Output layers
        train_sample = self.trained_unet.conv_norm_out(train_sample)
        train_sample = self.trained_unet.conv_act(train_sample)
        train_sample = self.trained_unet.conv_out(train_sample)









        # 2. pre-process
        # skip_sample = sample
        # sample = self.conv_in(sample)

        # # 3. down
        # down_block_res_samples = (sample,)
        # for downsample_block in self.down_blocks:
        #     if hasattr(downsample_block, "skip_conv"):
        #         sample, res_samples, skip_sample = downsample_block(
        #             hidden_states=sample, temb=emb, skip_sample=skip_sample
        #         )
        #     else:
        #         sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        #     down_block_res_samples += res_samples

        # # 4. mid
        # if self.mid_block is not None:
        #     sample = self.mid_block(sample, emb)

        # # 5. up
        # skip_sample = None
        # for upsample_block in self.up_blocks:
        #     res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        #     down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        #     if hasattr(upsample_block, "skip_conv"):
        #         sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
        #     else:
        #         sample = upsample_block(sample, res_samples, emb)

        # # 6. post-process
        # sample = self.conv_norm_out(sample)
        # sample = self.conv_act(sample)
        # sample = self.conv_out(sample)

        # if skip_sample is not None:
        #     sample += skip_sample




        
        return train_sample