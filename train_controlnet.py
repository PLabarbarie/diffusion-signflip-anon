import torch
from anonymization.diffusion import DiffusionModel
from anonymization.controlnet import ControlNet
import os
from diffusers import DDPMScheduler
from config.config_controlnet import config
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data import DataLoader
from data.dataset import load_dataset
from data.transforms import get_base_transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def debug(hint, latents, diffusion_model, save_path, epoch_idx, batch_idx):
    j = latents.shape[0]//2
    im = diffusion_model.decode_latent(latents[j].to("cuda")).cpu()[0].detach()
    plt.imshow((im*0.5 + 0.5).permute(1,2,0).cpu())
    blank_spots = (hint[j,0] != 0).nonzero(as_tuple=True)
    for y, x in zip(*blank_spots):
        plt.plot(x.item(), y.item(), 'ro', alpha=0.5)
    plt.savefig(os.path.join(save_path, f'train_latent_hint_epoch{epoch_idx}_batch{batch_idx}_sample{j}.png'))
    plt.close()

def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET = config.dataset_roots.DATASET

    batch_size = config['batch_size']

    # Load dataset (synthetic or real)
    tfms = get_base_transforms(DATASET)
    dataset = load_dataset(DATASET, transform=tfms, split="all", multiple_images=True, controlnet=True,
                           data_root=config.data_root, dataset_roots=config.dataset_roots,
                           controlnet_conditioning_dir = config.dataset_roots.FFHQ_latents) # or "LFW"
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Load the diffusion model and anonymizer
    diffusion_name = "ffhq"
    diffusion_model = DiffusionModel(name=diffusion_name,
                                    inverse_scheduler = "hugging_face", torch_device=device,
                                    models_root=config.models_root, weights_root=config.weights_root)
    diffusion_model.to_device(device)

    scheduler = DDPMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler",
                                                            local_files_only=True)

    controlnet = ControlNet(model_config=diffusion_model.unet.config,
                            model_ckpt=os.path.join(config.models_root, "ffhq-diffusers"),
                            hint_channels=3,
                            down_sample_factor=4,
                            device=device)

    controlnet.to_device(device)
    controlnet.train()

    # Create output directories
    save_path = config.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # # Load checkpoint if found
    # if os.path.exists(os.path.join(train_config['task_name'], train_config['controlnet_ckpt_name'])):
    #     print('Loading checkpoint as found one')
    #     model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
    #                                                     train_config['controlnet_ckpt_name']),
    #                                         map_location=device))

    # Specify training parameters
    num_epochs = config['controlnet_epochs']
    optimizer = Adam(controlnet.get_params(), lr=config['controlnet_lr'])
    lr_scheduler = MultiStepLR(optimizer, milestones=config['controlnet_lr_steps'], gamma=0.1)
    criterion = torch.nn.MSELoss()

    step_count = 0
    count = 0
    losses_over_epochs = []
    for epoch_idx in range(num_epochs):
        losses = []
        for i, (latents, hint, name) in enumerate(tqdm(dataset)):
            #im between -1 and 1, hint between 0 and 1
            optimizer.zero_grad()
            if i == 0 and epoch_idx % 5 == 0:
                debug(hint, latents, diffusion_model, save_path, epoch_idx, i)
            latents = latents.squeeze(1).to(device)
            hint = hint.to(device)

            noise = torch.randn_like(latents).to(device)
            t = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],)).to(device)
            noisy_latents = scheduler.add_noise(latents, noise, t)
            noise_pred = controlnet(noisy_latents, t, hint)
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            step_count += 1


        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))
        lr_scheduler.step()
        losses_over_epochs.append(np.mean(losses))
        if (epoch_idx + 1) % config['controlnet_save_interval'] == 0:
            torch.save(controlnet.state_dict(), os.path.join(save_path,
                                                        "controlnet_epoch_{}.pth".format(epoch_idx + 1)))

    print('Done Training ...')

    plt.plot(losses_over_epochs)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('ControlNet Training Loss over Epochs')
    plt.savefig(os.path.join(save_path, 'controlnet_training_loss.png'))
    plt.close()

if __name__ == "__main__":
    train()
