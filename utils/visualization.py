from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def save_image(image_tensor, path):
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    pil_image.save(path)

def display_images(images, titles=None, cols=3):
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i in range(n_images):
        axes[i].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axes[i].axis('off')
        if titles is not None:
            axes[i].set_title(titles[i])

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_histogram(data, bins=50, xlabel='Value', title='Histogram'):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, alpha=0.7)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def save_visualization(images, output_dir, prefix=''):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, img in enumerate(images):
        save_image(img, os.path.join(output_dir, f"{prefix}image_{i}.png"))