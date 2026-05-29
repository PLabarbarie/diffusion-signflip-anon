# Diffusion Sign-Flip Anonymization

This repository contains the code for the paper "Secure and reversible face anonymization based on
a diffusion model with face mask guidance" (*to be published*) by Pol Labarbarie, Vincent Itier and William Puech. The paper proposes a novel anonymization method that applies a secret sign-flip key to the latent space of diffusion models The method encodes face images into diffusion latents, applies a sign-flip key on selected latent regions, decodes the anonymized image, and evaluates whether the process reduces identity leakage while preserving image utility.

The code supports unconditional FFHQ diffusion models, ControlNet-conditioned variants, optional DDIM inversion, and reconstruction with the same latent sign-flip key. It also includes face-recognition and utility metrics used to evaluate anonymization, reconstruction, robustness, diversity, and AI-detection experiments.

## Repository Layout

```text
.
|-- anonymization/
|   |-- diffusion.py          # Diffusion and ControlNet model wrapper
|   |-- flip_anonymizer.py    # Latent sign-flip anonymization and reconstruction
|   |-- controlnet.py         # Single-condition ControlNet implementation
|   `-- controlnet_v2.py      # Multi-ControlNet wrapper
|-- config/
|   |-- config_main.py        # Main experiment configuration
|   `-- config_controlnet.py  # ControlNet training configuration
|-- data/
|   |-- dataset.py            # Dataset loaders for LFW, FFHQ, CelebA, CelebA-HQ
|   `-- transforms.py         # Image preprocessing and perturbations
|-- fr_system/
|   |-- fr_models.py          # Face-recognition backbones
|   |-- fr_pipeline.py        # Face detection plus recognition pipeline
|   |-- face_detector.py      # MTCNN, RetinaFace, and landmark detectors
|   `-- bisenet.py            # Face parsing model
|-- keys/                     # Precomputed anonymization keys for reproducibility
|-- notebooks/                # Exploration and analysis notebooks
|-- results/                  # Experiment outputs
|-- utils/                    # Image and metric utilities
|-- weights/                  # Local model checkpoints and trained weights
|-- main.py                   # Main anonymization and evaluation entry point
|-- train_controlnet.py       # ControlNet training script
`-- requirements.txt
```

## Configuration

Experiment settings are defined in Python config files under `config/`.

`config/config_main.py` controls the main anonymization/evaluation pipeline:

- `config.DATASET`: dataset to evaluate, such as `LFW`, `CelebA_HQ`, or `FFHQ`.
- `config.diffusion_name`: diffusion backend, such as `ffhq`, `controlnet_segmentation_masks`, `controlnet_landmarks_5`, or `multi_controlnet`.
- `config.guidance`, `config.controlnet`, and `config.ddim_inversion`: enable conditioning and inversion variants.
- `config.evaluation`: experiment mode, including `standard`, `diversity`, `diff_cryptanalysis`, `jpeg_compression`, `test_noise`, or `ai_detection`.
- `config.fr_models`: face-recognition models used for identity evaluation.
- `config.root`, `config.data_root`, `config.weights_root`, `config.models_root`, and `config.dataset_roots`: project, data, checkpoint, and dataset paths.

`config/config_controlnet.py` controls ControlNet training:

- `config.controlnet_conditioning`: conditioning target, such as `segmentation_masks`, `landmarks_5`, or `landmarks_26`.
- `config.dataset_roots.FFHQ` and `config.dataset_roots.FFHQ_latents`: image and conditioning/latent roots.
- `config.save_path`: output directory for trained ControlNet checkpoints.
- `config.controlnet_epochs`, `config.controlnet_lr`, and `config.controlnet_save_interval`: training hyperparameters.

The lower-level modules do not import these config files directly. Paths are passed from the entry-point scripts into dataset, diffusion, and face-recognition components.

## Expected Data and Weights

The default path layout is project-relative:

```text
<repo root>/
|-- weights/
|   |-- models/
|   |   |-- ffhq-diffusers/
|   |   |-- bisenet/resnet34.pt
|   |   `-- ms1mv3_arcface_r50_fp16/backbone.pth
|   `-- ffhq_controlnet/
|       |-- segmentation_masks/controlnet_epoch_15.pth
|       |-- landmarks_5/controlnet_epoch_15.pth
|       `-- landmarks_26/controlnet_epoch_15.pth
`-- keys/
    |-- keys_CelebA_HQ.pt
    `-- sub_keys_diversity_<n>.pt
```

Datasets are expected under `config.data_root`, which currently defaults to a sibling `data/` directory outside the repository root:

```text
<code root>/data/
|-- FlickrFace/
|-- LFW/
`-- CelebAMask-HQ/CelebA-HQ-img/
```

Adjust the paths in the config files if your local dataset or checkpoint layout differs.

These datasets can be downloaded from their original sources:
- FFHQ: [FlickrFace](https://github.com/nvlabs/ffhq-dataset)
- LFW: [LFW](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
- CelebA-HQ: [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)

To download the face-recognition models, check in the fr_system directory the corresponding directory for each model. A readme file from the original repository is included with instructions and links to the checkpoints. 

To download the bisenet face parsing model, check the following link:
- BiseNet: [BiseNet](https://yakhyo.github.io/face-parsing/)

For our anonymization model, we provide pretrained ControlNet checkpoints under the following huggingface repository: [FFHQ ControlNet for Diffusion Sign-Flip Anonymization](https://huggingface.co/PolLabarbarie/diffusion-signflip-anon-face). These weights were trained on FFHQ for 15 epochs with the script in `train_controlnet.py`. You can also train your own ControlNets with different conditioning or datasets by following the instructions in that script. 

The different secret-keys used in the paper experiments are provided under the `keys/` directory for reproducibility. The `keys_CelebA_HQ.pt` file contains the keys used for the CelebA-HQ experiments, while the `sub_keys_diversity_<n>.pt` files contain subsets of keys used for the diversity experiments. These can be downloaded also from the Hugging Face repository linked above.

The code uses `local_files_only=True` when loading these models, so you need to ensure the checkpoints are present in the expected paths before running experiments.


## Running Experiments

Edit `config/config_main.py` for the dataset, model variant, evaluation mode, and face-recognition backbones. Then run:

```bash
python main.py --n 0
```

The `--n` argument selects the dataset partition or key subset used by the current experiment. This argument is used for the `diversity`and `diff_cryptanalysis` modes. Outputs are written under:

```text
results/<DATASET>/<config.save_path_suffix>/
```

The saved result file contains the secret-key used, masks, utility metrics, and face-recognition templates for clean, anonymized, and reconstructed images. The exact contents depend on the evaluation mode. For example, the `diversity` mode saves multiple anonymized variants per image, while the `diff_cryptanalysis` mode saves results for different keys. 

Once the results are saved, you can run results.ipynb to load and analyze the outputs, generate plots, and compute additional metrics.

## Training ControlNet

Edit `config/config_controlnet.py` to select the conditioning type and training hyperparameters, then run:

```bash
python train_controlnet.py
```

Checkpoints and training diagnostics are saved under `config.save_path`, for example:

```text
weights/ffhq_controlnet/segmentation_masks/
```

## Citation

If you use this repository, please cite the associated paper once available.

```bibtex
@article{diffusion_signflip_anonymization,
  title   = {Secure and reversible face anonymization based on
a diffusion model with face mask guidance},
  author  = {Pol Labarbarie and Vincent Itier and William Puech},
  journal = {},
  year    = {2026}
}
```

