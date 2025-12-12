# DeepFit Training

A deep learning framework for virtual try-on using Stable Diffusion 1.5 Inpainting with ControlNet.

## Overview

DeepFit is a virtual try-on system that leverages diffusion models to realistically transfer garments onto person images. The framework uses a modified Stable Diffusion 1.5 Inpainting model combined with ControlNet for precise garment placement and pose-aware synthesis.

## Features

- **ControlNet Integration**: Uses pose and structural conditioning for accurate garment fitting
- **Multi-modal Input**: Supports garment images, person images, masks, and pose information
- **Distillation Support**: Includes LADD consistency distillation for faster inference
- **Distributed Training**: Built with Accelerate for multi-GPU training
- **Precomputed Embeddings**: Efficient training with precomputed text embeddings
- **W&B Integration**: Built-in Weights & Biases logging for experiment tracking

## Architecture

| Component | Model |
|-----------|-------|
| Base Model | `stable-diffusion-v1-5/stable-diffusion-inpainting` |
| ControlNet | `lllyasviel/control_v11p_sd15_inpaint` |
| VAE | SD1.5 VAE (4-channel latents) |
| Text Encoder | CLIP Text Encoder |
| Scheduler | DDPM Scheduler |

## Project Structure

```
Deepfit-Training/
├── model.py                    # DeepFit model architecture
├── train.py                    # Main training script
├── inference.py                # Inference script
├── utils.py                    # Utility functions
├── virtual_try_on_dataloader.py # Data loading utilities
├── precompute_captions.py      # Precompute text embeddings
├── distillation/               # Distillation training
│   ├── distill_train.py
│   ├── distill_infer.py
│   └── distill_utils.py
└── experiments/                # Experimental variants
    ├── experiment1/
    ├── experiment2/
    └── experiment3/
```

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Or install dependencies manually:

```bash
pip install torch torchvision diffusers transformers accelerate wandb pillow numpy tqdm
```

### Model Weights

The model automatically downloads pretrained weights from Hugging Face:
- `stable-diffusion-v1-5/stable-diffusion-inpainting`
- `lllyasviel/control_v11p_sd15_inpaint`

## Usage

### 1. Precompute Text Embeddings

Before training, precompute CLIP embeddings for your captions:

```bash
python precompute_captions.py
```

### 2. Training

Run the training script:

```bash
python train.py \
    --train_root /path/to/train/data \
    --val_root /path/to/val/data \
    --batch_size 2 \
    --num_epochs 10 \
    --lr 5e-6 \
    --wandb_project your_project_name
```

For distributed training:

```bash
accelerate launch train.py --train_root /path/to/data
```

### 3. Inference

Generate virtual try-on results:

```bash
python inference.py \
    --checkpoint_step 1000 \
    --person_path /path/to/person.jpg \
    --mask_path /path/to/mask.png \
    --clothing_path /path/to/garment.jpg \
    --prompt "a person wearing a red dress" \
    --output_path output.png
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_root` | Required | Path to training data |
| `--val_root` | Required | Path to validation data |
| `--batch_size` | 2 | Batch size per GPU |
| `--effective_batch_size` | 128 | Effective batch size with gradient accumulation |
| `--num_epochs` | 10 | Number of training epochs |
| `--lr` | 5e-6 | Learning rate |
| `--checkpoint_dir` | checkpoints | Directory to save checkpoints |
| `--save_every_steps` | None | Save checkpoint every N steps |
| `--wandb_project` | None | W&B project name (enables logging) |

### Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_step` | Required | Checkpoint step to load |
| `--person_path` | Required | Path to person image |
| `--mask_path` | Required | Path to mask image |
| `--clothing_path` | Required | Path to clothing image |
| `--prompt` | Required | Text prompt |
| `--height` | 512 | Output height |
| `--width` | 512 | Output width |
| `--guidance_scale` | 7.5 | CFG scale |
| `--num_inference_steps` | 50 | Number of denoising steps |

## Data Format

The dataloader expects the following structure:

```
data_root/
├── category1/
│   ├── images/           # Person images
│   ├── cloth/            # Garment images
│   ├── mask/             # Segmentation masks
│   ├── caption/          # Text captions (.txt files)
│   └── caption_embeds/   # Precomputed embeddings (.npz)
├── category2/
│   └── ...
```

## Distillation

For faster inference, train a distilled model:

```bash
python distillation/distill_train.py \
    --data_root /path/to/data \
    --batch_size 1 \
    --num_epochs 10 \
    --lr_student 1e-4 \
    --lr_discriminator 2e-4
```

## Acknowledgements

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [Diffusers](https://github.com/huggingface/diffusers)
