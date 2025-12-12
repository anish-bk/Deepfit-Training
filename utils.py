# utils.py

import os
import logging
from typing import Optional,List, Tuple, Dict, Any          
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class JointVirtualTryOnDataset(Dataset):
    """
    A dummy dataset that returns random tensors *and* prompt embeddings for testing. 

    Each sample dict contains:
      - "person_image":    Tensor[3, H, W]
      - "mask":            Tensor[1, H, W]
      - "clothing_image":  Tensor[3, H, W]
      - "tryon_gt":        Tensor[3, H, W]
      - "depth_gt":        Tensor[1, H, W]
      - "normal_gt":       Tensor[3, H, W]
      - "prompt_embeds":   Tensor[B, seq_len, dim]
    """
    def __init__(
        self,
        data_root: Optional[str] = None,
        transform=None,
        num_samples: int = 1000,
        image_size: tuple = (512, 512),  # SD1.5 default resolution
        # new args for encoding
        tokenizer: CLIPTokenizer = None,
        text_encoder: CLIPTextModel = None,
        device: str = "cuda",
        debug: bool = False
    ):
        super().__init__()
        self.transform = transform
        self.num_samples = num_samples
        self.C, self.H, self.W = 3, *image_size

        # for prompt encoding
        assert tokenizer is not None and text_encoder is not None, \
            "Must provide tokenizer and text encoder"
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.debug = debug

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # random log-uniform scale between 1e-3 and 1e3
        scale = 10 ** torch.empty(1).uniform_(-3, 3)

        # Base random Gaussian tensors
        person_image    = torch.randn(self.C, self.H, self.W) * scale
        mask            = torch.rand(1, self.H, self.W)         * scale
        clothing_image  = torch.randn(self.C, self.H, self.W) * scale
        tryon_gt        = torch.randn(self.C, self.H, self.W) * scale
        depth_gt        = torch.randn(1, self.H, self.W)     * scale
        normal_gt       = torch.randn(3, self.H, self.W)     * scale

        # Create a placeholder prompt
        prompt = f"sample prompt #{idx}"

        sample = {
            "person_image":   person_image,
            "mask":           mask,
            "clothing_image": clothing_image,
            "tryon_gt":       tryon_gt,
            "depth_gt":       depth_gt,
            "normal_gt":      normal_gt,
            "prompt":         prompt
        }

        # Apply any user-provided transform (e.g. normalization) to image tensors
        if self.transform is not None:
            sample = self.transform(sample)

        # Now encode the prompt into embeddings
        pe = encode_prompt(
            model=None,  # not used inside encode_prompt
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            prompts=[sample["prompt"]],
            device=self.device,
            debug=self.debug
        )
        # pe: [1, seq_len, dim]
        sample["prompt_embeds"] = pe.squeeze(0)

        # remove raw prompt string if you like
        sample.pop("prompt")

        return sample


def modify_unet_channels(unet: UNet2DConditionModel, new_in_channels: int, device: str):
    """
    Replace UNet conv_in to accept new_in_channels.
    Copies original weights into the first channels, zero-initializes new channels.
    """
    orig_conv_in = unet.conv_in
    orig_weight = orig_conv_in.weight.data  # shape [out_channels, in_orig, k, k]
    orig_bias = orig_conv_in.bias.data if orig_conv_in.bias is not None else None
    in_orig = orig_conv_in.in_channels
    out_channels = orig_conv_in.out_channels
    kernel_size = orig_conv_in.kernel_size
    stride = orig_conv_in.stride
    padding = orig_conv_in.padding
    
    if new_in_channels < in_orig:
        raise ValueError(f"new_in_channels ({new_in_channels}) < original in_channels ({in_orig})")
    
    new_conv_in = nn.Conv2d(
        new_in_channels, out_channels, kernel_size,
        stride=stride, padding=padding
    ).to(device).to(torch.float16)
    
    with torch.no_grad():
        new_conv_in.weight.data[:, :in_orig, :, :] = orig_weight
        if new_in_channels > in_orig:
            new_conv_in.weight.data[:, in_orig:, :, :].zero_()
        if orig_bias is not None:
            new_conv_in.bias.data[:] = orig_bias
    
    unet.conv_in = new_conv_in
    unet.config["in_channels"] = new_in_channels
    logger.info(f"[DEBUG] UNet conv_in replaced: in_channels {in_orig} → {new_in_channels}")
    
    return unet


def modify_controlnet_channels(controlnet: ControlNetModel, new_in_channels: int, device: str):
    """
    Modify ControlNet controlnet_cond_embedding to accept new_in_channels for conditioning.
    Copies weights and zero-initializes new channels.
    """
    # Modify the conditioning embedding (conv_in of controlnet_cond_embedding)
    orig_cond_embed = controlnet.controlnet_cond_embedding
    
    # For SD1.5 ControlNet, the conditioning embedding is a sequential of conv layers
    # We need to modify the first conv layer
    if hasattr(orig_cond_embed, 'conv_in'):
        orig_conv = orig_cond_embed.conv_in
    elif hasattr(orig_cond_embed, 'blocks') and len(orig_cond_embed.blocks) > 0:
        orig_conv = orig_cond_embed.blocks[0]
    else:
        # Try to access first layer directly
        orig_conv = list(orig_cond_embed.children())[0]
    
    orig_weight = orig_conv.weight.data
    orig_bias = orig_conv.bias.data if orig_conv.bias is not None else None
    in_orig = orig_conv.in_channels
    out_channels = orig_conv.out_channels
    kernel_size = orig_conv.kernel_size
    stride = orig_conv.stride
    padding = orig_conv.padding
    
    if new_in_channels < in_orig:
        raise ValueError(f"new_in_channels ({new_in_channels}) < original ({in_orig})")
    
    new_conv = nn.Conv2d(
        new_in_channels, out_channels, kernel_size,
        stride=stride, padding=padding
    ).to(device).to(torch.float16)
    
    with torch.no_grad():
        new_conv.weight.data[:, :in_orig, :, :] = orig_weight
        if new_in_channels > in_orig:
            new_conv.weight.data[:, in_orig:, :, :].zero_()
        if orig_bias is not None:
            new_conv.bias.data[:] = orig_bias
    
    # Replace the conv layer
    if hasattr(orig_cond_embed, 'conv_in'):
        orig_cond_embed.conv_in = new_conv
    elif hasattr(orig_cond_embed, 'blocks') and len(orig_cond_embed.blocks) > 0:
        orig_cond_embed.blocks[0] = new_conv
    else:
        # Replace first layer in sequential
        children = list(orig_cond_embed.children())
        children[0] = new_conv
        controlnet.controlnet_cond_embedding = nn.Sequential(*children)
    
    logger.info(f"[DEBUG] ControlNet cond_embedding replaced: in_channels {in_orig} → {new_in_channels}")
    
    return controlnet


def freeze_non_trainable_components(model: nn.Module):
    """
    Freeze VAE and text encoder; leave UNet and ControlNet parameters trainable.
    Assumes model has attributes: .vae, .text_encoder (or similar).
    """
    # Freeze VAE parameters
    if hasattr(model, "vae"):
        for param in model.vae.parameters():
            param.requires_grad = False
    # Freeze text encoder
    if hasattr(model, "text_encoder"):
        for param in model.text_encoder.parameters():
            param.requires_grad = False
    logger.info("[DEBUG] Frozen VAE and text encoder parameters. UNet and ControlNet left trainable.")


def encode_prompt(model: nn.Module, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel,
                  prompts: list, device: str, debug: bool = False) -> torch.Tensor:
    """
    Encode prompts using CLIP text encoder for SD1.5.
    Returns:
        prompt_embed: [B, seq_len, dim]
    """
    tokens = tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = text_encoder(tokens)
        prompt_embed = outputs.last_hidden_state.to(torch.float16)  # [B, 77, 768]
    
    if debug:
        logger.debug(f"[DEBUG] Prompt embeddings shape: {prompt_embed.shape}")
    return prompt_embed


def prepare_latents(batch_size: int, height: int, width: int, device: str, dtype=torch.float16) -> torch.Tensor:
    """
    Initialize random latents for inference.
    Output: [batch_size, 4, height//8, width//8] for SD1.5
    """
    shape = (batch_size, 4, height // 8, width // 8)
    latents = torch.randn(shape, device=device, dtype=dtype)
    logger.debug(f"[DEBUG] Initialized latents {latents.shape}")
    return latents


def encode_modality(vae: AutoencoderKL, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
    """
    Encode image x via VAE into latent space with scaling.
    x: [B, C, H, W]
    Returns [B, 4, H/8, W/8] for SD1.5
    """
    latent_dist = vae.encode(x).latent_dist
    latents = latent_dist.sample()
    # SD1.5 uses scaling factor of 0.18215
    latents = latents * vae.config.scaling_factor
    if debug:
        logger.debug(f"[DEBUG] VAE Encoded shape: {latents.shape}")
    return latents


def prepare_control_input(person_images: torch.Tensor, masks: torch.Tensor, clothing_images: torch.Tensor,
                          vae: AutoencoderKL, debug: bool = False) -> torch.Tensor:
    """
    Encode person_images and clothing_images via VAE, resize mask, then concat → [B,9,h/8,w/8] for SD1.5.
    (4 person latent + 1 mask + 4 clothing latent = 9 channels)
    """
    person_latents = encode_modality(vae, person_images, debug=debug)  # [B,4,h/8,w/8]
    clothing_latents = encode_modality(vae, clothing_images, debug=debug)  # [B,4,h/8,w/8]
    mask_latents = F.interpolate(masks, size=person_latents.shape[-2:], mode='nearest')  # [B,1,h/8,w/8]
    control_input = torch.cat([person_latents, mask_latents, clothing_latents], dim=1)  # [B,9,...]
    if debug:
        logger.debug(f"[DEBUG] Control input shape: {control_input.shape}")
    return control_input


def prepare_target_latents(tryon_gt: torch.Tensor, depth_gt: torch.Tensor, normal_gt: torch.Tensor,
                           vae: AutoencoderKL, debug: bool = False) -> torch.Tensor:
    """
    Encode tryon_gt via VAE → [B,4,h/8,w/8] for SD1.5.
    Note: For SD1.5, we typically just use the try-on latents without depth/normal concatenation.
    """
    tryon_latents = encode_modality(vae, tryon_gt, debug=debug)  # [B,4,...]
    if debug:
        logger.debug(f"[DEBUG] Target latents shape: {tryon_latents.shape}")
    return tryon_latents


def add_noise(target_latents: torch.Tensor, debug: bool = False, num_train_timesteps: int = 1000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add noise using DDPM-style discrete timesteps for SD1.5.
    Returns noisy_latents, noise, timesteps.
    """
    B = target_latents.shape[0]
    device = target_latents.device
    # Sample discrete timesteps from [0, num_train_timesteps)
    timesteps = torch.randint(0, num_train_timesteps, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(target_latents)
    
    # Simple linear noise schedule approximation
    # alphas_cumprod approximation for DDPM
    alphas_cumprod = torch.linspace(0.9999, 0.001, num_train_timesteps, device=device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[timesteps])[:, None, None, None]
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod[timesteps])[:, None, None, None]
    
    noisy_latents = sqrt_alphas_cumprod * target_latents + sqrt_one_minus_alphas_cumprod * noise
    
    if debug:
        logger.debug(f"[DEBUG] Noisy latents shape {noisy_latents.shape}, timesteps {timesteps.shape}")
    return noisy_latents, noise, timesteps


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, checkpoint_dir: str = "checkpoints"):
    """
    Save model.state_dict and optimizer.state_dict to checkpoint_dir.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"deepfit_step_{step}.pth")
    optim_path = os.path.join(checkpoint_dir, f"optim_step_{step}.pth")
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optim_path)
    logger.info(f"[DEBUG] Saved checkpoint at step {step}: {model_path}, {optim_path}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int,
                    checkpoint_dir: str = "checkpoints", device: str = "cuda"):
    """
    Load model and optimizer state dicts for given step.
    """
    model_path = os.path.join(checkpoint_dir, f"deepfit_step_{step}.pth")
    optim_path = os.path.join(checkpoint_dir, f"optim_step_{step}.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No model checkpoint at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    if os.path.isfile(optim_path):
        optimizer.load_state_dict(torch.load(optim_path, map_location=device))
        logger.info(f"[DEBUG] Loaded checkpoint step {step} into model and optimizer")
    else:
        logger.warning(f"No optimizer checkpoint at {optim_path}; loaded model only")
    return model, optimizer


def setup_optimizer(model: nn.Module, lr: float = 1e-4) -> torch.optim.Optimizer:
    """
    Set up AdamW optimizer including only parameters with requires_grad=True.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        logger.warning("No trainable parameters found for optimizer!")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    return optimizer


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seeded everything with seed={seed}")


def setup_wandb(wandb_config: Dict[str, Any]):
    """
    Initialize W&B if wandb_config provided.
    """
    import wandb
    if wandb_config is not None:
        wandb.init(
            project=wandb_config.get("project", "sd15-virtual-tryon"),
            name=wandb_config.get("name", None),
            entity=wandb_config.get("entity", None),
            config=wandb_config.get("config", {}),
            tags=wandb_config.get("tags", [])
        )
        logger.info("[DEBUG] W&B initialized")
        return True
    else:
        logger.info("[DEBUG] No W&B logging")
        return False






def print_trainable_parameters(model, logger: logging.Logger = None):
    """
    Print (or log) all trainable parameters of the given model.

    Args:
        model: a torch.nn.Module whose trainable parameters we want to inspect.
        logger: optional logging.Logger. If provided, uses logger.info to output;
                otherwise, uses print().
    Returns:
        A list of tuples (name, parameter) for trainable parameters.
    """
    use_logger = logger is not None
    def _out(msg):
        if use_logger:
            logger.info(msg)
        else:
            print(msg)

    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    _out(f"Trainable parameters ({len(trainable)} tensors):")
    total_params = 0
    for name, param in trainable:
        shape = tuple(param.shape)
        num = param.numel()
        total_params += num
        _out(f"  {name}: shape={shape}, params={num}")
    _out(f"Total trainable parameters: {total_params}")
    return trainable
