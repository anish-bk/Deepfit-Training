# inference.py

import os
import argparse
import logging
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from model import DeepFit
from utils import encode_prompt, prepare_control_input, prepare_latents, load_checkpoint

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with DeepFit (SD1.5 Inpainting + ControlNet) - image-only latents")
    parser.add_argument("--checkpoint_step", type=int, required=True, help="Checkpoint step to load")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--output_path", type=str, default="output.png", help="Where to save generated image")

    parser.add_argument("--person_path", type=str, required=True, help="Path to person image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to mask image (grayscale)")
    parser.add_argument("--clothing_path", type=str, required=True, help="Path to clothing image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")

    parser.add_argument("--height", type=int, default=512, help="Height for inference")
    parser.add_argument("--width", type=int, default=512, help="Width for inference")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of scheduler timesteps")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    return parser.parse_args()


def load_image_as_tensor(path: str, device: str, dtype=torch.float16) -> torch.Tensor:
    """
    Load RGB image [H,W,3], convert to [3,H,W], float in [0,1].
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device=device, dtype=dtype)
    return tensor


def load_mask_as_tensor(path: str, device: str, dtype=torch.float16) -> torch.Tensor:
    """
    Load mask image as grayscale, threshold to {0,1}, shape [1,H,W].
    """
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=dtype)
    return tensor


def main():
    args = parse_args()
    # Set logging level if debug
    if args.debug:
        logger.setLevel(logging.DEBUG)

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Instantiate model and load checkpoint
    model = DeepFit(
        device=device,
        debug=args.debug,
        unet_in_channels=13,
        unet_out_channels=4,
        controlnet_in_channels=13,
        controlnet_cond_channels=9
    ).to(device)
    model.eval()
    logger.info("Model instantiated (16-channel latents) in eval mode.")
    try:
        optimizer_dummy = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model, _ = load_checkpoint(
            model, optimizer_dummy, args.checkpoint_step,
            checkpoint_dir=args.checkpoint_dir, device=device
        )
        logger.info(f"Loaded checkpoint step {args.checkpoint_step}.")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return

    # Load images for ControlNet conditioning
    logger.info("Loading input images for ControlNet conditioning...")
    person = load_image_as_tensor(args.person_path, device=device)      # [3,H,W]
    mask = load_mask_as_tensor(args.mask_path, device=device)           # [1,H,W]
    clothing = load_image_as_tensor(args.clothing_path, device=device)  # [3,H,W]
    if args.debug:
        logger.debug(f"Person shape: {person.shape}, mask: {mask.shape}, clothing: {clothing.shape}")

    # Add batch dim if needed
    if person.ndim == 3:
        person = person.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if clothing.ndim == 3:
        clothing = clothing.unsqueeze(0)
    B = person.shape[0]
    prompts = [args.prompt] * B
    logger.info(f"Batch size for inference: {B}. Prompt: '{args.prompt}'")

    # Encode conditional and unconditional prompts
    logger.info("Encoding conditional prompt...")
    prompt_embeds, pooled_prompt = encode_prompt(
        model=model,
        tokenizer1=model.tokenizer1,
        text_encoder1=model.text_encoder1,
        tokenizer2=model.tokenizer2,
        text_encoder2=model.text_encoder2,
        tokenizer3=model.tokenizer3,
        text_encoder3=model.text_encoder3,
        prompts=prompts,
        device=device,
        debug=args.debug
    )
    if args.debug:
        logger.debug(f"Conditional prompt_embeds shape: {prompt_embeds.shape}")

    logger.info("Encoding unconditional (empty) prompt...")
    uncond_prompt = [""] * B
    prompt_embeds_uncond, pooled_prompt_uncond = encode_prompt(
        model=model,
        tokenizer1=model.tokenizer1,
        text_encoder1=model.text_encoder1,
        tokenizer2=model.tokenizer2,
        text_encoder2=model.text_encoder2,
        tokenizer3=model.tokenizer3,
        text_encoder3=model.text_encoder3,
        prompts=uncond_prompt,
        device=device,
        debug=args.debug
    )
    if args.debug:
        logger.debug(f"Unconditional prompt_embeds shape: {prompt_embeds_uncond.shape}")

    # Prepare control input (unchanged)
    logger.info("Preparing control input...")
    control_input = prepare_control_input(person, mask, clothing, debug=args.debug)
    control_input_uncond = torch.zeros_like(control_input)
    if args.debug:
        logger.debug(f"Control input shape: {control_input.shape}, Uncond control input shape: {control_input_uncond.shape}")

    # Prepare initial image latents: random [B,16,h/8,w/8]
    logger.info("Initializing random image latents...")
    latent_h = args.height // 8
    latent_w = args.width // 8
    latents_img = torch.randn((B, 16, latent_h, latent_w), device=device, dtype=torch.float16)
    latents = latents_img  # we only work with image latents
    if args.debug:
        logger.debug(f"Initial image latents shape: {latents.shape}")

    # Denoising loop
    num_steps = args.num_inference_steps
    scheduler = model.scheduler
    scheduler.set_timesteps(num_steps)
    logger.info(f"Starting denoising loop with {num_steps} steps, guidance_scale={args.guidance_scale}.")

    for i, t in enumerate(scheduler.timesteps):
        # Duplicate latents for uncond + cond
        latent_model_input = torch.cat([latents, latents], dim=0)  # [2*B,16,...]
        # Prepare timesteps tensor
        if isinstance(t, torch.Tensor):
            timestep = t.expand(latent_model_input.shape[0])
        else:
            timestep = torch.tensor([t] * latent_model_input.shape[0], device=device)
        # Prepare embeddings
        encoder_states = torch.cat([prompt_embeds_uncond, prompt_embeds], dim=0)
        pooled_states = torch.cat([pooled_prompt_uncond, pooled_prompt], dim=0)
        # Prepare control inputs
        control_inputs = torch.cat([control_input_uncond, control_input], dim=0)

        if args.debug:
            logger.debug(f"Step {i+1}/{num_steps}, timestep {t}: latent_model_input shape {latent_model_input.shape}")

        # Forward through ControlNet & Transformer
        control_block = model.controlnet(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=encoder_states,
            pooled_projections=pooled_states,
            controlnet_cond=control_inputs,
            conditioning_scale=1.0,
            return_dict=False,
        )[0]
        if args.debug:
            logger.debug(f"ControlNet output shape: {control_block.shape}")

        noise_pred = model.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=encoder_states,
            pooled_projections=pooled_states,
            block_controlnet_hidden_states=control_block,
            return_dict=False,
        )[0]  # [2*B,16,latent_h,latent_w]
        if args.debug:
            logger.debug(f"Transformer output (noise_pred) shape: {noise_pred.shape}")

        # Split uncond/cond
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
        # Guidance on image latents
        guided_noise = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)  # [B,16,...]

        # Step scheduler: update only image latents
        latents = scheduler.step(guided_noise, t, latents).prev_sample  # [B,16,...]
        if args.debug:
            logger.debug(f"After scheduler.step: image latents shape: {latents.shape}")

    # After denoising, decode image latents via VAE
    logger.info("Decoding final image from latents...")
    final_img_latents = latents  # [B,16,latent_h,latent_w]
    # Undo VAE scaling: (latents / scaling_factor) + shift_factor
    final_img_latents = (final_img_latents / model.vae.config.scaling_factor) + model.vae.config.shift_factor
    final_img_latents = final_img_latents.to(dtype=model.vae.dtype)
    with torch.no_grad():
        decoded = model.vae.decode(final_img_latents, return_dict=False)[0]  # [B,3,H,W]
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()  # [B,H,W,3]
    # Save first image
    img_arr = (decoded[0] * 255).astype(np.uint8)
    out_img = Image.fromarray(img_arr)
    out_img.save(args.output_path)
    logger.info(f"Saved generated image to {args.output_path}")


if __name__ == "__main__":
    main()
