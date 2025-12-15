# train_experimental.py

import os
import argparse
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from model import DeepFit
from utils import (
    JointVirtualTryOnDataset,
    seed_everything,
    setup_wandb,
    encode_prompt,
    prepare_control_input,
    setup_optimizer,
    save_checkpoint,
    load_checkpoint
)
# Import function providing both train and validation loaders
from virtual_try_on_dataloader import get_train_val_loaders

import wandb  # for logging if enabled

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepFit (SD1.5 Inpainting + ControlNet) for Virtual Try-On"
    )
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (if logging)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_experimental")
    parser.add_argument("--save_every_steps", type=int, default=100)
    parser.add_argument("--resume_step", type=int, default=None, help="If provided, resume from this step")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    # If the loader function supports val_fraction, you can include it; otherwise omit
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction for validation split (if loader uses it)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    seed_everything(args.seed)

    # W&B setup
    wandb_config = None
    if args.wandb_project:
        wandb_config = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_name,
            "config": {
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "seed": args.seed,
                "val_fraction": args.val_fraction,
            },
            "tags": ["sd15", "controlnet", "virtual-tryon", "inpainting"]
        }
    use_wandb = setup_wandb(wandb_config)

    # Device handling
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        device = "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Obtain train and validation loaders from provided function
    try:
        train_loader, val_loader = get_train_val_loaders(
            args.data_root,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
            num_workers=4
        )
    except Exception as e:
        logger.error(f"Error obtaining train/val loaders: {e}")
        return
    logger.info(f"Obtained train loader with {len(train_loader)} batches and val loader with {len(val_loader)} batches.")

    # Model instantiation (SD1.5 4-channel latents)
    model = DeepFit(
        device=device,
        debug=args.debug,
        unet_in_channels=13,
        unet_out_channels=4,
        controlnet_in_channels=13,
        controlnet_cond_channels=9
    ).to(device)
    model.train()
    logger.info("Model instantiated and set to train mode.")

    # Optimizer
    optimizer = setup_optimizer(model, lr=args.lr)
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Optimizer lr={args.lr}; trainable params: {num_trainable:,}")

    # Resume checkpoint if requested
    global_step = 0
    if args.resume_step is not None:
        try:
            model, optimizer = load_checkpoint(
                model, optimizer, args.resume_step,
                checkpoint_dir=args.checkpoint_dir, device=device
            )
            global_step = args.resume_step + 1
            logger.info(f"Resumed from step {args.resume_step}; continuing at {global_step}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint at step {args.resume_step}: {e}")
            return

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    try:
        for epoch in range(args.num_epochs):
            # --- Training ---
            model.train()
            train_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
            for batch in pbar:
                # Expect batch to include prompt_embeds and pooled_prompt directly
                overlay = batch["overlay_image"].to(device, dtype=torch.float16)
                mask = batch["mask"].to(device, dtype=torch.float16)
                clothing = batch["clothing_image"].to(device, dtype=torch.float16)
                tryon_gt = batch["tryon_gt"].to(device, dtype=torch.float16)
                prompt_embeds = batch.get("prompt_embeds", None)
                pooled_prompt = batch.get("pooled_prompt", None)
                if prompt_embeds is not None:
                    prompt_embeds = prompt_embeds.to(device, dtype=torch.float16)
                if pooled_prompt is not None:
                    pooled_prompt = pooled_prompt.to(device, dtype=torch.float16)

                B = person.size(0)

                # 1. Prepare control input
                control_input = prepare_control_input(overlay, mask, clothing, debug=args.debug)

                # 2. Encode tryon_gt via VAE to image latents
                with torch.no_grad():
                    latent_dist = model.vae.encode(tryon_gt).latent_dist
                    tryon_latents = latent_dist.sample()
                tryon_latents = (tryon_latents - model.vae.config.shift_factor) * model.vae.config.scaling_factor

                # 3. Add noise to image latents
                timesteps = torch.rand(B, device=device)
                noise_img = torch.randn_like(tryon_latents)
                noisy_img = tryon_latents + noise_img * timesteps[:, None, None, None]

                # 4. Forward + loss
                pred_noise = model(noisy_img, timesteps, control_input, prompt_embeds, pooled_prompt)
                loss = F.mse_loss(pred_noise.float(), noise_img.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                global_step += 1

                # W&B logging
                if use_wandb:
                    wandb.log({"train/loss": loss.item()}, step=global_step)

                # Checkpoint
                if args.save_every_steps and global_step % args.save_every_steps == 0:
                    save_checkpoint(model, optimizer, global_step, checkpoint_dir=args.checkpoint_dir)

                pbar.set_postfix(train_loss=float(np.mean(train_losses[-10:])))

            avg_train = float(np.mean(train_losses)) if train_losses else 0.0
            logger.info(f"Epoch {epoch+1} train loss: {avg_train:.6f}")
            if use_wandb:
                wandb.log({"epoch/train_loss": avg_train}, step=global_step)

            # --- Validation ---
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    overlay = batch["overlay_image"].to(device, dtype=torch.float16)
                    mask = batch["mask"].to(device, dtype=torch.float16)
                    clothing = batch["clothing_image"].to(device, dtype=torch.float16)
                    tryon_gt = batch["tryon_gt"].to(device, dtype=torch.float16)
                    prompt_embeds = batch.get("prompt_embeds", None)
                    pooled_prompt = batch.get("pooled_prompt", None)
                    if prompt_embeds is not None:
                        prompt_embeds = prompt_embeds.to(device, dtype=torch.float16)
                    if pooled_prompt is not None:
                        pooled_prompt = pooled_prompt.to(device, dtype=torch.float16)

                    control_input = prepare_control_input(overlay, mask, clothing, debug=False)
                    latent_dist = model.vae.encode(tryon_gt).latent_dist
                    tryon_latents = (latent_dist.sample() - model.vae.config.shift_factor) * model.vae.config.scaling_factor
                    timesteps = torch.rand(overlay_image.size(0), device=device)
                    noise_img = torch.randn_like(tryon_latents)
                    noisy_img = tryon_latents + noise_img * timesteps[:, None, None, None]

                    pred_noise = model(noisy_img, timesteps, control_input, prompt_embeds, pooled_prompt)
                    v_loss = F.mse_loss(pred_noise.float(), noise_img.float())
                    val_losses.append(v_loss.item())

            avg_val = float(np.mean(val_losses)) if val_losses else 0.0
            logger.info(f"Epoch {epoch+1} val loss: {avg_val:.6f}")
            if use_wandb:
                wandb.log({"epoch/val_loss": avg_val}, step=global_step)

            # Save checkpoint at epoch end
            save_checkpoint(model, optimizer, global_step, checkpoint_dir=args.checkpoint_dir)
            logger.info(f"Saved checkpoint at end of epoch {epoch+1}, step {global_step}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted; saving checkpoint...")
        save_checkpoint(model, optimizer, global_step, checkpoint_dir=args.checkpoint_dir)
    finally:
        if use_wandb:
            wandb.finish()
        logger.info("Training complete.")


if __name__ == "__main__":
    main()

