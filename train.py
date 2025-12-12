import os
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb  # for logging

from model import DeepFit
from utils import (
    seed_everything,
    setup_wandb,
    prepare_control_input,
    prepare_target_latents,
    add_noise,
    setup_optimizer,
    save_checkpoint,
    load_checkpoint,
    print_trainable_parameters
)
from virtual_try_on_dataloader import get_train_val_loaders

# ---------------------------------
# Train DeepFit (SD1.5 Inpainting + ControlNet) with W&B logging
# ---------------------------------

# Hard‐coded W&B key (or pull from env)
WANDB_API_KEY = "522f467266173e2d09d35d6e899e2d39a2fb2b49"

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="Train DeepFit with W&B logging")
    p.add_argument("--train_root",     type=str,   default="D:\PUL - DeepFit\Dresscode")
    p.add_argument("--val_root",       type=str,   default="D:\PUL - DeepFit\Dresscode\test")
    p.add_argument(
        "--categories",
        nargs='+',
        default=['dresses', 'upper_body', 'lower_body', 'upper_body1'],
        help=(
            "Space‑separated list of subfolder names under train_root;\n"
            "defaults to ['dresses','upper_body','lower_body','upper_body1']"
        )
    )
    p.add_argument("--device",         type=str,   default="cuda")
    p.add_argument("--batch_size",     type=int,   default=2)
    p.add_argument("--effective_batch_size", type=int, default=128)
    p.add_argument("--num_epochs",     type=int,   default=10)
    p.add_argument("--lr",             type=float, default=5*1e-6)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--wandb_project",  type=str,   default=None)
    p.add_argument("--wandb_entity",   type=str,   default=None)
    p.add_argument("--wandb_name",     type=str,   default=None)
    p.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    p.add_argument("--resume_step",    type=int,   default=None)
    p.add_argument("--save_every_steps", type=int, default=None,
                   help="If set, save a checkpoint every N global steps")
    p.add_argument("--debug",          action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # W&B login
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
    else:
        logger.warning("No W&B API key found.")

    # W&B init
    wandb_cfg = None
    if args.wandb_project:
        wandb_cfg = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_name,
            "config": {
                "batch_size": args.batch_size,
                "effective_batch_size": args.effective_batch_size,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "seed": args.seed,
                "categories": args.categories
            },
            "tags": ["sd15", "controlnet", "virtual-tryon", "inpainting"]
        }
    use_wandb = setup_wandb(wandb_cfg)

    # Seed
    seed_everything(args.seed)

    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Gradient accumulation
    if args.effective_batch_size % args.batch_size != 0:
        raise ValueError("effective_batch_size must be multiple of batch_size")
    accum_steps = args.effective_batch_size // args.batch_size
    logger.info(f"Accumulating over {accum_steps} steps to reach {args.effective_batch_size}")

    # Data loaders
    train_loader, val_loader = get_train_val_loaders(
        train_root=args.train_root,
        val_root=args.val_root,
        categories=args.categories,
        batch_size=args.batch_size,
        shuffle_train=True,
        num_workers=8
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model & optimizer
    model = DeepFit(device=device, debug=args.debug).to(device)
    model.train()
    optimizer = setup_optimizer(model, lr=args.lr)
    n_params = print_trainable_parameters(model)
    logger.info(f"Trainable parameters: {n_params}")

    # Resume checkpoint
    start_step = args.resume_step + 1 if args.resume_step is not None else 0
    if args.resume_step is not None:
        model, optimizer = load_checkpoint(
            model, optimizer, args.resume_step,
            checkpoint_dir=args.checkpoint_dir, device=device
        )
    global_step = start_step
    optimizer.zero_grad()

    # Training loop
    for epoch in range(args.num_epochs):
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            overlay = batch["overlay_image"].to(device, dtype=torch.float16)
            mask    = batch["mask"].to(device, dtype=torch.float16)
            cloth   = batch["cloth_image"].to(device, dtype=torch.float16)
            depth   = batch["depth_map"].to(device, dtype=torch.float16)
            normal  = batch["normal_map"].to(device, dtype=torch.float16)
            pe      = batch["prompt_embeds"].to(device, dtype=torch.float16)
            pp      = batch["pooled_prompt"].to(device, dtype=torch.float16)

            # Forward
            ctrl = prepare_control_input(overlay, mask, cloth, model.vae, args.debug)
            tgt  = prepare_target_latents(overlay, depth, normal, model.vae, args.debug)
            noised, noise, t = add_noise(tgt, args.debug)
            noised = noised.to(device, dtype=torch.float16)
            noise  = noise .to(device, dtype=torch.float16)
            pred   = model(noised, t, ctrl, pe, pp)

           

            loss = F.mse_loss(pred.float(), noise.float()) / accum_steps
            train_losses.append(loss.item() * accum_steps)
            loss.backward()
            # if use_wandb and wandb.run is not None:
            #     logger.info(f"W&B URL: {wandb.run.get_url}")
            # print(wandb.run.get_url)    
        

            # Step, logging & periodic save
            if (batch_idx + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = float(np.mean(train_losses[-accum_steps:]))
                if use_wandb:
                    wandb.log({"train/loss": avg_loss}, step=global_step)
                pbar.set_postfix(loss=avg_loss)

                # Save every N steps if requested
                if args.save_every_steps and global_step % args.save_every_steps == 0:
                    save_checkpoint(
                        model, optimizer, global_step,
                        checkpoint_dir=args.checkpoint_dir
                    )
                    logger.info(f"Saved checkpoint at step {global_step}")

        # End of epoch: logging and final save
            
        avg_train = float(np.mean(train_losses))
        logger.info(f"Epoch {epoch+1} train loss: {avg_train:.4f}")
        if use_wandb:
            wandb.log({"epoch/train_loss": avg_train}, step=global_step)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                overlay = batch["overlay_image"].to(device, dtype=torch.float16)
                mask    = batch["mask"].to(device, dtype=torch.float16)
                cloth   = batch["cloth_image"].to(device, dtype=torch.float16)
                depth   = batch["depth_map"].to(device, dtype=torch.float16)
                normal  = batch["normal_map"].to(device, dtype=torch.float16)
                pe      = batch["prompt_embeds"].to(device, dtype=torch.float16)
                pp      = batch["pooled_prompt"].to(device, dtype=torch.float16)

                ctrl = prepare_control_input(overlay, mask, cloth, model.vae, False)
                tgt  = prepare_target_latents(overlay, depth, normal, model.vae, False)
                noised, noise, t = add_noise(tgt, False)
                pred = model(noised, t, ctrl, pe, pp)

                val_losses.append(F.mse_loss(pred.float(), noise.float()).item())

        avg_val = float(np.mean(val_losses))
        logger.info(f"Epoch {epoch+1} val loss: {avg_val:.4f}")
        if use_wandb:
            wandb.log({"epoch/val_loss": avg_val}, step=global_step)

        # Always save at end of epoch
        save_checkpoint(
            model, optimizer, global_step,
            checkpoint_dir=args.checkpoint_dir
        )
        logger.info(f"Saved end‑of‑epoch checkpoint at step {global_step}")
        model.train()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

