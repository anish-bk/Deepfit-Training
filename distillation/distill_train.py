#!/usr/bin/env python
"""
distill_train.py

Train DeepFit with LADD distillation using latent discriminator based on teacher features.
Assumes distill_utils.LatentDiscriminator and LADDDistillationWrapper as above.

Usage:
    python distill_train.py \
        --data_root /path/to/data \
        --batch_size 2 \
        --num_epochs 10 \
        --lr_student 1e-4 \
        --lr_discriminator 2e-4 \
        --wandb_project your_project \
        --wandb_entity your_entity \
        --wandb_name "distill_run" \
        --checkpoint_dir checkpoints_distill \
        --save_every_steps 100 \
        --debug
"""

import os
import argparse
import logging
import torch
from tqdm import tqdm
from accelerate import Accelerator

from model import DeepFit
from utils import JointVirtualTryOnDataset, seed_everything, setup_wandb
from distillation.distill_utils import LADDDistillationWrapper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepFit with LADD Distillation (latent discriminator)")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr_student", type=float, default=1e-4, help="Learning rate for student")
    parser.add_argument("--lr_discriminator", type=float, default=2e-4, help="Learning rate for discriminator")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (optional)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name (optional)")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name (optional)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_distill", help="Directory for saving checkpoints")
    parser.add_argument("--save_every_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Seed
    seed_everything(args.seed)

    # W&B setup
    if args.wandb_project:
        wandb_config = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_name,
            "config": {
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "lr_student": args.lr_student,
                "lr_discriminator": args.lr_discriminator,
                "seed": args.seed
            },
            "tags": ["sd3", "controlnet", "LADD-distill", "latent-discriminator"]
        }
    else:
        wandb_config = None
    use_wandb = setup_wandb(wandb_config)

    # Accelerator for mixed precision / multi-GPU
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Using device via Accelerator: {device}")

    # Dataset & DataLoader
    try:
        dataset = JointVirtualTryOnDataset(data_root=args.data_root)
    except NotImplementedError:
        logger.error("JointVirtualTryOnDataset not implemented. Please implement in utils.py.")
        return
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    logger.info(f"Dataset loaded: {len(dataset)} samples. Batch size: {args.batch_size}")

    # Instantiate student and teacher models (SD1.5 4-channel latents)
    student = DeepFit(
        device=str(device),
        debug=args.debug,
        unet_in_channels=13,
        unet_out_channels=4,
        controlnet_in_channels=13,
        controlnet_cond_channels=9
    )
    teacher = DeepFit(
        device=str(device),
        debug=args.debug,
        unet_in_channels=13,
        unet_out_channels=4,
        controlnet_in_channels=13,
        controlnet_cond_channels=9
    )
    teacher.load_state_dict(student.state_dict())
    teacher.eval()

    # Distillation wrapper
    distill_wrapper = LADDDistillationWrapper(student_model=student, teacher_model=teacher)

    # Override optimizers with CLI learning rates
    distill_wrapper.student_optimizer = torch.optim.AdamW(
        [p for p in distill_wrapper.student.parameters() if p.requires_grad],
        lr=args.lr_student,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    distill_wrapper.discriminator_optimizer = torch.optim.AdamW(
        distill_wrapper.latent_discriminator.parameters(),
        lr=args.lr_discriminator,
        betas=(0.5, 0.999)
    )

    # Prepare with Accelerator
    distill_wrapper, train_dataloader = accelerator.prepare(distill_wrapper, train_dataloader)

    # Checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Resume logic: implement loading if desired
    if args.resume:
        logger.info("Resume requested: please implement checkpoint loading logic if needed.")

    global_step = 0
    for epoch in range(args.num_epochs):
        logger.info(f"=== Epoch {epoch+1}/{args.num_epochs} ===")
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            logs = distill_wrapper.train_step(batch, global_step)
            global_step += 1

            if use_wandb:
                import wandb
                wandb.log({f"distill/{k}": v for k, v in logs.items()}, step=global_step)

            if global_step % args.save_every_steps == 0:
                # Save student and discriminator state_dicts
                student_ckpt = os.path.join(args.checkpoint_dir, f"student_step_{global_step}.pth")
                disc_ckpt = os.path.join(args.checkpoint_dir, f"discriminator_step_{global_step}.pth")
                accelerator.save(accelerator.unwrap_model(distill_wrapper.student).state_dict(), student_ckpt)
                accelerator.save(accelerator.unwrap_model(distill_wrapper.latent_discriminator).state_dict(), disc_ckpt)
                logger.info(f"Saved checkpoints at step {global_step}")

        # End-of-epoch checkpoint
        student_ckpt = os.path.join(args.checkpoint_dir, f"student_epoch_{epoch+1}.pth")
        disc_ckpt = os.path.join(args.checkpoint_dir, f"discriminator_epoch_{epoch+1}.pth")
        accelerator.save(accelerator.unwrap_model(distill_wrapper.student).state_dict(), student_ckpt)
        accelerator.save(accelerator.unwrap_model(distill_wrapper.latent_discriminator).state_dict(), disc_ckpt)
        logger.info(f"Saved end-of-epoch checkpoints at epoch {epoch+1}")

    if use_wandb:
        import wandb
        wandb.finish()
        logger.info("W&B run finished.")


if __name__ == "__main__":
    main()
