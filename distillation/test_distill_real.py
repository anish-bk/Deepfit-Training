# test_distill_real.py

import torch
import logging
import time

from model import DeepFit
from distillation.distill_utils import LatentDiscriminator, LADDDistillationWrapper

# Optionally import GPU check
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_batch(B: int, H: int, W: int, device: torch.device):
    """
    Create a dummy batch dict for training/inference:
    - person_image: random [B,3,H,W] in [0,1]
    - mask: random binary [B,1,H,W]
    - clothing_image: random [B,3,H,W]
    - tryon_gt: random [B,3,H,W]
    - depth_gt: random [B,1,H,W]
    - normal_gt: random [B,3,H,W]
    - prompt: list of strings length B
    """
    # Use uniform [0,1] for images, convert to float32
    person_image = torch.rand((B, 3, H, W), device=device, dtype=torch.float32)
    # Binary mask 0 or 1
    mask = (torch.rand((B, 1, H, W), device=device) > 0.5).float()
    clothing_image = torch.rand((B, 3, H, W), device=device, dtype=torch.float32)
    tryon_gt = torch.rand((B, 3, H, W), device=device, dtype=torch.float32)
    depth_gt = torch.rand((B, 1, H, W), device=device, dtype=torch.float32)
    normal_gt = torch.rand((B, 3, H, W), device=device, dtype=torch.float32)
    prompts = ["a test prompt"] * B
    batch = {
        "person_image": person_image,
        "mask": mask,
        "clothing_image": clothing_image,
        "tryon_gt": tryon_gt,
        "depth_gt": depth_gt,
        "normal_gt": normal_gt,
        "prompt": prompts
    }
    return batch

def test_latent_discriminator_with_real_model():
    """
    Test LatentDiscriminator on real DeepFit transformer:
    - Instantiate DeepFit teacher
    - Create dummy latent input [B,20,h8,w8], dummy prompt_embeds, pooled_prompt, control_input
    - Forward through LatentDiscriminator and observe printed shapes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Testing LatentDiscriminator on device: {device}")

    # Instantiate a DeepFit model as teacher; minimal instantiation, loads pretrained weights
    # Use debug=False to reduce prints
    # Use smaller device string for DeepFit
    teacher = DeepFit(
        device=str(device),
        debug=False,
        unet_in_channels=13,
        unet_out_channels=4,
        controlnet_in_channels=13,
        controlnet_cond_channels=9
    )
    teacher.to(device)
    teacher.eval()

    # Instantiate LatentDiscriminator
    disc = LatentDiscriminator(teacher).to(device)

    # Dummy inputs for discriminator:
    B = 2
    # We choose dummy spatial resolution H=W=256 for images, so latent spatial dims h8=w8=32
    H_img, W_img = 256, 256
    h8, w8 = H_img // 8, W_img // 8  # typically 32
    # Dummy noisy_latents: [B,4,h8,w8] for SD1.5
    noisy_latents = torch.randn((B, 4, h8, w8), device=device, dtype=torch.float16)
    # For prompt_embeds: use encode_prompt
    prompts = ["test prompt"] * B
    with torch.no_grad():
        prompt_embeds = teacher._encode_prompt(prompts)
    # control_input: use teacher._prepare_control_input with dummy person/clothing
    # Need dummy person_image [B,3,H_img,W_img], mask [B,1,H_img,W_img], clothing_image [B,3,H_img,W_img]
    person = torch.rand((B,3,H_img,W_img), device=device, dtype=torch.float32)
    mask = (torch.rand((B,1,H_img,W_img), device=device) > 0.5).float()
    clothing = torch.rand((B,3,H_img,W_img), device=device, dtype=torch.float32)
    with torch.no_grad():
        control_input = teacher._prepare_control_input(person, mask, clothing)
    logger.info(f"Dummy inputs shapes for discriminator: noisy_latents {noisy_latents.shape}, "
                f"prompt_embeds {prompt_embeds.shape}, "
                f"control_input {control_input.shape}")

    # Forward through discriminator; it will print intermediate shapes inside
    logger.info("Running LatentDiscriminator forward pass...")
    scores = disc(noisy_latents, prompt_embeds, control_input)
    logger.info(f"LatentDiscriminator output scores shape: {scores.shape} (expected [B,1])")

def test_wrapper_train_step_with_real_model():
    """
    Test LADDDistillationWrapper.train_step on real DeepFit models:
    - Instantiate student and teacher DeepFit
    - Create dummy batch with random tensors
    - Run wrapper.train_step once, print returned losses
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Testing LADDDistillationWrapper.train_step on device: {device}")

    # Instantiate student and teacher DeepFit models (SD1.5 config)
    student = DeepFit(
        device=str(device),
        debug=False,
        unet_in_channels=13,
        unet_out_channels=4,
        controlnet_in_channels=13,
        controlnet_cond_channels=9
    )
    student.to(device)
    student.eval()
    # Teacher copy
    teacher = DeepFit(
        device=str(device),
        debug=False,
        unet_in_channels=13,
        unet_out_channels=4,
        controlnet_in_channels=13,
        controlnet_cond_channels=9
    )
    teacher.load_state_dict(student.state_dict())
    teacher.to(device)
    teacher.eval()

    # Instantiate wrapper
    wrapper = LADDDistillationWrapper(student_model=student, teacher_model=teacher)
    # Override optimizers to avoid large default lr if desired (already default small)
    # wrapper.student_optimizer = torch.optim.AdamW([...], lr=1e-5)
    # wrapper.discriminator_optimizer = torch.optim.AdamW([...], lr=1e-5)

    # Create dummy batch
    B = 1  # for speed, use batch size 1
    H_img, W_img = 256, 256  # multiple of 8
    batch = create_dummy_batch(B, H_img, W_img, device)

    # Print shapes
    logger.info(f"Dummy batch shapes: person_image {batch['person_image'].shape}, "
                f"mask {batch['mask'].shape}, clothing_image {batch['clothing_image'].shape}, "
                f"tryon_gt {batch['tryon_gt'].shape}, depth_gt {batch['depth_gt'].shape}, "
                f"normal_gt {batch['normal_gt'].shape}, prompts len {len(batch['prompt'])}")

    # Run one training step
    logger.info("Running one train_step...")
    start = time.time()
    logs = wrapper.train_step(batch, step=0)
    elapsed = time.time() - start
    logger.info(f"train_step completed in {elapsed:.2f}s, logs: {logs}")

    # Check keys
    expected_keys = {"loss/mse", "loss/adv", "loss/consistency", "loss/discriminator", "loss/total"}
    if not expected_keys.issubset(set(logs.keys())):
        logger.error(f"Returned logs missing keys: {set(logs.keys())}")
    else:
        logger.info("LADDDistillationWrapper.train_step returned expected loss keys.")

def main():
    # Check if GPU is available
    if torch.cuda.is_available():
        logger.info("CUDA is available; using GPU.")
    else:
        logger.info("CUDA not available; using CPU (this may be slow).")

    # Test latent discriminator
    try:
        test_latent_discriminator_with_real_model()
    except Exception as e:
        logger.error(f"Error during LatentDiscriminator test: {e}", exc_info=True)

    # Test wrapper train_step
    try:
        test_wrapper_train_step_with_real_model()
    except Exception as e:
        logger.error(f"Error during LADDDistillationWrapper.train_step test: {e}", exc_info=True)

if __name__ == "__main__":
    main()
