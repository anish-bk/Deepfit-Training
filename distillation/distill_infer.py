#!/usr/bin/env python
"""
distill_infer.py

Inference with the distilled DeepFit student model (trained with latent discriminator).
Generates only the image output.

Usage:
    python distill_infer.py \
        --student_ckpt checkpoints_distill/student_step_100.pth \
        --person_path /path/to/person.png \
        --mask_path /path/to/mask.png \
        --clothing_path /path/to/clothing.png \
        --prompt "A person wearing a red dress" \
        --height 1024 \
        --width 1024 \
        --guidance_scale 7.0 \
        --num_inference_steps 28 \
        --output_path output.png \
        --use_ema \
        --debug
"""

import argparse
import logging
import torch
from PIL import Image
import numpy as np

from model import DeepFit
from distillation.distill_utils import LADDDistillationWrapper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with distilled DeepFit model")
    parser.add_argument("--student_ckpt", type=str, required=True, help="Path to student model state_dict (.pth)")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights for inference (if available)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--person_path", type=str, required=True, help="Path to person image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to mask image (grayscale)")
    parser.add_argument("--clothing_path", type=str, required=True, help="Path to clothing image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--height", type=int, default=1024, help="Height for inference")
    parser.add_argument("--width", type=int, default=1024, help="Width for inference")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of scheduler timesteps")
    parser.add_argument("--output_path", type=str, default="output.png", help="Where to save generated image")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def load_image_as_tensor(path: str, device: str, dtype=torch.float16) -> torch.Tensor:
    """
    Load RGB image from path, convert to tensor [3,H,W], values in [0,1], dtype float16.
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device=device, dtype=dtype)
    return tensor


def load_mask_as_tensor(path: str, device: str, dtype=torch.float16) -> torch.Tensor:
    """
    Load mask image as grayscale, threshold to {0,1}, return tensor [1,H,W], dtype float16.
    """
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=dtype)
    return tensor


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Instantiate DeepFit student (SD1.5 config)
    student = DeepFit(
        device=device,
        debug=args.debug,
        unet_in_channels=13,
        unet_out_channels=4,
        controlnet_in_channels=13,
        controlnet_cond_channels=9
    )
    # Load student state_dict
    sd = torch.load(args.student_ckpt, map_location=device)
    student.load_state_dict(sd)
    student.to(device)
    student.eval()
    logger.info(f"Loaded student model from {args.student_ckpt}")

    # Wrap in distillation wrapper for infer()
    wrapper = LADDDistillationWrapper(student_model=student)
    # If EMA weights exist, load them into wrapper.ema_student here before inference.

    # Load inputs
    person = load_image_as_tensor(args.person_path, device=device)
    mask = load_mask_as_tensor(args.mask_path, device=device)
    clothing = load_image_as_tensor(args.clothing_path, device=device)
    prompt = args.prompt

    result_image = wrapper.infer(
        person_image=person,
        mask=mask,
        clothing_image=clothing,
        prompt=prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        use_ema=args.use_ema
    )
    result_image.save(args.output_path)
    logger.info(f"Saved generated image to {args.output_path}")


if __name__ == "__main__":
    main()
