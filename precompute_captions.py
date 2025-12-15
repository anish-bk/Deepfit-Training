#!/usr/bin/env python
"""
precompute_captions.py

Precompute caption embeddings for Stable Diffusion 1.5 Inpainting.

Usage:
    # Single category
    python precompute_captions.py \
        --data_root "path/to/dresscode/dresses" \
        --output_dir "path/to/dresscode/dresses/caption_embeds"
    
    # Full dresscode dataset (processes all subdirectories)
    python precompute_captions.py \
        --data_root "path/to/dresscode" \
        --batch_size 8
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm

from utils import encode_prompt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute caption embeddings for SD1.5")
    parser.add_argument(
        "--data_root", 
        type=str, 
        required=True,
        help="Root directory for dataset. Can be a single category (with caption/ subfolder) "
             "or parent directory containing multiple categories (dresses, upper_body, lower_body)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory for embeddings. If not specified, saves to <data_root>/caption_embeds "
             "or <category>/caption_embeds for each category"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for encoding captions"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-inpainting",
        help="Hugging Face model ID for tokenizer and text encoder"
    )
    return parser.parse_args()


def process_category(cap_dir: str, output_dir: str, tokenizer, text_encoder, batch_size: int, device: str, category_name: str = ""):
    """Process a single category directory with caption/ subfolder."""
    
    if not os.path.isdir(cap_dir):
        logger.warning(f"Caption directory not found: {cap_dir}")
        return False
    
    # Collect all caption files
    files = sorted(f for f in os.listdir(cap_dir) if f.endswith(".txt"))
    if not files:
        logger.warning(f"No .txt files found in {cap_dir}")
        return False
    
    names = [os.path.splitext(f)[0] for f in files]
    logger.info(f"[{category_name}] Found {len(names)} captions to process")
    
    # Read all captions
    captions = []
    for name in names:
        caption_path = os.path.join(cap_dir, name + ".txt")
        with open(caption_path, "r", encoding="utf-8") as f:
            captions.append(f.read().strip())
    
    # Dummy model for encode_prompt compatibility
    model = nn.Identity()
    
    # Batch encode
    prompt_embeds_list = []
    for i in tqdm(range(0, len(captions), batch_size), desc=f"Encoding [{category_name}]"):
        batch = captions[i : i + batch_size]
        pe = encode_prompt(
            model,
            tokenizer, 
            text_encoder,
            batch,
            device,
            debug=False,
        )
        pe = pe.detach().cpu().half().numpy()
        prompt_embeds_list.append(pe)
    
    # Concatenate all embeddings
    all_embeds = np.concatenate(prompt_embeds_list, axis=0)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as compressed npz
    out_path = os.path.join(output_dir, "precomputed_prompts.npz")
    np.savez_compressed(out_path, pe=all_embeds, names=np.array(names))
    
    logger.info(f"[{category_name}] Saved {all_embeds.shape} embeddings → {out_path}")
    return True


def main():
    args = parse_args()
    
    # Clear CUDA cache
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and text encoder
    logger.info(f"Loading tokenizer and text encoder from {args.model_id}...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id,
        subfolder="tokenizer"
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    text_encoder.eval()
    logger.info("Text encoder loaded successfully")
    
    # Check if data_root is a single category or parent with multiple categories
    cap_dir_direct = os.path.join(args.data_root, "caption")
    
    if os.path.isdir(cap_dir_direct):
        # Single category mode
        logger.info(f"Single category mode: {args.data_root}")
        output_dir = args.output_dir if args.output_dir else os.path.join(args.data_root, "caption_embeds")
        category_name = os.path.basename(args.data_root)
        
        success = process_category(
            cap_dir=cap_dir_direct,
            output_dir=output_dir,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            batch_size=args.batch_size,
            device=device,
            category_name=category_name
        )
        
        if success:
            logger.info("Done!")
        else:
            logger.error("Failed to process captions")
    else:
        # Multi-category mode (parent directory)
        logger.info(f"Multi-category mode: {args.data_root}")
        categories = sorted([
            d for d in os.listdir(args.data_root) 
            if os.path.isdir(os.path.join(args.data_root, d))
        ])
        
        if not categories:
            logger.error(f"No subdirectories found in {args.data_root}")
            return
        
        logger.info(f"Found categories: {categories}")
        
        processed = 0
        for category in categories:
            category_path = os.path.join(args.data_root, category)
            cap_dir = os.path.join(category_path, "caption")
            
            if not os.path.isdir(cap_dir):
                logger.info(f"Skipping {category} (no caption/ folder)")
                continue
            
            # Determine output directory
            if args.output_dir:
                output_dir = os.path.join(args.output_dir, category)
            else:
                output_dir = os.path.join(category_path, "caption_embeds")
            
            success = process_category(
                cap_dir=cap_dir,
                output_dir=output_dir,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                batch_size=args.batch_size,
                device=device,
                category_name=category
            )
            
            if success:
                processed += 1
        
        logger.info(f"Done! Processed {processed}/{len(categories)} categories")


if __name__ == "__main__":
    main()