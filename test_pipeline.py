#!/usr/bin/env python
"""
test_pipeline.py

Comprehensive test script to verify the entire DeepFit pipeline:
1. Model instantiation
2. Forward pass
3. Training step
4. Inference
5. Distillation
6. Dataloader

Usage:
    python test_pipeline.py --test all
    python test_pipeline.py --test model
    python test_pipeline.py --test train
    python test_pipeline.py --test inference
    python test_pipeline.py --test distill
    python test_pipeline.py --test dataloader
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import tempfile
import shutil
import torchvision.transforms as transforms

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------- DATASET CONFIGURATION ----------------------
# Path to the test dataset folder
DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
IMAGE_SIZE = (512, 512)

# Transforms matching the dataloader
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

depth_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

basic_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])


def load_test_sample(sample_idx=0):
    """
    Load a sample from the test dataset.
    
    Returns a dictionary with:
        - person_image: (3, H, W) tensor normalized to [-1, 1]
        - cloth_image: (3, H, W) tensor normalized to [-1, 1]
        - normal_map: (3, H, W) tensor normalized to [-1, 1]
        - depth_map: (1, H, W) tensor normalized to [-1, 1]
        - mask: (1, H, W) tensor in [0, 1]
        - overlay_image: (3, H, W) tensor with masked region grayed out
        - caption: string
        - filename: base filename
    """
    # List available samples
    image_dir = os.path.join(DATASET_ROOT, "image")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Dataset not found at {DATASET_ROOT}. Please ensure the 'dataset' folder exists.")
    
    # Get all _0 images (person images)
    person_files = sorted([f for f in os.listdir(image_dir) if f.endswith("_0.jpg")])
    if len(person_files) == 0:
        raise FileNotFoundError(f"No person images found in {image_dir}")
    
    if sample_idx >= len(person_files):
        sample_idx = sample_idx % len(person_files)
    
    # Get base name (e.g., "020714")
    base_name = person_files[sample_idx].replace("_0.jpg", "")
    logger.info(f"Loading test sample: {base_name}")
    
    # Build paths
    person_path = os.path.join(DATASET_ROOT, "image", f"{base_name}_0.jpg")
    cloth_path = os.path.join(DATASET_ROOT, "cloth", f"{base_name}_1.jpg")
    normal_path = os.path.join(DATASET_ROOT, "normal", f"{base_name}_0.jpg")
    depth_path = os.path.join(DATASET_ROOT, "depth", f"{base_name}_0.jpg")
    mask_path = os.path.join(DATASET_ROOT, "mask", f"{base_name}_0.png")
    caption_path = os.path.join(DATASET_ROOT, "caption", f"{base_name}_0.txt")
    
    # Verify files exist
    for path, name in [(person_path, "person"), (cloth_path, "cloth"), 
                        (normal_path, "normal"), (depth_path, "depth"),
                        (mask_path, "mask")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {name} file: {path}")
    
    # Load images
    person_pil = Image.open(person_path).convert("RGB")
    cloth_pil = Image.open(cloth_path).convert("RGB")
    normal_pil = Image.open(normal_path).convert("RGB")
    depth_pil = Image.open(depth_path).convert("L")
    mask_pil = Image.open(mask_path).convert("L")
    
    # Create overlay (gray out masked region)
    person_np = np.array(person_pil.resize(IMAGE_SIZE))
    mask_np = np.array(mask_pil.resize(IMAGE_SIZE))
    blurred = cv2.GaussianBlur(mask_np, (21, 21), sigmaX=10)
    alpha = np.expand_dims(blurred.astype(np.float32) / 255.0, 2)
    grey = np.full_like(person_np, 128, np.uint8)
    overlay_np = (person_np * (1 - alpha) + grey * alpha).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay_np)
    
    # Apply transforms
    person_image = transform(person_pil)
    cloth_image = transform(cloth_pil)
    normal_map = transform(normal_pil)
    depth_map = depth_transform(depth_pil)
    mask_tensor = basic_transform(mask_pil)
    overlay_image = transform(overlay_pil)
    
    # Load caption
    caption = ""
    if os.path.isfile(caption_path):
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
    
    return {
        "person_image": person_image,
        "cloth_image": cloth_image,
        "normal_map": normal_map,
        "depth_map": depth_map,
        "mask": mask_tensor,
        "overlay_image": overlay_image,
        "caption": caption,
        "filename": base_name
    }


def load_test_batch(batch_size=2, device="cuda", dtype=torch.float16):
    """
    Load a batch of samples from the test dataset.
    
    Returns tensors ready for model input.
    """
    samples = []
    for i in range(batch_size):
        samples.append(load_test_sample(i))
    
    # Stack into batches
    batch = {
        "person_image": torch.stack([s["person_image"] for s in samples]).to(device, dtype=dtype),
        "cloth_image": torch.stack([s["cloth_image"] for s in samples]).to(device, dtype=dtype),
        "normal_map": torch.stack([s["normal_map"] for s in samples]).to(device, dtype=dtype),
        "depth_map": torch.stack([s["depth_map"] for s in samples]).to(device, dtype=dtype),
        "mask": torch.stack([s["mask"] for s in samples]).to(device, dtype=dtype),
        "overlay_image": torch.stack([s["overlay_image"] for s in samples]).to(device, dtype=dtype),
        "captions": [s["caption"] for s in samples],
        "filenames": [s["filename"] for s in samples]
    }
    
    return batch


def test_model_instantiation():
    """Test 1: Model instantiation and basic forward pass."""
    logger.info("=" * 60)
    logger.info("TEST 1: Model Instantiation")
    logger.info("=" * 60)
    
    try:
        from model import DeepFit
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Instantiate model with pixel-space ControlNet conditioning
        logger.info("Instantiating DeepFit model...")
        model = DeepFit(
            device=device,
            debug=False,
            unet_in_channels=4,
            unet_out_channels=4,
            controlnet_in_channels=4,
            controlnet_cond_channels=7  # Pixel space: 3 person RGB + 1 mask + 3 clothing RGB
        )
        model.to(device)
        model.eval()
        
        # Check components
        assert hasattr(model, 'vae'), "Missing VAE"
        assert hasattr(model, 'unet'), "Missing UNet"
        assert hasattr(model, 'controlnet'), "Missing ControlNet"
        assert hasattr(model, 'scheduler'), "Missing Scheduler"
        
        logger.info("✓ Model instantiation successful")
        logger.info(f"  - VAE: {type(model.vae).__name__}")
        logger.info(f"  - UNet: {type(model.unet).__name__}")
        logger.info(f"  - ControlNet: {type(model.controlnet).__name__}")
        logger.info(f"  - Scheduler: {type(model.scheduler).__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        
        return True, model
        
    except Exception as e:
        logger.error(f"✗ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(model=None):
    """Test 2: Forward pass with real dataset images."""
    logger.info("=" * 60)
    logger.info("TEST 2: Forward Pass (Real Data)")
    logger.info("=" * 60)
    
    try:
        from model import DeepFit
        from utils import prepare_control_input
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model is None:
            model = DeepFit(device=device, debug=False).to(device)
        model.eval()
        
        # Load real test sample
        logger.info("Loading real test sample from dataset...")
        sample = load_test_sample(0)
        
        # Add batch dimension
        overlay = sample["overlay_image"].unsqueeze(0).to(device, dtype=torch.float16)
        mask = sample["mask"].unsqueeze(0).to(device, dtype=torch.float16)
        cloth = sample["cloth_image"].unsqueeze(0).to(device, dtype=torch.float16)
        
        logger.info(f"  - Loaded sample: {sample['filename']}")
        logger.info(f"  - Caption: {sample['caption'][:80]}...")
        
        # Prepare control input in pixel space (no VAE encoding)
        ctrl = prepare_control_input(overlay, mask, cloth, debug=False)
        
        B = ctrl.shape[0]
        H, W = ctrl.shape[2], ctrl.shape[3]  # Pixel space dimensions (e.g., 512x512)
        h, w = H // 8, W // 8  # Latent space dimensions (e.g., 64x64)
        logger.info(f"  - Control input shape (pixel space): {ctrl.shape}")
        
        # Create noisy latents and prompt embeddings at latent resolution
        noisy_latents = torch.randn(B, 4, h, w, device=device, dtype=torch.float16)
        timesteps = torch.randint(0, 1000, (B,), device=device, dtype=torch.long)
        prompt_embeds = torch.randn(B, 77, 768, device=device, dtype=torch.float16)
        
        logger.info("Running forward pass...")
        with torch.no_grad():
            output = model(noisy_latents, timesteps, ctrl, prompt_embeds)
        
        # Verify output shape (latent space)
        expected_shape = (B, 4, h, w)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        logger.info(f"✓ Forward pass successful")
        logger.info(f"  - Input shape: {noisy_latents.shape}")
        logger.info(f"  - Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test 3: Single training step with real dataset images."""
    logger.info("=" * 60)
    logger.info("TEST 3: Training Step (Real Data)")
    logger.info("=" * 60)
    
    try:
        from model import DeepFit
        from utils import (
            prepare_control_input,
            prepare_target_latents,
            add_noise,
            setup_optimizer
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Instantiate model
        logger.info("Setting up model for training...")
        model = DeepFit(device=device, debug=False).to(device)
        model.train()
        
        optimizer = setup_optimizer(model, lr=1e-5)
        
        # Load real batch
        logger.info("Loading real training batch from dataset...")
        batch = load_test_batch(batch_size=2, device=device, dtype=torch.float16)
        
        logger.info(f"  - Loaded samples: {batch['filenames']}")
        logger.info(f"  - Image shape: {batch['overlay_image'].shape}")
        
        # Prepare inputs
        logger.info("Preparing control and target latents...")
        with torch.no_grad():
            ctrl = prepare_control_input(
                batch["overlay_image"], 
                batch["mask"], 
                batch["cloth_image"],
                debug=False
            )
            tgt = prepare_target_latents(
                batch["overlay_image"], 
                batch["depth_map"], 
                batch["normal_map"], 
                model.vae, 
                debug=False
            )
        
        # Create prompt embeddings (random for now, real encoding tested in inference)
        B = batch["overlay_image"].shape[0]
        prompt_embeds = torch.randn(B, 77, 768, device=device, dtype=torch.float16)
        
        # Add noise
        noised, noise, t = add_noise(tgt, debug=False)
        noised = noised.to(device, dtype=torch.float16)
        noise = noise.to(device, dtype=torch.float16)
        
        # Forward pass
        logger.info("Running training forward pass...")
        pred = model(noised, t, ctrl, prompt_embeds)
        
        # Compute loss
        loss = F.mse_loss(pred.float(), noise.float())
        
        # Backward pass
        logger.info("Running backward pass...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"✓ Training step successful")
        logger.info(f"  - Loss: {loss.item():.6f}")
        logger.info(f"  - Control input shape: {ctrl.shape}")
        logger.info(f"  - Target latents shape: {tgt.shape}")
        logger.info(f"  - Prediction shape: {pred.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test 4: Full inference pipeline with real dataset images."""
    logger.info("=" * 60)
    logger.info("TEST 4: Inference Pipeline (Real Data)")
    logger.info("=" * 60)
    
    try:
        from model import DeepFit
        from utils import encode_prompt, prepare_control_input, prepare_latents
        from transformers import CLIPTokenizer, CLIPTextModel
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Instantiate model
        logger.info("Setting up model for inference...")
        model = DeepFit(device=device, debug=False).to(device)
        model.eval()
        
        # Load tokenizer and text encoder
        logger.info("Loading text encoder...")
        tokenizer = CLIPTokenizer.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            subfolder="text_encoder",
            torch_dtype=torch.float16
        ).to(device)
        
        # Load real test sample
        logger.info("Loading real test sample from dataset...")
        sample = load_test_sample(0)
        
        # Add batch dimension
        overlay = sample["overlay_image"].unsqueeze(0).to(device, dtype=torch.float16)
        mask = sample["mask"].unsqueeze(0).to(device, dtype=torch.float16)
        cloth = sample["cloth_image"].unsqueeze(0).to(device, dtype=torch.float16)
        
        B = 1
        H, W = IMAGE_SIZE
        
        logger.info(f"  - Sample: {sample['filename']}")
        logger.info(f"  - Caption: {sample['caption'][:80]}...")
        
        # Encode real caption
        prompt = [sample["caption"]] if sample["caption"] else ["a person wearing clothes"]
        prompt_embeds = encode_prompt(
            model=model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=prompt,
            device=device
        )
        logger.info(f"  - Prompt embeddings shape: {prompt_embeds.shape}")
        
        # Prepare control input
        with torch.no_grad():
            control_input = prepare_control_input(overlay, mask, cloth, debug=False)
        logger.info(f"  - Control input shape (pixel space): {control_input.shape}")
        
        # Initialize latents
        latents = prepare_latents(B, H, W, device=device)
        logger.info(f"  - Initial latents shape: {latents.shape}")
        
        # Run a few denoising steps (not full inference for speed)
        num_steps = 10
        scheduler = model.scheduler
        scheduler.set_timesteps(num_steps)
        
        logger.info(f"Running {num_steps} denoising steps...")
        
        with torch.no_grad():
            for i, t in enumerate(scheduler.timesteps):
                timestep = torch.tensor([t] * B, device=device)
                
                # ControlNet
                down_block_res_samples, mid_block_res_sample = model.controlnet(
                    sample=latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=control_input,
                    conditioning_scale=1.0,
                    return_dict=False
                )
                
                # UNet
                noise_pred = model.unet(
                    sample=latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False
                )[0]
                
                # Scheduler step
                latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode
        logger.info("Decoding latents...")
        latents_scaled = latents / model.vae.config.scaling_factor
        with torch.no_grad():
            decoded = model.vae.decode(latents_scaled, return_dict=False)[0]
        
        # Verify output
        expected_shape = (B, 3, H, W)
        assert decoded.shape == expected_shape, f"Unexpected output shape: {decoded.shape}"
        
        # Save output image for visual verification
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to PIL and save
        output_image = ((decoded[0].cpu().float().clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
        output_image = output_image.permute(1, 2, 0).numpy()
        output_pil = Image.fromarray(output_image)
        output_path = os.path.join(output_dir, f"inference_test_{sample['filename']}.png")
        output_pil.save(output_path)
        
        logger.info(f"✓ Inference pipeline successful")
        logger.info(f"  - Final latents shape: {latents.shape}")
        logger.info(f"  - Decoded image shape: {decoded.shape}")
        logger.info(f"  - Output saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Inference pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distillation():
    """Test 5: Distillation components with real dataset images."""
    logger.info("=" * 60)
    logger.info("TEST 5: Distillation Components (Real Data)")
    logger.info("=" * 60)
    
    try:
        from model import DeepFit
        from utils import prepare_control_input
        from distillation.distill_utils import (
            LatentDiscriminator,
            ConsistencySampler,
            LADDDistillationWrapper
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Instantiate student and teacher
        logger.info("Instantiating student and teacher models...")
        student = DeepFit(
            device=device,
            debug=False,
            unet_in_channels=4,
            unet_out_channels=4,
            controlnet_in_channels=4,
            controlnet_cond_channels=7  # Pixel space: 3 person RGB + 1 mask + 3 clothing RGB
        ).to(device)
        
        teacher = DeepFit(
            device=device,
            debug=False,
            unet_in_channels=4,
            unet_out_channels=4,
            controlnet_in_channels=4,
            controlnet_cond_channels=7
        ).to(device)
        teacher.load_state_dict(student.state_dict())
        teacher.eval()
        
        logger.info("✓ Student and teacher instantiated")
        
        # Test LatentDiscriminator
        logger.info("Testing LatentDiscriminator...")
        discriminator = LatentDiscriminator(teacher).to(device)
        logger.info(f"  - Feature layers: {discriminator.feature_layers}")
        logger.info("✓ LatentDiscriminator created")
        
        # Test ConsistencySampler
        logger.info("Testing ConsistencySampler...")
        sampler = ConsistencySampler(student)
        logger.info(f"  - Sigma range: [{sampler.sigma_min}, {sampler.sigma_max}]")
        logger.info("✓ ConsistencySampler created")
        
        # Test LADDDistillationWrapper
        logger.info("Testing LADDDistillationWrapper...")
        wrapper = LADDDistillationWrapper(
            student_model=student,
            teacher_model=teacher
        )
        
        # Check wrapper components
        assert hasattr(wrapper, 'student'), "Missing student"
        assert hasattr(wrapper, 'teacher'), "Missing teacher"
        assert hasattr(wrapper, 'latent_discriminator'), "Missing discriminator"
        assert hasattr(wrapper, 'consistency_sampler'), "Missing consistency sampler"
        
        logger.info("✓ LADDDistillationWrapper created")
        logger.info(f"  - Lambda MSE: {wrapper.lambda_mse}")
        logger.info(f"  - Lambda ADV: {wrapper.lambda_adv}")
        logger.info(f"  - Lambda Consistency: {wrapper.lambda_consistency}")
        
        # Load real test sample for discriminator test
        logger.info("Loading real test sample for discriminator test...")
        sample = load_test_sample(0)
        
        overlay = sample["overlay_image"].unsqueeze(0).to(device, dtype=torch.float16)
        mask = sample["mask"].unsqueeze(0).to(device, dtype=torch.float16)
        cloth = sample["cloth_image"].unsqueeze(0).to(device, dtype=torch.float16)
        
        # Prepare control input in pixel space (no VAE encoding)
        control_input = prepare_control_input(overlay, mask, cloth, debug=False)
        
        B = control_input.shape[0]
        H, W = control_input.shape[2], control_input.shape[3]  # Pixel space
        h, w = H // 8, W // 8  # Latent space
        noisy_latents = torch.randn(B, 4, h, w, device=device, dtype=torch.float16)
        prompt_embeds = torch.randn(B, 77, 768, device=device, dtype=torch.float16)
        
        logger.info("Testing discriminator forward pass with real data...")
        with torch.no_grad():
            scores = discriminator(noisy_latents, prompt_embeds, control_input)
        
        assert scores.shape == (B, 1), f"Unexpected discriminator output shape: {scores.shape}"
        logger.info(f"✓ Discriminator forward pass successful")
        logger.info(f"  - Input sample: {sample['filename']}")
        logger.info(f"  - Control input shape: {control_input.shape}")
        logger.info(f"  - Output shape: {scores.shape}")
        logger.info(f"  - Score: {scores.item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Distillation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test 6: Utility functions."""
    logger.info("=" * 60)
    logger.info("TEST 6: Utility Functions")
    logger.info("=" * 60)
    
    try:
        from utils import (
            seed_everything,
            prepare_latents,
            add_noise,
            modify_unet_channels,
            modify_controlnet_channels
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Test seed_everything
        logger.info("Testing seed_everything...")
        seed_everything(42)
        logger.info("✓ seed_everything works")
        
        # Test prepare_latents
        logger.info("Testing prepare_latents...")
        latents = prepare_latents(2, 64, 64, device=device)
        assert latents.shape == (2, 4, 8, 8), f"Unexpected shape: {latents.shape}"
        logger.info(f"✓ prepare_latents works: {latents.shape}")
        
        # Test add_noise
        logger.info("Testing add_noise...")
        target = torch.randn(2, 4, 8, 8, device=device)
        noisy, noise, timesteps = add_noise(target)
        assert noisy.shape == target.shape
        assert noise.shape == target.shape
        assert timesteps.shape == (2,)
        logger.info(f"✓ add_noise works")
        logger.info(f"  - Noisy shape: {noisy.shape}")
        logger.info(f"  - Timesteps: {timesteps.tolist()}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Utility functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """Test 7: Dataloader with real dataset."""
    logger.info("=" * 60)
    logger.info("TEST 7: Dataloader (Real Dataset)")
    logger.info("=" * 60)
    
    try:
        from torch.utils.data import DataLoader, Dataset
        
        # Create a simple test dataset that matches our folder structure
        # Our dataset has: image/*_0.jpg, cloth/*_1.jpg, mask/*_0.png, etc.
        logger.info(f"Testing custom TestDataset with: {DATASET_ROOT}")
        
        if not os.path.isdir(DATASET_ROOT):
            raise FileNotFoundError(f"Dataset not found at {DATASET_ROOT}")
        
        class TestDataset(Dataset):
            """Test dataset matching our specific folder structure."""
            def __init__(self, root_dir):
                self.root_dir = root_dir
                image_dir = os.path.join(root_dir, "image")
                # Get all _0 images (person images)
                self.samples = sorted([
                    f.replace("_0.jpg", "") 
                    for f in os.listdir(image_dir) 
                    if f.endswith("_0.jpg")
                ])
                logger.info(f"  - Found {len(self.samples)} samples")
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                base = self.samples[idx]
                
                # Load images with correct naming convention
                person_pil = Image.open(os.path.join(self.root_dir, "image", f"{base}_0.jpg")).convert("RGB")
                cloth_pil = Image.open(os.path.join(self.root_dir, "cloth", f"{base}_1.jpg")).convert("RGB")
                normal_pil = Image.open(os.path.join(self.root_dir, "normal", f"{base}_0.jpg")).convert("RGB")
                depth_pil = Image.open(os.path.join(self.root_dir, "depth", f"{base}_0.jpg")).convert("L")
                mask_pil = Image.open(os.path.join(self.root_dir, "mask", f"{base}_0.png")).convert("L")
                
                # Create overlay
                person_np = np.array(person_pil.resize(IMAGE_SIZE))
                mask_np = np.array(mask_pil.resize(IMAGE_SIZE))
                blurred = cv2.GaussianBlur(mask_np, (21, 21), sigmaX=10)
                alpha = np.expand_dims(blurred.astype(np.float32) / 255.0, 2)
                grey = np.full_like(person_np, 128, np.uint8)
                overlay_np = (person_np * (1 - alpha) + grey * alpha).astype(np.uint8)
                overlay_pil = Image.fromarray(overlay_np)
                
                # Load caption
                caption_path = os.path.join(self.root_dir, "caption", f"{base}_0.txt")
                caption = ""
                if os.path.isfile(caption_path):
                    with open(caption_path, "r", encoding="utf-8") as f:
                        caption = f.read().strip()
                
                return {
                    "person_image": transform(person_pil),
                    "cloth_image": transform(cloth_pil),
                    "normal_map": transform(normal_pil),
                    "depth_map": depth_transform(depth_pil),
                    "mask": basic_transform(mask_pil),
                    "overlay_image": transform(overlay_pil),
                    "caption": caption,
                    "filename": base
                }
        
        dataset = TestDataset(DATASET_ROOT)
        logger.info(f"  - Dataset size: {len(dataset)} samples")
        
        # Test getting a single sample
        logger.info("Testing single sample retrieval...")
        sample = dataset[0]
        
        # Verify sample structure
        expected_keys = ["person_image", "cloth_image", "normal_map", "depth_map", 
                         "mask", "overlay_image", "caption", "filename"]
        for key in expected_keys:
            assert key in sample, f"Missing key: {key}"
        
        logger.info("  Sample contents:")
        logger.info(f"    - filename: {sample['filename']}")
        logger.info(f"    - person_image: {sample['person_image'].shape}")
        logger.info(f"    - cloth_image: {sample['cloth_image'].shape}")
        logger.info(f"    - normal_map: {sample['normal_map'].shape}")
        logger.info(f"    - depth_map: {sample['depth_map'].shape}")
        logger.info(f"    - mask: {sample['mask'].shape}")
        logger.info(f"    - overlay_image: {sample['overlay_image'].shape}")
        logger.info(f"    - caption: {sample['caption'][:60]}...")
        
        # Verify tensor shapes
        assert sample["person_image"].shape == (3, 512, 512), f"Unexpected person_image shape"
        assert sample["cloth_image"].shape == (3, 512, 512), f"Unexpected cloth_image shape"
        assert sample["mask"].shape == (1, 512, 512), f"Unexpected mask shape"
        
        # Test DataLoader
        logger.info("Testing DataLoader batch loading...")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        batch = next(iter(dataloader))
        logger.info(f"  - Batch size: {batch['person_image'].shape[0]}")
        logger.info(f"  - Person image batch: {batch['person_image'].shape}")
        logger.info(f"  - Cloth image batch: {batch['cloth_image'].shape}")
        
        # Verify batch shapes
        assert batch["person_image"].shape == (2, 3, 512, 512), f"Unexpected batch shape"
        
        logger.info("✓ Dataloader test successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test 8: Test the load_test_sample and load_test_batch functions."""
    logger.info("=" * 60)
    logger.info("TEST 8: Dataset Loading Functions")
    logger.info("=" * 60)
    
    try:
        # Test load_test_sample
        logger.info("Testing load_test_sample...")
        sample = load_test_sample(0)
        
        expected_keys = ["person_image", "cloth_image", "normal_map", "depth_map", 
                         "mask", "overlay_image", "caption", "filename"]
        for key in expected_keys:
            assert key in sample, f"Missing key: {key}"
        
        logger.info(f"  - Sample loaded: {sample['filename']}")
        logger.info(f"  - Person image: {sample['person_image'].shape}")
        logger.info(f"  - Cloth image: {sample['cloth_image'].shape}")
        logger.info(f"  - Mask: {sample['mask'].shape}")
        logger.info(f"  - Caption: {sample['caption'][:60]}...")
        
        # Test load_test_batch
        logger.info("Testing load_test_batch...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch = load_test_batch(batch_size=3, device=device, dtype=torch.float16)
        
        assert batch["person_image"].shape[0] == 3, "Batch size mismatch"
        logger.info(f"  - Batch loaded with {len(batch['filenames'])} samples")
        logger.info(f"  - Filenames: {batch['filenames']}")
        logger.info(f"  - Person image batch: {batch['person_image'].shape}")
        logger.info(f"  - Device: {batch['person_image'].device}")
        logger.info(f"  - Dtype: {batch['person_image'].dtype}")
        
        # List all available samples
        image_dir = os.path.join(DATASET_ROOT, "image")
        person_files = sorted([f for f in os.listdir(image_dir) if f.endswith("_0.jpg")])
        logger.info(f"  - Total samples available: {len(person_files)}")
        
        logger.info("✓ Dataset loading functions test successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Dataset loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    logger.info("\n" + "=" * 60)
    logger.info("DEEPFIT PIPELINE TEST SUITE (with Real Dataset)")
    logger.info("=" * 60)
    logger.info(f"Dataset path: {DATASET_ROOT}\n")
    
    results = {}
    
    # Test 8: Dataset loading (run first to validate dataset)
    results["Dataset Loading"] = test_dataset_loading()
    print()
    
    # Test 7: Dataloader
    results["Dataloader"] = test_dataloader()
    print()
    
    # Test 1: Model instantiation
    success, model = test_model_instantiation()
    results["Model Instantiation"] = success
    print()
    
    # Test 2: Forward pass
    if success:
        results["Forward Pass"] = test_forward_pass(model)
    else:
        results["Forward Pass"] = False
        logger.warning("Skipping forward pass test (model failed)")
    print()
    
    # Test 3: Training step
    results["Training Step"] = test_training_step()
    print()
    
    # Test 4: Inference
    results["Inference"] = test_inference()
    print()
    
    # Test 5: Distillation
    results["Distillation"] = test_distillation()
    print()
    
    # Test 6: Utilities
    results["Utilities"] = test_utils()
    print()
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info("-" * 60)
    logger.info(f"  Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed!")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed")
        return 1


def parse_args():
    parser = argparse.ArgumentParser(description="Test DeepFit pipeline with real dataset")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "model", "forward", "train", "inference", "distill", "utils", "dataloader", "dataset"],
        help="Which test to run"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset folder (optional)")
    return parser.parse_args()


def main():
    global DATASET_ROOT
    args = parse_args()
    
    # Override dataset path if provided
    if args.dataset:
        DATASET_ROOT = args.dataset
        logger.info(f"Using custom dataset path: {DATASET_ROOT}")
    
    if args.test == "all":
        return run_all_tests()
    elif args.test == "model":
        success, _ = test_model_instantiation()
        return 0 if success else 1
    elif args.test == "forward":
        return 0 if test_forward_pass() else 1
    elif args.test == "train":
        return 0 if test_training_step() else 1
    elif args.test == "inference":
        return 0 if test_inference() else 1
    elif args.test == "distill":
        return 0 if test_distillation() else 1
    elif args.test == "utils":
        return 0 if test_utils() else 1
    elif args.test == "dataloader":
        return 0 if test_dataloader() else 1
    elif args.test == "dataset":
        return 0 if test_dataset_loading() else 1


if __name__ == "__main__":
    sys.exit(main())
