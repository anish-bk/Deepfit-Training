#!/usr/bin/env python
"""
test_pipeline.py

Comprehensive test script to verify the entire DeepFit pipeline:
1. Model instantiation
2. Forward pass
3. Training step
4. Inference
5. Distillation

Usage:
    python test_pipeline.py --test all
    python test_pipeline.py --test model
    python test_pipeline.py --test train
    python test_pipeline.py --test inference
    python test_pipeline.py --test distill
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import tempfile
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_model_instantiation():
    """Test 1: Model instantiation and basic forward pass."""
    logger.info("=" * 60)
    logger.info("TEST 1: Model Instantiation")
    logger.info("=" * 60)
    
    try:
        from model import DeepFit
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Instantiate model
        logger.info("Instantiating DeepFit model...")
        model = DeepFit(
            device=device,
            debug=False,
            unet_in_channels=13,
            unet_out_channels=4,
            controlnet_in_channels=13,
            controlnet_cond_channels=9
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
    """Test 2: Forward pass with dummy data."""
    logger.info("=" * 60)
    logger.info("TEST 2: Forward Pass")
    logger.info("=" * 60)
    
    try:
        from model import DeepFit
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model is None:
            model = DeepFit(device=device, debug=False).to(device)
        model.eval()
        
        # Create dummy inputs
        B = 1
        H, W = 64, 64  # Small size for testing
        h, w = H // 8, W // 8  # Latent size
        
        logger.info(f"Creating dummy inputs: batch={B}, latent_size={h}x{w}")
        
        # Dummy tensors
        noisy_latents = torch.randn(B, 4, h, w, device=device, dtype=torch.float16)
        timesteps = torch.randint(0, 1000, (B,), device=device, dtype=torch.long)
        control_input = torch.randn(B, 9, h, w, device=device, dtype=torch.float16)
        prompt_embeds = torch.randn(B, 77, 768, device=device, dtype=torch.float16)
        
        logger.info("Running forward pass...")
        with torch.no_grad():
            output = model(noisy_latents, timesteps, control_input, prompt_embeds)
        
        # Verify output shape
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
    """Test 3: Single training step with dummy data."""
    logger.info("=" * 60)
    logger.info("TEST 3: Training Step")
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
        
        # Create dummy batch
        B = 1
        H, W = 64, 64  # Small size for testing
        
        logger.info(f"Creating dummy training batch: {B}x3x{H}x{W}")
        
        # Dummy images (normalized to [-1, 1])
        overlay = torch.randn(B, 3, H, W, device=device, dtype=torch.float16)
        mask = torch.rand(B, 1, H, W, device=device, dtype=torch.float16)
        cloth = torch.randn(B, 3, H, W, device=device, dtype=torch.float16)
        depth = torch.randn(B, 1, H, W, device=device, dtype=torch.float16)
        normal = torch.randn(B, 3, H, W, device=device, dtype=torch.float16)
        prompt_embeds = torch.randn(B, 77, 768, device=device, dtype=torch.float16)
        
        # Prepare inputs
        logger.info("Preparing control and target latents...")
        with torch.no_grad():
            ctrl = prepare_control_input(overlay, mask, cloth, model.vae, debug=False)
            tgt = prepare_target_latents(overlay, depth, normal, model.vae, debug=False)
        
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
    """Test 4: Full inference pipeline."""
    logger.info("=" * 60)
    logger.info("TEST 4: Inference Pipeline")
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
        
        # Create dummy inputs
        B = 1
        H, W = 64, 64  # Small size for testing
        
        logger.info(f"Creating dummy inference inputs...")
        
        person = torch.rand(B, 3, H, W, device=device, dtype=torch.float16)
        mask = torch.rand(B, 1, H, W, device=device, dtype=torch.float16)
        clothing = torch.rand(B, 3, H, W, device=device, dtype=torch.float16)
        
        # Encode prompt
        prompt = ["a person wearing clothes"]
        prompt_embeds = encode_prompt(
            model=model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=prompt,
            device=device
        )
        
        # Prepare control input
        with torch.no_grad():
            control_input = prepare_control_input(person, mask, clothing, model.vae, debug=False)
        
        # Initialize latents
        latents = prepare_latents(B, H, W, device=device)
        
        # Run a few denoising steps (not full inference for speed)
        num_steps = 5
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
        assert decoded.shape == (B, 3, H, W), f"Unexpected output shape: {decoded.shape}"
        
        logger.info(f"✓ Inference pipeline successful")
        logger.info(f"  - Final latents shape: {latents.shape}")
        logger.info(f"  - Decoded image shape: {decoded.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Inference pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distillation():
    """Test 5: Distillation components."""
    logger.info("=" * 60)
    logger.info("TEST 5: Distillation Components")
    logger.info("=" * 60)
    
    try:
        from model import DeepFit
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
            unet_in_channels=13,
            unet_out_channels=4,
            controlnet_in_channels=13,
            controlnet_cond_channels=9
        ).to(device)
        
        teacher = DeepFit(
            device=device,
            debug=False,
            unet_in_channels=13,
            unet_out_channels=4,
            controlnet_in_channels=13,
            controlnet_cond_channels=9
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
        
        # Test forward pass through discriminator
        logger.info("Testing discriminator forward pass...")
        B = 1
        h, w = 8, 8
        noisy_latents = torch.randn(B, 4, h, w, device=device, dtype=torch.float16)
        prompt_embeds = torch.randn(B, 77, 768, device=device, dtype=torch.float16)
        control_input = torch.randn(B, 9, h, w, device=device, dtype=torch.float16)
        
        with torch.no_grad():
            scores = discriminator(noisy_latents, prompt_embeds, control_input)
        
        assert scores.shape == (B, 1), f"Unexpected discriminator output shape: {scores.shape}"
        logger.info(f"✓ Discriminator forward pass successful")
        logger.info(f"  - Output shape: {scores.shape}")
        
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


def run_all_tests():
    """Run all tests and report results."""
    logger.info("\n" + "=" * 60)
    logger.info("DEEPFIT PIPELINE TEST SUITE")
    logger.info("=" * 60 + "\n")
    
    results = {}
    
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
    parser = argparse.ArgumentParser(description="Test DeepFit pipeline")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "model", "forward", "train", "inference", "distill", "utils"],
        help="Which test to run"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def main():
    args = parse_args()
    
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


if __name__ == "__main__":
    sys.exit(main())
