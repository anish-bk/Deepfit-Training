# distill_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from model import DeepFit

from ema_pytorch import EMA

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LatentDiscriminator(nn.Module):
    """
    Latent-space discriminator that extracts features from the UNet
    of the teacher model, applies projection heads, and produces a scalar score per sample.
    For SD1.5, we use the UNet output directly as features.
    """
    def __init__(self, teacher: DeepFit, feature_layers: list = None, proj_hidden_dim: int = 128):
        """
        teacher: the teacher DeepFit model whose UNet we extract features from.
        feature_layers: indices of layers to extract features from. For SD1.5 UNet, we use [0] by default.
        proj_hidden_dim: hidden dimension for projection heads.
        """
        super().__init__()
        self.teacher = teacher
        # For SD1.5 UNet, we use the output as a single feature layer
        # Select feature layers - for SD1.5 we just use the output
        if feature_layers is None:
            feature_layers = [0]  # Just use the UNet output
        self.feature_layers = feature_layers

        self.proj_hidden_dim = proj_hidden_dim
        # Projection heads will be created on first forward based on feature shapes
        self.proj_heads = nn.ModuleDict()
        self.initialized = False

    def _init_heads(self, sample_latent: torch.Tensor,
                    prompt_embeds: torch.Tensor, control_input: torch.Tensor):
        """
        Initialize projection heads by doing a forward pass through teacher.unet
        to inspect feature shapes.
        sample_latent: [B, C_latent, H, W], e.g. [B,4,h/8,w/8] for SD1.5
        prompt_embeds: [B, seq_len, dim]
        control_input: [B, cond_C, h/8, w/8]
        """
        device = sample_latent.device
        B = sample_latent.shape[0]
        # Create dummy timesteps tensor (zeros)
        t_dummy = torch.zeros(B, device=device, dtype=torch.long)
        # Forward through teacher.unet
        try:
            # Get ControlNet outputs first
            down_block_res_samples, mid_block_res_sample = self.teacher.controlnet(
                sample=sample_latent,
                timestep=t_dummy,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=control_input,
                conditioning_scale=1.0,
                return_dict=False
            )
            # UNet forward
            outputs = self.teacher.unet(
                sample=sample_latent,
                timestep=t_dummy,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False
            )
            # Use the output as feature for discriminator
            hidden_states = [outputs[0]]  # [B, C, H, W]
        except Exception as e:
            raise RuntimeError(f"UNet forward failed: {e}")

        # For each selected layer, inspect shape and create projection head
        for idx in self.feature_layers:
            if idx >= len(hidden_states):
                # Skip if layer index exceeds available hidden states
                continue
            feat = hidden_states[idx]  # [B, C, H, W]
            C = feat.shape[1]
            # Projection head: global average pooling -> Linear(C, proj_hidden_dim) -> LeakyReLU -> Linear(proj_hidden_dim,1)
            head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),      # [B, C, 1, 1]
                nn.Flatten(),                 # [B, C]
                nn.Linear(C, self.proj_hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.proj_hidden_dim, 1)
            )
            # Match the dtype of the input features (e.g., float16)
            self.proj_heads[str(idx)] = head.to(device=device, dtype=feat.dtype)
        self.initialized = True
        logger.info(f"[LatentDiscriminator] Initialized projection heads for layers {self.feature_layers}")

    def forward(self, noisy_latents: torch.Tensor,
                prompt_embeds: torch.Tensor, control_input: torch.Tensor, pooled_prompt: torch.Tensor = None):
        """
        noisy_latents: [B, C_latent, H, W], e.g. [B,4,h/8,w/8] for SD1.5
        prompt_embeds: [B, seq_len, dim]
        control_input: [B, cond_C, h/8, w/8]
        pooled_prompt: Not used for SD1.5 but kept for compatibility
        Returns: scores: [B, 1], discriminator score per sample.
        """
        if not self.initialized:
            # Initialize projection heads on first batch
            self._init_heads(noisy_latents, prompt_embeds, control_input)

        # Forward through teacher.unet to get hidden states
        with torch.no_grad():
            B = noisy_latents.shape[0]
            t_dummy = torch.zeros(B, device=noisy_latents.device, dtype=torch.long)
            try:
                # Get ControlNet outputs first
                down_block_res_samples, mid_block_res_sample = self.teacher.controlnet(
                    sample=noisy_latents,
                    timestep=t_dummy,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=control_input,
                    conditioning_scale=1.0,
                    return_dict=False
                )
                # UNet forward
                outputs = self.teacher.unet(
                    sample=noisy_latents,
                    timestep=t_dummy,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False
                )
                # Use the output as hidden states for discriminator
                hidden_states = [outputs[0]]
            except Exception as e:
                raise RuntimeError(f"UNet forward failed: {e}")

        # For each selected layer, apply projection head
        scores = []
        for idx in self.feature_layers:
            if idx < len(hidden_states):
                feat = hidden_states[idx]  # [B, C, H, W]
                head = self.proj_heads[str(idx)]
                out = head(feat)  # [B,1]
                scores.append(out)
        # Aggregate: mean across heads
        if len(scores) > 0:
            scores = torch.stack(scores, dim=0)  # [num_layers, B,1]
            scores = scores.mean(dim=0)          # [B,1]
        else:
            scores = torch.zeros(noisy_latents.shape[0], 1, device=noisy_latents.device)
        return scores  # [B,1]


class ConsistencySampler:
    """
    Samples noisy latents at sigma_max, passes through student UNet to get predictions,
    used for consistency loss.
    Works on latents, e.g., 4 channels for SD1.5.
    """
    def __init__(self, model: DeepFit, sigma_min: float = 0.002, sigma_max: float = 80.0):
        """
        model: the student DeepFit instance
        sigma_min, sigma_max: as in LADD definitions
        """
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, latents: torch.Tensor, prompt_embeds: torch.Tensor,
                 control_input: torch.Tensor, pooled_prompt: torch.Tensor = None) -> torch.Tensor:
        """
        latents: [B, 4, h/8, w/8] latents for SD1.5
        prompt_embeds: [B, seq_len, dim]
        control_input: [B, cond_C, h/8, w/8]
        pooled_prompt: Not used for SD1.5
        Returns predicted noise for latents at sigma_max: [B, 4, h/8, w/8]
        """
        noisy_latents = latents + self.sigma_max * torch.randn_like(latents)
        with torch.no_grad():
            # ControlNet forward
            down_block_res_samples, mid_block_res_sample = self.model.controlnet(
                sample=noisy_latents,
                timestep=torch.tensor([int(self.sigma_max)] * noisy_latents.shape[0], device=latents.device, dtype=torch.long),
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=control_input,
                conditioning_scale=1.0,
                return_dict=False
            )
            # UNet forward
            pred = self.model.unet(
                sample=noisy_latents,
                timestep=torch.tensor([int(self.sigma_max)] * noisy_latents.shape[0], device=latents.device, dtype=torch.long),
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False
            )[0]
        return pred


class LADDDistillationWrapper(nn.Module):
    """
    LADD Distillation wrapper for DeepFit models (20-channel latents):
    - student_model: DeepFit instance (20-channel config).
    - teacher_model: DeepFit instance; if provided, use it; else clone student weights.
    - Uses LatentDiscriminator to provide adversarial loss based on teacher’s own features.
    - Uses consistency via EMA(student).
    """
    def __init__(self, student_model: DeepFit, teacher_model: DeepFit = None):
        super().__init__()
        if student_model is None:
            raise ValueError("Student model must be provided for LADD distillation.")
        self.student = student_model

        # Teacher: either provided or copy of student
        if teacher_model is None:
            teacher = DeepFit(
                device=self.student.device,
                debug=self.student.debug,
                unet_in_channels=13,
                unet_out_channels=4,
                controlnet_in_channels=13,
                controlnet_cond_channels=9
            ).to(self.student.device)
            teacher.load_state_dict(self.student.state_dict())
            teacher.eval()
            self.teacher = teacher
        else:
            self.teacher = teacher_model
            self.teacher.eval()

        # Latent discriminator using teacher features
        self.latent_discriminator = LatentDiscriminator(self.teacher)

        # Consistency sampler
        self.consistency_sampler = ConsistencySampler(self.student)

        # Optimizers: placeholders; override externally if needed
        self.student_optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4
        )
        self.discriminator_optimizer = torch.optim.AdamW(
            self.latent_discriminator.parameters(),
            lr=2e-4, betas=(0.5, 0.999)
        )

        # EMA for student
        self.ema_student = EMA(self.student, beta=0.999, update_every=10)

        # Loss weights
        self.lambda_mse = 1.0
        self.lambda_adv = 0.5
        self.lambda_consistency = 0.3
        self.lambda_gp = 10.0

    def compute_losses(self, batch: dict, step: int):
        """
        Compute LADD losses for one batch:
        - MSE between student_pred and teacher_pred
        - Adversarial loss (student_pred vs teacher_pred) via latent discriminator
        - Consistency loss (student_pred vs EMA prediction)
        Returns:
            total_loss, loss_mse, loss_adv, loss_consistency,
            teacher_pred (detached), student_pred (for use in discriminator step), noisy_latents,
            prompt_embeds, pooled_prompt, control_input
        """
        # Move tensors to device
        batch = {k: v.to(self.student.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        person_images = batch["person_image"]       # [B,3,H,W]
        masks = batch["mask"]                      # [B,1,H,W]
        clothing_images = batch["clothing_image"]  # [B,3,H,W]
        tryon_gt = batch["tryon_gt"]               # [B,3,H,W]
        depth_gt = batch["depth_gt"]               # [B,1,H,W]
        normal_gt = batch["normal_gt"]             # [B,3,H,W]
        prompts = batch["prompt"]                  # list[str], length B
        B = person_images.shape[0]

        # 1. Encode prompt using student._encode_prompt
        prompt_embeds, pooled_prompt = self.student._encode_prompt(prompts)
        # 2. Prepare control input
        control_input = self.student._prepare_control_input(person_images, masks, clothing_images)
        # 3. Prepare target latents [B,20,h/8,w/8]
        target_latents = self.student._prepare_target_latents(tryon_gt, depth_gt, normal_gt)
        # 4. Add noise
        timesteps = torch.rand(B, device=self.student.device)  # [B]
        noise = torch.randn_like(target_latents)              # [B,20,h/8,w/8]
        noisy_latents = target_latents + noise * timesteps[:, None, None, None]

        # 5. Student forward
        control_block_s = self.student.controlnet(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt,
            controlnet_cond=control_input,
            conditioning_scale=1.0,
            return_dict=False
        )[0]
        student_pred = self.student.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt,
            block_controlnet_hidden_states=control_block_s,
            return_dict=False
        )[0]  # [B,20,h/8,w/8]

        # 6. Teacher forward (no grad)
        with torch.no_grad():
            control_block_t = self.teacher.controlnet(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt,
                controlnet_cond=control_input,
                conditioning_scale=1.0,
                return_dict=False
            )[0]
            teacher_pred = self.teacher.transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt,
                block_controlnet_hidden_states=control_block_t,
                return_dict=False
            )[0]  # [B,20,h/8,w/8]

        # 7. MSE loss
        loss_mse = F.mse_loss(student_pred.float(), teacher_pred.float())

        # 8. Adversarial loss: encourage student_pred to match teacher_pred latent features
        # Latent discriminator: input = predicted latent/noise? Here we pass the predicted residual/noise latent: teacher_pred vs student_pred
        real_scores = self.latent_discriminator(teacher_pred.detach(), prompt_embeds, pooled_prompt, control_input)  # [B,1]
        fake_scores = self.latent_discriminator(student_pred, prompt_embeds, pooled_prompt, control_input)          # [B,1]
        # Generator adv loss: -E[D(student_pred)]
        loss_adv = -fake_scores.mean()

        # 9. Consistency loss via EMA
        with torch.no_grad():
            ema_pred = self.consistency_sampler(target_latents, prompt_embeds, pooled_prompt, control_input)
        loss_consistency = F.mse_loss(student_pred.float(), ema_pred.float())

        total_loss = self.lambda_mse * loss_mse + self.lambda_adv * loss_adv + self.lambda_consistency * loss_consistency

        return total_loss, loss_mse, loss_adv, loss_consistency, teacher_pred.detach(), student_pred.detach(), noisy_latents, prompt_embeds, pooled_prompt, control_input

    def train_step(self, batch: dict, step: int) -> dict:
        """
        One distillation training step:
        - Update student (MSE + adv + consistency)
        - Update latent discriminator (WGAN-GP) on teacher_pred vs student_pred.detach()
        """
        # Compute losses and needed tensors
        (total_loss, loss_mse, loss_adv, loss_consistency,
         teacher_pred, student_pred_detached, noisy_latents,
         prompt_embeds, pooled_prompt, control_input) = self.compute_losses(batch, step)

        # 1. Student update
        self.student_optimizer.zero_grad()
        total_loss.backward()
        self.student_optimizer.step()
        # EMA update
        self.ema_student.update()

        # 2. Discriminator update: WGAN-GP between teacher_pred (real) and student_pred_detached (fake)
        real_latents = teacher_pred  # [B,20,h/8,w/8]
        fake_latents = student_pred_detached  # [B,20,h/8,w/8]

        # Gradient penalty: interpolate in latent space
        alpha = torch.rand(real_latents.size(0), 1, 1, 1, device=real_latents.device)
        interpolates = (alpha * real_latents + (1 - alpha) * fake_latents).requires_grad_(True)
        # Discriminator score on interpolates
        d_interpolates = self.latent_discriminator(interpolates, prompt_embeds, pooled_prompt, control_input)  # [B,1]
        # Compute gradients wrt interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates.sum(),
            inputs=interpolates,
            create_graph=True, retain_graph=True
        )[0]  # [B,20,h/8,w/8]
        gradients = gradients.view(real_latents.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        # Discriminator losses
        real_scores = self.latent_discriminator(real_latents, prompt_embeds, pooled_prompt, control_input)  # [B,1]
        fake_scores = self.latent_discriminator(fake_latents, prompt_embeds, pooled_prompt, control_input)  # [B,1]
        # WGAN-GP discriminator loss: E[fake] - E[real] + λ_gp * penalty
        d_loss = fake_scores.mean() - real_scores.mean() + self.lambda_gp * gradient_penalty

        self.discriminator_optimizer.zero_grad()
        d_loss.backward()
        self.discriminator_optimizer.step()

        logs = {
            "loss/mse": loss_mse.item(),
            "loss/adv": loss_adv.item(),
            "loss/consistency": loss_consistency.item(),
            "loss/discriminator": d_loss.item(),
            "loss/total": total_loss.item()
        }
        if step % 10 == 0:
            logger.info(f"[Distill] Step {step} | MSE: {loss_mse.item():.4f} | Adv: {loss_adv.item():.4f} | "
                        f"Consistency: {loss_consistency.item():.4f} | D_loss: {d_loss.item():.4f}")
        return logs

    def infer(self, person_image: torch.Tensor, mask: torch.Tensor, clothing_image: torch.Tensor,
              prompt: str, height: int = 1024, width: int = 1024,
              guidance_scale: float = 7.0, num_inference_steps: int = 28,
              use_ema: bool = True) -> "PIL.Image":
        """
        Inference with distilled student model (20-channel training). Generates only the image output.
        person_image: [3,H,W] or [1,3,H,W]
        mask: [1,H,W] or [B,1,H,W]
        clothing_image: [3,H,W] or [1,3,H,W]
        prompt: str
        Returns PIL.Image of generated tryon image.
        Note: depth/normal channels are training-only; inference generates only image via VAE decode.
        """
        from PIL import Image
        model = self.ema_student.module if use_ema else self.student
        model.eval()
        device = model.device

        # Prepare batch size 1
        if person_image.ndim == 3:
            person = person_image.unsqueeze(0).to(device, dtype=torch.float16)
        else:
            person = person_image.to(device, dtype=torch.float16)
        if mask.ndim == 3:
            m = mask.unsqueeze(0).to(device, dtype=torch.float16)
        else:
            m = mask.to(device, dtype=torch.float16)
        if clothing_image.ndim == 3:
            cloth = clothing_image.unsqueeze(0).to(device, dtype=torch.float16)
        else:
            cloth = clothing_image.to(device, dtype=torch.float16)
        prompts = [prompt]

        # Encode prompt
        prompt_embeds, pooled_prompt = model._encode_prompt(prompts)
        # Prepare control input
        control_input = model._prepare_control_input(person, m, cloth)

        # Initialize random image latents [B=1,16,h/8,w/8]
        B = 1
        latent_h = height // 8
        latent_w = width // 8
        latents = torch.randn((B, 16, latent_h, latent_w), device=device, dtype=torch.float16)

        scheduler = model.scheduler
        scheduler.set_timesteps(num_inference_steps)
        for t in scheduler.timesteps:
            # Duplicate for classifier-free guidance: [2,16,...]
            latent_model_input = torch.cat([latents, latents], dim=0)
            if isinstance(t, torch.Tensor):
                timestep = t.expand(latent_model_input.shape[0])
            else:
                timestep = torch.tensor([t] * latent_model_input.shape[0], device=device)
            # embeddings
            encoder_states = torch.cat([prompt_embeds, prompt_embeds], dim=0)
            pooled_states = torch.cat([pooled_prompt, pooled_prompt], dim=0)
            # For unconditional branch, zero control input
            zero_control = torch.zeros_like(control_input)
            control_inputs = torch.cat([zero_control, control_input], dim=0)

            # ControlNet + Transformer
            control_block = model.controlnet(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=encoder_states,
                pooled_projections=pooled_states,
                controlnet_cond=control_inputs,
                conditioning_scale=1.0,
                return_dict=False
            )[0]
            noise_pred = model.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=encoder_states,
                pooled_projections=pooled_states,
                block_controlnet_hidden_states=control_block,
                return_dict=False
            )[0]  # [2,16,h/8,w/8]
            noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
            guided_noise = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            latents = scheduler.step(guided_noise, t, latents).prev_sample

        # Decode via VAE only image latents
        final_img_latents = latents  # [1,16,h/8,w/8]
        final_img_latents = (final_img_latents / model.vae.config.scaling_factor) + model.vae.config.shift_factor
        final_img_latents = final_img_latents.to(dtype=model.vae.dtype)
        with torch.no_grad():
            decoded = model.vae.decode(final_img_latents, return_dict=False)[0]  # [1,3,H,W]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()  # [1,H,W,3]
        img_arr = (decoded[0] * 255).astype("uint8")
        return Image.fromarray(img_arr)
