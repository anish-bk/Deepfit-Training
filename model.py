# # model.py

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from utils import modify_unet_channels, modify_controlnet_channels  # remove freeze_non_trainable_components

# class DeepFit(nn.Module):
#     """
#     DeepFit model: loads VAE, text encoders, SD3 Transformer & ControlNet.
#     Modifies Transformer and ControlNet channels to desired sizes, copies original weights and zero-inits new channels.
#     Freezes most parts; only specific submodules of Transformer and ControlNet remain trainable.
#     """
#     def __init__(self,
#                  device: str = "cuda",
#                  debug: bool = False,
#                  transformer_in_channels: int = 20,
#                  transformer_out_channels: int = 20,
#                  controlnet_in_latent_channels: int = 20,
#                  controlnet_cond_channels: int = 33):
#         """
#         Args:
#             device: "cuda" or "cpu"
#             debug: whether to print debug info
#             transformer_in_channels: e.g. 20
#             transformer_out_channels: e.g. 20
#             controlnet_in_latent_channels: should equal transformer_out_channels
#             controlnet_cond_channels: e.g. 33 (16 person + 1 mask + 16 clothing)
#         """
#         super().__init__()
#         self.device = device
#         self.debug = debug

#         # 1. Load VAE
#         if debug:
#             print("[DEBUG] Loading VAE...")
#         self.vae = AutoencoderKL.from_pretrained(
#             "stabilityai/stable-diffusion-3-medium-diffusers",
#             subfolder="vae",
#             torch_dtype=torch.float16
#         ).to(device)
#         if debug:
#             print("[DEBUG] VAE loaded on", next(self.vae.parameters()).device)

#         # 2. Load CLIP Text Encoder 1 & tokenizer
#         if debug:
#             print("[DEBUG] Loading CLIP Text Encoder 1...")
#         self.tokenizer1 = CLIPTokenizer.from_pretrained(
#             "stabilityai/stable-diffusion-3-medium-diffusers",
#             subfolder="tokenizer"
#         )
#         self.text_encoder1 = CLIPTextModelWithProjection.from_pretrained(
#             "stabilityai/stable-diffusion-3-medium-diffusers",
#             subfolder="text_encoder",
#             torch_dtype=torch.float16
#         ).to(device)
#         if debug:
#             print("[DEBUG] CLIP Text Encoder 1 loaded on", next(self.text_encoder1.parameters()).device)

#         # 3. Load CLIP Text Encoder 2 & tokenizer
#         if debug:
#             print("[DEBUG] Loading CLIP Text Encoder 2...")
#         self.tokenizer2 = CLIPTokenizer.from_pretrained(
#             "stabilityai/stable-diffusion-3-medium-diffusers",
#             subfolder="tokenizer_2"
#         )
#         self.text_encoder2 = CLIPTextModelWithProjection.from_pretrained(
#             "stabilityai/stable-diffusion-3-medium-diffusers",
#             subfolder="text_encoder_2",
#             torch_dtype=torch.float16
#         ).to(device)
#         if debug:
#             print("[DEBUG] CLIP Text Encoder 2 loaded on", next(self.text_encoder2.parameters()).device)

#         # 4. Load T5 Encoder & tokenizer
#         if debug:
#             print("[DEBUG] Loading T5 Text Encoder...")
#         self.tokenizer3 = T5TokenizerFast.from_pretrained(
#             "stabilityai/stable-diffusion-3-medium-diffusers",
#             subfolder="tokenizer_3"
#         )
#         self.text_encoder3 = T5EncoderModel.from_pretrained(
#             "stabilityai/stable-diffusion-3-medium-diffusers",
#             subfolder="text_encoder_3",
#             torch_dtype=torch.float16
#         ).to(device)
#         if debug:
#             print("[DEBUG] T5 Text Encoder loaded on", next(self.text_encoder3.parameters()).device)

#         # 5. Load Transformer
#         if debug:
#             print("[DEBUG] Loading Transformer...")
#         self.original_transformer = SD3Transformer2DModel.from_pretrained(
#             "stabilityai/stable-diffusion-3-medium-diffusers",
#             subfolder="transformer",
#             torch_dtype=torch.float16
#         ).to(device)
#         if debug:
#             print("[DEBUG] Original Transformer loaded on", next(self.original_transformer.parameters()).device)

#         # Modify Transformer channels
#         if debug:
#             print(f"[DEBUG] Modifying Transformer: in_channels={transformer_in_channels}, out_channels={transformer_out_channels}...")
#         self.transformer = modify_transformer_channels(
#             transformer=self.original_transformer,
#             new_in_channels=transformer_in_channels,
#             new_out_channels=transformer_out_channels,
#             device=device
#         )
#         if debug:
#             print("[DEBUG] Transformer modified.")

#         # 6. Load ControlNet
#         if debug:
#             print("[DEBUG] Loading ControlNet...")
#         self.controlnet_orig = SD3ControlNetModel.from_pretrained(
#             "alimama-creative/SD3-Controlnet-Inpainting",
#             use_safetensors=True,
#             extra_conditioning_channels=1,
#             torch_dtype=torch.float16,
#             ignore_mismatched_sizes=True,
#             low_cpu_mem_usage=False
#         ).to(device)
#         if debug:
#             print("[DEBUG] ControlNet loaded on", next(self.controlnet_orig.parameters()).device)

#         # Modify ControlNet channels
#         if debug:
#             print(f"[DEBUG] Modifying ControlNet: latent_channels={controlnet_in_latent_channels}, cond_channels={controlnet_cond_channels}...")
#         self.controlnet = modify_controlnet_channels(
#             controlnet=self.controlnet_orig,
#             in_channels_latent=controlnet_in_latent_channels,
#             new_in_channels_cond=controlnet_cond_channels,
#             device=device
#         )
#         if debug:
#             print("[DEBUG] ControlNet modified.")

#         # 7. Freeze most parameters, only keep the specified submodules trainable
#         if debug:
#             print("[DEBUG] Freezing parameters except specified submodules...")

#         # First, freeze all parameters globally
#         for name, param in self.named_parameters():
#             param.requires_grad = False

#         # Helper: enable requires_grad for parameters whose name matches certain patterns.
#         # Transformer:
#         for name, param in self.transformer.named_parameters():
#             # 1) Transformer PatchEmbed layer: pos_embed.proj
#             if "pos_embed.proj" in name:
#                 param.requires_grad = True
#                 if debug:
#                     print(f"[DEBUG] Unfreeze transformer PatchEmbed param: {name}")
#                 continue

#             # 2) Attention weights in transformer blocks
#             if ".transformer_blocks." in name and ".attn." in name:
#                 param.requires_grad = True
#                 if debug:
#                     print(f"[DEBUG] Unfreeze transformer attention param: {name}")
#                 continue

#             # 3) Adaptive LayerNorm modulation weights in transformer blocks
#             #    norm1.linear and norm1_context.linear
#             if ".transformer_blocks." in name and (".norm1.linear" in name or ".norm1_context.linear" in name):
#                 param.requires_grad = True
#                 if debug:
#                     print(f"[DEBUG] Unfreeze transformer AdaLayerNorm param: {name}")
#                 continue

#             # 4) Final adaptive norm in transformer, if present (e.g., norm_out.linear)
#             if "norm_out.linear" in name:
#                 param.requires_grad = True
#                 if debug:
#                     print(f"[DEBUG] Unfreeze transformer final norm param: {name}")
#                 continue

#         # ControlNet:
#         for name, param in self.controlnet.named_parameters():
#             # 1) ControlNet PatchEmbed layers: often named pos_embed.proj or pos_embed_input.proj
#             if ("pos_embed.proj" in name) or ("pos_embed_input.proj" in name):
#                 param.requires_grad = True
#                 if debug:
#                     print(f"[DEBUG] Unfreeze controlnet PatchEmbed param: {name}")
#                 continue

#             # 2) Attention weights in controlnet transformer blocks
#             if ".transformer_blocks." in name and ".attn." in name:
#                 param.requires_grad = True
#                 if debug:
#                     print(f"[DEBUG] Unfreeze controlnet attention param: {name}")
#                 continue

#             # If you also want adaptive LayerNorm weights in ControlNet trainable,
#             # you could uncomment the following:
#             # if ".transformer_blocks." in name and (".norm1.linear" in name or ".norm1_context.linear" in name):
#             #     param.requires_grad = True
#             #     if debug:
#             #         print(f"[DEBUG] Unfreeze controlnet AdaLayerNorm param: {name}")
#             #     continue

#         if debug:
#             # Report total trainable parameters
#             trainable = [(n, p.shape) for n,p in self.named_parameters() if p.requires_grad]
#             print(f"[DEBUG] Total trainable parameters after freezing: {len(trainable)}")
#             for n,sh in trainable:
#                 print(f"  {n}: {sh}")

#         # 8. Scheduler for inference
#         if debug:
#             print("[DEBUG] Loading Scheduler...")
#         self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
#             "stabilityai/stable-diffusion-3-medium-diffusers",
#             subfolder="scheduler"
#         )
#         if debug:
#             print("[DEBUG] Scheduler loaded.")

#     def forward(self,
#                 noisy_latents: torch.Tensor,
#                 timesteps: torch.Tensor,
#                 control_input: torch.Tensor,
#                 prompt_embeds: torch.Tensor,
#                 pooled_prompt: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass: ControlNet then Transformer to predict noise (for training).
#         noisy_latents: [B, latent_channels, h/8, w/8], float16
#         timesteps: [B], float or int as required by scheduler/SD3
#         control_input: [B, cond_channels, h/8, w/8], float16
#         prompt_embeds: [B, seq_len, dim], float16
#         pooled_prompt: [B, 2048], float16
#         Returns model_pred [B, latent_channels, h/8, w/8], float16
#         """
#         control_block = self.controlnet(
#             hidden_states=noisy_latents,
#             timestep=timesteps,
#             encoder_hidden_states=prompt_embeds,
#             pooled_projections=pooled_prompt,
#             controlnet_cond=control_input,
#             conditioning_scale=1.0,
#             return_dict=False,
#         )[0]
#         model_pred = self.transformer(
#             hidden_states=noisy_latents,
#             timestep=timesteps,
#             encoder_hidden_states=prompt_embeds,
#             pooled_projections=pooled_prompt,
#             block_controlnet_hidden_states=control_block,
#             return_dict=False,
#         )[0]
#         return model_pred







class DeepFit(nn.Module):
    """
    DeepFit model: loads VAE, UNet & ControlNet for SD1.5 Inpainting.
    Assumes prompt embeddings are provided by the dataset, so no tokenizers or text encoders are needed here.
    """
    def __init__(
        self,
        device: str = "cuda",
        debug: bool = False,
        unet_in_channels: int = 13,  # SD1.5 inpainting: 4 latent + 4 masked latent + 1 mask + 4 clothing latent = 13
        unet_out_channels: int = 4,
        controlnet_in_channels: int = 13,
        controlnet_cond_channels: int = 9  # person latent (4) + mask (1) + clothing latent (4)
    ):
        super().__init__()
        self.device = device
        self.debug = debug

        # 1. Load VAE
        if debug:
            print("[DEBUG] Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            subfolder="vae",
            torch_dtype=torch.float16
        ).to(device)

        # 2. Load and modify UNet
        if debug:
            print("[DEBUG] Loading and modifying UNet...")
        orig_unet = UNet2DConditionModel.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            subfolder="unet",
            torch_dtype=torch.float16
        ).to(device)
        self.unet = modify_unet_channels(
            unet=orig_unet,
            new_in_channels=unet_in_channels,
            device=device
        )

        # 3. Load and modify ControlNet
        if debug:
            print("[DEBUG] Loading and modifying ControlNet...")
        orig_cn = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint",
            torch_dtype=torch.float16
        ).to(device)
        self.controlnet = modify_controlnet_channels(
            controlnet=orig_cn,
            new_in_channels=controlnet_cond_channels,
            device=device
        )

        # 4. Freeze most parameters, enable only selected submodules
        if debug:
            print("[DEBUG] Freezing parameters except selected submodules...")
        for _, p in self.named_parameters():
            p.requires_grad = False

        # Unfreeze UNet input conv and attention layers
        for n, p in self.unet.named_parameters():
            if "conv_in" in n or ".attentions." in n or ".attn" in n:
                p.requires_grad = True
                if debug:
                    print(f"[DEBUG] Unfrozen unet param: {n}")

        # Unfreeze ControlNet input conv and attention layers
        for n, p in self.controlnet.named_parameters():
            if "conv_in" in n or "controlnet_cond_embedding" in n or ".attentions." in n or ".attn" in n:
                p.requires_grad = True
                if debug:
                    print(f"[DEBUG] Unfrozen controlnet param: {n}")

        # 5. Scheduler for inference
        if debug:
            print("[DEBUG] Loading Scheduler...")
        self.scheduler = DDPMScheduler.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            subfolder="scheduler"
        )

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        control_input: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt: torch.Tensor = None  # Not used for SD1.5 but kept for compatibility
    ) -> torch.Tensor:
        """
        Forward pass: apply ControlNet then UNet to predict the added noise.
        Inputs:
          - noisy_latents: [B, latent_C, h/8, w/8]
          - timesteps:     [B]
          - control_input: [B, cond_C, h/8, w/8]
          - prompt_embeds: [B, seq_len, dim]
          - pooled_prompt: Not used for SD1.5
        Returns:
          - model_pred:   [B, latent_C, h/8, w/8]
        """
        # ControlNet forward
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=control_input,
            conditioning_scale=1.0,
            return_dict=False
        )

        # UNet forward with ControlNet residuals
        model_pred = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False
        )[0]

        return model_pred
