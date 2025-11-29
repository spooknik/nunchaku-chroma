"""
This module provides a wrapper for the NunchakuChromaTransformer2DModel,
enabling integration with ComfyUI forward.
"""

import torch
from einops import rearrange, repeat
from torch import nn

# Import Chroma LoRA converter and compose function - try multiple import paths for flexibility
try:
    # When installed as part of nunchaku-chroma package
    from ..lora.diffusers_converter import to_diffusers as chroma_to_diffusers
    from ..lora.compose import compose_lora
except ImportError:
    try:
        # When installed to ComfyUI-nunchaku wrappers directory
        from .lora.diffusers_converter import to_diffusers as chroma_to_diffusers
        from .lora.compose import compose_lora
    except ImportError:
        # Fallback to nunchaku's chroma LoRA converter
        from nunchaku.lora.chroma.diffusers_converter import to_diffusers as chroma_to_diffusers
        from nunchaku.lora.chroma.compose import compose_lora


class ComfyChromaWrapper(nn.Module):
    """
    Wrapper for NunchakuChromaTransformer2DModel to support ComfyUI workflows.

    Parameters
    ----------
    model : NunchakuChromaTransformer2DModel
        The underlying Nunchaku model to wrap.
    config : dict
        Model configuration dictionary.

    Attributes
    ----------
    model : NunchakuChromaTransformer2DModel
        The wrapped model.
    dtype : torch.dtype
        Data type of the model parameters.
    config : dict
        Model configuration.
    loras : list
        List of (lora_path, strength) tuples for tracking loaded LoRAs.
    """

    def __init__(self, model, config: dict):
        super(ComfyChromaWrapper, self).__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config
        self.loras = []  # List of (lora_path, strength) tuples

        # Initialize LoRA tracking attributes on the model if they don't exist
        if not hasattr(model, "comfy_lora_meta_list"):
            model.comfy_lora_meta_list = []
        if not hasattr(model, "comfy_lora_sd_list"):
            model.comfy_lora_sd_list = []

    def pack_latents(self, x):
        """
        Pack latents from (B, 16, H, W) to (B, 64, H/2, W/2) format.

        Chroma expects packed latents where 2x2 spatial patches are packed into channels.

        Parameters
        ----------
        x : torch.Tensor
            Input latent tensor of shape (batch, 16, height, width).

        Returns
        -------
        torch.Tensor
            Packed latent tensor of shape (batch, 64, height/2, width/2).
        """
        # Pack 2x2 spatial patches into channels: (B, 16, H, W) -> (B, 64, H/2, W/2)
        x = rearrange(x, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=2, pw=2)
        return x

    def unpack_latents(self, x, h, w):
        """
        Unpack latents from (B, 64, H/2, W/2) back to (B, 16, H, W) format.

        Parameters
        ----------
        x : torch.Tensor
            Packed latent tensor of shape (batch, 64, height/2, width/2).
        h : int
            Target height (original unpacked height).
        w : int
            Target width (original unpacked width).

        Returns
        -------
        torch.Tensor
            Unpacked latent tensor of shape (batch, 16, height, width).
        """
        # Unpack channels back to 2x2 spatial patches: (B, 64, H/2, W/2) -> (B, 16, H, W)
        x = rearrange(x, "b (c ph pw) h w -> b c (h ph) (w pw)", ph=2, pw=2, h=h // 2, w=w // 2)
        return x

    def process_img(self, x, pad_to_patch_size_fn, index=0, h_offset=0, w_offset=0):
        """
        Preprocess an input image tensor for the model.

        Pads and rearranges the image into patches and generates corresponding image IDs.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, channels, height, width).
            For Chroma, this should be packed latents with 64 channels.
        pad_to_patch_size_fn : callable
            Function to pad tensor to patch size.
        index : int, optional
            Index for image ID encoding.
        h_offset : int, optional
            Height offset for patch IDs.
        w_offset : int, optional
            Width offset for patch IDs.

        Returns
        -------
        img : torch.Tensor
            Rearranged image tensor of shape (batch, num_patches, patch_dim).
        img_ids : torch.Tensor
            Image ID tensor of shape (batch, num_patches, 3).
        """
        bs, c, h, w = x.shape
        patch_size = self.config.get("patch_size", 1)  # Chroma uses patch_size=1
        x = pad_to_patch_size_fn(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size

        h_offset = (h_offset + (patch_size // 2)) // patch_size
        w_offset = (w_offset + (patch_size // 2)) // patch_size

        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(
            h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype
        ).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(
            w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)

    def forward(
        self,
        x,
        timestep,
        context,
        y=None,
        guidance=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        """
        Forward pass for the wrapped model.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor.
        timestep : float or torch.Tensor
            Diffusion timestep.
        context : torch.Tensor
            Context tensor (e.g., text embeddings).
        y : torch.Tensor, optional
            Pooled projections or additional conditioning.
        guidance : torch.Tensor, optional
            Guidance embedding or value.
        control : dict, optional
            ControlNet input and output samples.
        transformer_options : dict, optional
            Additional transformer options.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        out : torch.Tensor
            Output tensor of the same spatial size as the input.
        """
        # Import here to avoid issues when not running in ComfyUI
        from comfy.ldm.common_dit import pad_to_patch_size

        model = self.model

        bs, c, h_orig, w_orig = x.shape

        # ComfyUI provides unpacked latents (B, 16, H, W)
        # Chroma expects packed latents (B, 64, H/2, W/2)
        x_packed = self.pack_latents(x)
        _, c_packed, h_packed, w_packed = x_packed.shape

        patch_size = self.config.get("patch_size", 1)  # Chroma uses patch_size=1
        h_len = (h_packed + (patch_size // 2)) // patch_size
        w_len = (w_packed + (patch_size // 2)) // patch_size

        img, img_ids = self.process_img(x_packed, pad_to_patch_size)
        img_tokens = img.shape[1]

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)

        # Load and compose LoRA if needed
        if self.loras != model.comfy_lora_meta_list:
            lora_to_be_composed = []

            # Remove extra LoRAs from tracking lists
            for _ in range(max(0, len(model.comfy_lora_meta_list) - len(self.loras))):
                model.comfy_lora_meta_list.pop()
                model.comfy_lora_sd_list.pop()

            # Process each LoRA
            for i in range(len(self.loras)):
                meta = self.loras[i]
                if i >= len(model.comfy_lora_meta_list):
                    # New LoRA - load and convert to diffusers format
                    sd = chroma_to_diffusers(meta[0])
                    model.comfy_lora_meta_list.append(meta)
                    model.comfy_lora_sd_list.append(sd)
                elif model.comfy_lora_meta_list[i] != meta:
                    # LoRA changed
                    if meta[0] != model.comfy_lora_meta_list[i][0]:
                        sd = chroma_to_diffusers(meta[0])
                        model.comfy_lora_sd_list[i] = sd
                    model.comfy_lora_meta_list[i] = meta
                lora_to_be_composed.append(({k: v for k, v in model.comfy_lora_sd_list[i].items()}, meta[1]))

            # Compose all LoRAs into one
            composed_lora = compose_lora(lora_to_be_composed)

            if len(composed_lora) == 0:
                model.reset_lora()
            else:
                # Check if x_embedder needs to be reset
                if "x_embedder.lora_A.weight" in composed_lora:
                    new_in_channels = composed_lora["x_embedder.lora_A.weight"].shape[1]
                    current_in_channels = model.x_embedder.in_features
                    if new_in_channels < current_in_channels:
                        model.reset_x_embedder()
                model.update_lora_params(composed_lora)

        controlnet_block_samples = None if control is None else [sample.to(x.dtype) for sample in control["input"]]
        controlnet_single_block_samples = None if control is None else [sample.to(x.dtype) for sample in control["output"]]

        # Chroma uses timestep / 1000 (same as FLUX schnell)
        out = model(
            hidden_states=img,
            encoder_hidden_states=context,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        ).sample

        out = out[:, :img_tokens]
        out = rearrange(
            out,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h_len,
            w=w_len,
            ph=patch_size,
            pw=patch_size,
        )
        out = out[:, :, :h_packed, :w_packed]

        # Unpack output back to (B, 16, H, W) format for ComfyUI
        out = self.unpack_latents(out, h_orig, w_orig)

        return out
