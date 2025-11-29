"""
This module implements the functions to convert Chroma LoRA weights from various formats
to a common intermediate format for processing.

Chroma uses the same LoRA formats as FLUX (ComfyUI/Kohya style), so we can
largely reuse the FLUX conversion logic.
"""

import logging
import os
import re

import torch
from safetensors.torch import load_file as load_safetensors

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_state_dict_in_safetensors(path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load a safetensors file."""
    return load_safetensors(path, device=device)


def handle_comfyui_chroma_lora(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert ComfyUI/Kohya Chroma LoRA format to an intermediate format.

    ComfyUI format uses keys like:
    - lora_unet_double_blocks_0_img_attn_proj.lora_down.weight
    - lora_unet_single_blocks_0_linear1.lora_down.weight

    This converts to diffusers-style keys:
    - transformer_blocks.0.attn.to_out.0.lora_A.weight
    - single_transformer_blocks.0.proj_mlp.lora_A.weight (for mlp part)
    - single_transformer_blocks.0.attn.to_q.lora_A.weight (for qkv part)

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        LoRA weights in ComfyUI format.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in diffusers format.
    """
    new_state_dict = {}

    # Check if this is ComfyUI format
    if not any(k.startswith("lora_unet_") for k in state_dict.keys()):
        return state_dict

    logger.debug("Converting ComfyUI Chroma LoRA format to diffusers format")

    # Mapping from ComfyUI layer names to diffusers layer names
    double_block_map = {
        "img_attn_proj": "attn.to_out.0",
        "img_attn_qkv": "attn.to_qkv",
        "img_mlp_0": "ff.net.0.proj",
        "img_mlp_2": "ff.net.2",
        "txt_attn_proj": "attn.to_add_out",
        "txt_attn_qkv": "attn.add_qkv_proj",
        "txt_mlp_0": "ff_context.net.0.proj",
        "txt_mlp_2": "ff_context.net.2",
    }

    for key, value in state_dict.items():
        # Skip alpha values (we apply them below)
        if ".alpha" in key:
            continue

        new_key = key

        # Handle double blocks
        match = re.match(r"lora_unet_double_blocks_(\d+)_([^.]+)\.(lora_(?:down|up))\.weight", key)
        if match:
            block_idx = match.group(1)
            layer_name = match.group(2)
            lora_type = match.group(3)  # lora_down or lora_up

            # Map to diffusers format
            if layer_name in double_block_map:
                diffusers_layer = double_block_map[layer_name]
                lora_suffix = "lora_A" if lora_type == "lora_down" else "lora_B"
                new_key = f"transformer_blocks.{block_idx}.{diffusers_layer}.{lora_suffix}.weight"

                # Get alpha for scaling
                alpha_key = key.replace(f".{lora_type}.weight", ".alpha")
                if alpha_key in state_dict:
                    alpha = state_dict[alpha_key].item()
                    rank = value.shape[0] if lora_type == "lora_down" else value.shape[1]
                    if lora_type == "lora_down":
                        value = value * (alpha / rank)

                new_state_dict[new_key] = value
                continue

        # Handle single blocks
        match = re.match(r"lora_unet_single_blocks_(\d+)_([^.]+)\.(lora_(?:down|up))\.weight", key)
        if match:
            block_idx = match.group(1)
            layer_name = match.group(2)
            lora_type = match.group(3)  # lora_down or lora_up
            lora_suffix = "lora_A" if lora_type == "lora_down" else "lora_B"

            # Get alpha for scaling
            alpha_key = key.replace(f".{lora_type}.weight", ".alpha")
            alpha = 1.0
            if alpha_key in state_dict:
                alpha = state_dict[alpha_key].item()
                rank = value.shape[0] if lora_type == "lora_down" else value.shape[1]
                if lora_type == "lora_down":
                    value = value * (alpha / rank)

            if layer_name == "linear1":
                # linear1 is fused (mlp + qkv): first 12288 dims are mlp, next 9216 are qkv
                # Split lora_B (up projection) which has the output dimension
                if lora_type == "lora_up":
                    # lora_up has shape [out_features, rank]
                    # Split into mlp (12288) and qkv (9216)
                    if value.shape[0] == 21504:  # 12288 + 9216
                        mlp_value = value[:12288, :].clone()
                        qkv_value = value[12288:, :].clone()

                        new_state_dict[f"single_transformer_blocks.{block_idx}.proj_mlp.{lora_suffix}.weight"] = (
                            mlp_value
                        )
                        # Split QKV into Q, K, V (each 3072)
                        new_state_dict[f"single_transformer_blocks.{block_idx}.attn.to_q.{lora_suffix}.weight"] = (
                            qkv_value[:3072, :]
                        )
                        new_state_dict[f"single_transformer_blocks.{block_idx}.attn.to_k.{lora_suffix}.weight"] = (
                            qkv_value[3072:6144, :]
                        )
                        new_state_dict[f"single_transformer_blocks.{block_idx}.attn.to_v.{lora_suffix}.weight"] = (
                            qkv_value[6144:, :]
                        )
                    else:
                        # Just use as mlp if dimensions don't match expected fused format
                        new_state_dict[f"single_transformer_blocks.{block_idx}.proj_mlp.{lora_suffix}.weight"] = value
                else:
                    # lora_down is shared across the split outputs
                    new_state_dict[f"single_transformer_blocks.{block_idx}.proj_mlp.{lora_suffix}.weight"] = value
                    new_state_dict[f"single_transformer_blocks.{block_idx}.attn.to_q.{lora_suffix}.weight"] = value
                    new_state_dict[f"single_transformer_blocks.{block_idx}.attn.to_k.{lora_suffix}.weight"] = value
                    new_state_dict[f"single_transformer_blocks.{block_idx}.attn.to_v.{lora_suffix}.weight"] = value
                continue

            elif layer_name == "linear2":
                # linear2 is also fused: first part is attn_out, second is mlp_fc2
                if lora_type == "lora_down":
                    # lora_down has shape [rank, in_features]
                    # Split based on input dimension (3072 for attn_out + 12288 for mlp_fc2 = 15360)
                    if value.shape[1] == 15360:
                        attn_value = value[:, :3072].clone()
                        mlp_value = value[:, 3072:].clone()

                        new_state_dict[
                            f"single_transformer_blocks.{block_idx}.proj_out.linears.0.{lora_suffix}.weight"
                        ] = attn_value
                        new_state_dict[
                            f"single_transformer_blocks.{block_idx}.proj_out.linears.1.{lora_suffix}.weight"
                        ] = mlp_value
                    else:
                        # Dimensions don't match expected fused format
                        new_state_dict[
                            f"single_transformer_blocks.{block_idx}.proj_out.linears.0.{lora_suffix}.weight"
                        ] = value
                        new_state_dict[
                            f"single_transformer_blocks.{block_idx}.proj_out.linears.1.{lora_suffix}.weight"
                        ] = value
                else:
                    # lora_up is shared
                    new_state_dict[
                        f"single_transformer_blocks.{block_idx}.proj_out.linears.0.{lora_suffix}.weight"
                    ] = value
                    new_state_dict[
                        f"single_transformer_blocks.{block_idx}.proj_out.linears.1.{lora_suffix}.weight"
                    ] = value
                continue

        # If no pattern matched, keep original key
        if key not in new_state_dict:
            new_state_dict[new_key] = value

    return new_state_dict


def handle_bfl_chroma_lora(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert BFL/ComfyUI native Chroma LoRA format to diffusers format.

    This format uses keys like:
    - diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight
    - diffusion_model.single_blocks.0.linear1.lora_A.weight

    This converts to diffusers-style keys:
    - transformer_blocks.0.attn.to_out.0.lora_A.weight
    - single_transformer_blocks.0.proj_mlp.lora_A.weight

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        LoRA weights in BFL/native format.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in diffusers format.
    """
    new_state_dict = {}

    # Check if this is BFL format
    if not any(k.startswith("diffusion_model.") for k in state_dict.keys()):
        return state_dict

    logger.debug("Converting BFL/native Chroma LoRA format to diffusers format")

    # Mapping from BFL layer names to diffusers layer names
    double_block_map = {
        "img_attn.proj": "attn.to_out.0",
        "img_attn.qkv": "attn.to_qkv",
        "img_mlp.0": "ff.net.0.proj",
        "img_mlp.2": "ff.net.2",
        "txt_attn.proj": "attn.to_add_out",
        "txt_attn.qkv": "attn.add_qkv_proj",
        "txt_mlp.0": "ff_context.net.0.proj",
        "txt_mlp.2": "ff_context.net.2",
    }

    for key, value in state_dict.items():
        # Skip non-LoRA keys
        if ".lora_A." not in key and ".lora_B." not in key:
            continue

        new_key = key

        # Handle double blocks: diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight
        match = re.match(r"diffusion_model\.double_blocks\.(\d+)\.([^.]+\.[^.]+)\.(lora_[AB])\.weight", key)
        if match:
            block_idx = match.group(1)
            layer_name = match.group(2)  # e.g., "img_attn.proj"
            lora_type = match.group(3)  # lora_A or lora_B

            if layer_name in double_block_map:
                diffusers_layer = double_block_map[layer_name]
                new_key = f"transformer_blocks.{block_idx}.{diffusers_layer}.{lora_type}.weight"
                new_state_dict[new_key] = value
                continue

        # Handle single blocks: diffusion_model.single_blocks.0.linear1.lora_A.weight
        match = re.match(r"diffusion_model\.single_blocks\.(\d+)\.([^.]+)\.(lora_[AB])\.weight", key)
        if match:
            block_idx = match.group(1)
            layer_name = match.group(2)  # e.g., "linear1" or "linear2"
            lora_type = match.group(3)  # lora_A or lora_B

            if layer_name == "linear1":
                # linear1 is fused: first part is QKV (split to Q, K, V), second part is mlp_fc1
                if lora_type == "lora_A":
                    # lora_A has shape [rank, in_features]
                    # For Chroma: QKV part is 3072*3=9216, MLP part is 12288, total 21504
                    # But we need to check the actual shape
                    if value.shape[1] == 21504:  # Full fused format
                        qkv_value = value[:, :9216].clone()
                        mlp_value = value[:, 9216:].clone()

                        # Split QKV into Q, K, V (each 3072)
                        q_value = qkv_value[:, :3072].clone()
                        k_value = qkv_value[:, 3072:6144].clone()
                        v_value = qkv_value[:, 6144:].clone()

                        new_state_dict[f"single_transformer_blocks.{block_idx}.attn.to_q.{lora_type}.weight"] = q_value
                        new_state_dict[f"single_transformer_blocks.{block_idx}.attn.to_k.{lora_type}.weight"] = k_value
                        new_state_dict[f"single_transformer_blocks.{block_idx}.attn.to_v.{lora_type}.weight"] = v_value
                        new_state_dict[f"single_transformer_blocks.{block_idx}.proj_mlp.{lora_type}.weight"] = mlp_value
                    else:
                        # Just store as proj_mlp if dimensions don't match expected fused format
                        new_state_dict[f"single_transformer_blocks.{block_idx}.proj_mlp.{lora_type}.weight"] = value
                else:
                    # lora_B has shape [out_features, rank]
                    # For fused, out is 21504 (9216 QKV + 12288 MLP)
                    if value.shape[0] == 21504:
                        qkv_value = value[:9216, :].clone()
                        mlp_value = value[9216:, :].clone()

                        # Split QKV into Q, K, V
                        q_value = qkv_value[:3072, :].clone()
                        k_value = qkv_value[3072:6144, :].clone()
                        v_value = qkv_value[6144:, :].clone()

                        new_state_dict[f"single_transformer_blocks.{block_idx}.attn.to_q.{lora_type}.weight"] = q_value
                        new_state_dict[f"single_transformer_blocks.{block_idx}.attn.to_k.{lora_type}.weight"] = k_value
                        new_state_dict[f"single_transformer_blocks.{block_idx}.attn.to_v.{lora_type}.weight"] = v_value
                        new_state_dict[f"single_transformer_blocks.{block_idx}.proj_mlp.{lora_type}.weight"] = mlp_value
                    else:
                        new_state_dict[f"single_transformer_blocks.{block_idx}.proj_mlp.{lora_type}.weight"] = value
                continue

            elif layer_name == "linear2":
                # linear2 is fused: first part is attn_out (3072), second is mlp_fc2 (12288)
                if lora_type == "lora_A":
                    # lora_A: [rank, in_features] where in_features = 3072 + 12288 = 15360
                    if value.shape[1] == 15360:
                        attn_value = value[:, :3072].clone()
                        mlp_value = value[:, 3072:].clone()

                        new_state_dict[f"single_transformer_blocks.{block_idx}.proj_out.linears.0.{lora_type}.weight"] = attn_value
                        new_state_dict[f"single_transformer_blocks.{block_idx}.proj_out.linears.1.{lora_type}.weight"] = mlp_value
                    else:
                        new_state_dict[f"single_transformer_blocks.{block_idx}.proj_out.linears.0.{lora_type}.weight"] = value
                        new_state_dict[f"single_transformer_blocks.{block_idx}.proj_out.linears.1.{lora_type}.weight"] = value
                else:
                    # lora_B: [out_features, rank] - shared for both outputs
                    new_state_dict[f"single_transformer_blocks.{block_idx}.proj_out.linears.0.{lora_type}.weight"] = value
                    new_state_dict[f"single_transformer_blocks.{block_idx}.proj_out.linears.1.{lora_type}.weight"] = value
                continue

        # If no pattern matched, keep original key
        if new_key not in new_state_dict:
            new_state_dict[new_key] = value

    return new_state_dict


def to_diffusers(input_lora: str | dict[str, torch.Tensor], output_path: str | None = None) -> dict[str, torch.Tensor]:
    """
    Convert Chroma LoRA weights to an intermediate diffusers-like format.

    Parameters
    ----------
    input_lora : str or dict[str, torch.Tensor]
        Path to a safetensors file or a LoRA weight dictionary.
    output_path : str, optional
        If given, save the converted weights to this path.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in intermediate format with keys like:
        - transformer_blocks.{idx}.{layer}.lora_A.weight
        - single_transformer_blocks.{idx}.{layer}.lora_A.weight
    """
    if isinstance(input_lora, str):
        tensors = load_state_dict_in_safetensors(input_lora, device="cpu")
    else:
        tensors = {k: v for k, v in input_lora.items()}

    # Convert FP8 tensors to BF16
    for k, v in tensors.items():
        if v.dtype not in [torch.float64, torch.float32, torch.bfloat16, torch.float16]:
            tensors[k] = v.to(torch.bfloat16)

    # Check if already in diffusers format - return as-is
    if any(k.startswith("transformer_blocks.") or k.startswith("single_transformer_blocks.") for k in tensors.keys()):
        new_tensors = tensors
    # Handle ComfyUI/Kohya format (lora_unet_*)
    elif any(k.startswith("lora_unet_") for k in tensors.keys()):
        new_tensors = handle_comfyui_chroma_lora(tensors)
    # Handle BFL/native format (diffusion_model.*)
    elif any(k.startswith("diffusion_model.") for k in tensors.keys()):
        new_tensors = handle_bfl_chroma_lora(tensors)
    else:
        # Try using FLUX's conversion logic as fallback
        try:
            from diffusers.loaders import FluxLoraLoaderMixin
            from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft

            new_tensors, alphas = FluxLoraLoaderMixin.lora_state_dict(tensors, return_alphas=True)
            new_tensors = convert_unet_state_dict_to_peft(new_tensors)

            if alphas is not None and len(alphas) > 0:
                for k, v in alphas.items():
                    key_A = k.replace(".alpha", ".lora_A.weight")
                    key_B = k.replace(".alpha", ".lora_B.weight")
                    if key_A in new_tensors and key_B in new_tensors:
                        rank = new_tensors[key_A].shape[0]
                        new_tensors[key_A] = new_tensors[key_A] * v / rank
        except Exception:
            # If diffusers conversion fails, just return as-is
            new_tensors = tensors

    if output_path is not None:
        from safetensors.torch import save_file

        import os

        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        save_file(new_tensors, output_path)

    return new_tensors
