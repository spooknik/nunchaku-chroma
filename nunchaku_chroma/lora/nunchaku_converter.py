"""
Nunchaku LoRA format converter for Chroma models.

This module provides utilities to convert LoRA weights from Diffusers format
to Nunchaku format for efficient quantized inference in Chroma models.
"""

import logging
import os

import torch

from nunchaku.lora.flux.nunchaku_converter import pack_lowrank_weight, unpack_lowrank_weight
from nunchaku.lora.flux.utils import pad
from .diffusers_converter import to_diffusers

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Mapping from diffusers LoRA keys to Nunchaku model keys
TRANSFORMER_BLOCK_MAP = {
    "attn.to_qkv": "attn.to_qkv",
    "attn.to_out.0": "attn.to_out.0",
    "attn.add_qkv_proj": "attn.add_qkv_proj",
    "attn.to_add_out": "attn.to_add_out",
    "ff.net.0.proj": "ff.net.0.proj",
    "ff.net.2": "ff.net.2",
    "ff_context.net.0.proj": "ff_context.net.0.proj",
    "ff_context.net.2": "ff_context.net.2",
}

SINGLE_BLOCK_MAP = {
    "proj_mlp": "mlp_fc1",
    "attn.to_q": "attn.to_qkv",
    "attn.to_k": "attn.to_qkv",
    "attn.to_v": "attn.to_qkv",
    "proj_out.linears.0": "attn.to_out",
    "proj_out.linears.1": "mlp_fc2",
    # Alternative names that might appear
    "mlp_fc1": "mlp_fc1",
    "mlp_fc2": "mlp_fc2",
    "attn.to_qkv": "attn.to_qkv",
    "attn.to_out": "attn.to_out",
}


def merge_lora_with_svd_branch(
    orig_down: torch.Tensor,
    orig_up: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    strength: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Merge an external LoRA with the existing SVD low-rank branch.

    The SVD branch computes: output += proj_up @ proj_down @ input
    The LoRA branch computes: output += strength * lora_B @ lora_A @ input

    Combined: output += [proj_up | strength*lora_B] @ [proj_down; lora_A] @ input

    This concatenates the LoRA weights with the existing SVD weights.

    Parameters
    ----------
    orig_down : torch.Tensor
        Original proj_down (unpacked), shape [rank, in_features]
    orig_up : torch.Tensor
        Original proj_up (unpacked), shape [out_features, rank]
    lora_A : torch.Tensor
        LoRA down projection, shape [lora_rank, in_features]
    lora_B : torch.Tensor
        LoRA up projection, shape [out_features, lora_rank]
    strength : float
        LoRA strength multiplier

    Returns
    -------
    new_down : torch.Tensor
        Merged proj_down, shape [rank + lora_rank, in_features]
    new_up : torch.Tensor
        Merged proj_up, shape [out_features, rank + lora_rank]
    """
    # Move LoRA tensors to the same device and dtype as original
    device = orig_down.device
    dtype = orig_down.dtype

    # LoRA A: [lora_rank, in_features] - same convention as orig_down
    # LoRA B: [out_features, lora_rank] - same convention as orig_up
    lora_A = lora_A.to(device=device, dtype=dtype)
    lora_B = (lora_B * strength).to(device=device, dtype=orig_up.dtype)

    # Concatenate along the rank dimension
    # orig_down is [rank, in_features], lora_A is [lora_rank, in_features]
    # Result: [rank + lora_rank, in_features]
    new_down = torch.cat([orig_down, lora_A], dim=0)

    # orig_up is [out_features, rank], lora_B is [out_features, lora_rank]
    # Result: [out_features, rank + lora_rank]
    new_up = torch.cat([orig_up, lora_B], dim=1)

    return new_down, new_up


def convert_chroma_lora_to_nunchaku(
    lora_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
    strength: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Convert LoRA weights to Nunchaku format by merging with existing SVD branch.

    Parameters
    ----------
    lora_dict : dict[str, torch.Tensor]
        LoRA weights in diffusers format (from to_diffusers)
    model_state_dict : dict[str, torch.Tensor]
        Current model state dict with SVD weights
    strength : float
        LoRA strength multiplier

    Returns
    -------
    dict[str, torch.Tensor]
        Updated state dict with merged LoRA weights
    """
    result = {}

    # Group LoRA weights by layer
    lora_layers: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in lora_dict.items():
        # Parse key like "transformer_blocks.0.attn.to_qkv.lora_A.weight"
        parts = key.rsplit(".", 2)
        if len(parts) < 3:
            continue

        layer_name = parts[0]  # e.g., "transformer_blocks.0.attn.to_qkv"
        lora_type = parts[1]  # "lora_A" or "lora_B"

        if layer_name not in lora_layers:
            lora_layers[layer_name] = {}

        lora_layers[layer_name][lora_type] = value

    # Handle QKV fusion for single blocks
    fused_qkv: dict[str, dict[str, list[torch.Tensor]]] = {}
    for layer_name, lora_weights in list(lora_layers.items()):
        if "single_transformer_blocks" in layer_name:
            # Check if this is a split QKV - use endswith to avoid substring issues
            for qkv_type in ["attn.to_q", "attn.to_k", "attn.to_v"]:
                if layer_name.endswith(qkv_type):
                    # Get base layer name (block.attn.to_qkv)
                    base_name = layer_name[: -len(qkv_type)] + "attn.to_qkv"
                    if base_name not in fused_qkv:
                        fused_qkv[base_name] = {"lora_A": [], "lora_B": []}

                    if "lora_A" in lora_weights:
                        fused_qkv[base_name]["lora_A"].append(lora_weights["lora_A"])
                    if "lora_B" in lora_weights:
                        fused_qkv[base_name]["lora_B"].append(lora_weights["lora_B"])

                    # Remove from lora_layers
                    del lora_layers[layer_name]
                    break

    # Fuse the split QKV back together
    for base_name, parts_dict in fused_qkv.items():
        if len(parts_dict["lora_A"]) > 0 and len(parts_dict["lora_B"]) > 0:
            # All Q,K,V share the same lora_A
            lora_A = parts_dict["lora_A"][0]
            # Concatenate lora_B parts
            lora_B = torch.cat(parts_dict["lora_B"], dim=0)
            lora_layers[base_name] = {"lora_A": lora_A, "lora_B": lora_B}

    # Process each layer
    for lora_layer_name, lora_weights in lora_layers.items():
        if "lora_A" not in lora_weights or "lora_B" not in lora_weights:
            continue

        lora_A = lora_weights["lora_A"]
        lora_B = lora_weights["lora_B"]

        # Map to model layer name
        # Check single_transformer_blocks FIRST (it contains "transformer_blocks" as substring)
        if "single_transformer_blocks." in lora_layer_name:
            parts = lora_layer_name.split(".")
            block_idx = parts[1]
            layer_type = ".".join(parts[2:])

            # Get model layer name
            if layer_type in SINGLE_BLOCK_MAP:
                model_layer = f"single_transformer_blocks.{block_idx}.{SINGLE_BLOCK_MAP[layer_type]}"
            else:
                logger.warning(f"Unknown single block layer type: {layer_type}")
                continue

        elif "transformer_blocks." in lora_layer_name:
            # Extract block index and layer type
            parts = lora_layer_name.split(".")
            block_idx = parts[1]
            layer_type = ".".join(parts[2:])

            # Get model layer name
            if layer_type in TRANSFORMER_BLOCK_MAP:
                model_layer = f"transformer_blocks.{block_idx}.{TRANSFORMER_BLOCK_MAP[layer_type]}"
            else:
                logger.warning(f"Unknown transformer block layer type: {layer_type}")
                continue
        else:
            logger.warning(f"Unknown layer format: {lora_layer_name}")
            continue

        # Get original SVD weights
        down_key = f"{model_layer}.proj_down"
        up_key = f"{model_layer}.proj_up"

        if down_key not in model_state_dict or up_key not in model_state_dict:
            logger.warning(f"Could not find SVD weights for {model_layer}")
            continue

        orig_down_packed = model_state_dict[down_key]
        orig_up_packed = model_state_dict[up_key]

        # Unpack original weights
        orig_down = unpack_lowrank_weight(orig_down_packed, down=True)
        orig_up = unpack_lowrank_weight(orig_up_packed, down=False)

        # Validate dimension compatibility
        # lora_A: [lora_rank, in_features], orig_down: [rank, in_features]
        # lora_B: [out_features, lora_rank], orig_up: [out_features, rank]
        if lora_A.shape[1] != orig_down.shape[1]:
            logger.warning(
                f"Dimension mismatch for {model_layer}: "
                f"LoRA lora_A has in_features={lora_A.shape[1]}, "
                f"model proj_down has in_features={orig_down.shape[1]}. Skipping layer."
            )
            continue

        if lora_B.shape[0] != orig_up.shape[0]:
            logger.warning(
                f"Dimension mismatch for {model_layer}: "
                f"LoRA lora_B has out_features={lora_B.shape[0]}, "
                f"model proj_up has out_features={orig_up.shape[0]}. Skipping layer."
            )
            continue

        # Merge LoRA with SVD branch
        new_down, new_up = merge_lora_with_svd_branch(orig_down, orig_up, lora_A, lora_B, strength)

        # Pad rank to multiple of 16 (required for CUDA kernels)
        # new_down: [rank + lora_rank, in_features], pad dim=0 (rank dimension)
        # new_up: [out_features, rank + lora_rank], pad dim=1 (rank dimension)
        new_rank = new_down.shape[0]
        if new_rank % 16 != 0:
            new_down = pad(new_down, divisor=16, dim=0)
            new_up = pad(new_up, divisor=16, dim=1)

        # Pack for Nunchaku
        new_down_packed = pack_lowrank_weight(new_down, down=True)
        new_up_packed = pack_lowrank_weight(new_up, down=False)

        result[down_key] = new_down_packed
        result[up_key] = new_up_packed

        logger.debug(f"Merged LoRA for {model_layer}: rank {orig_down.shape[0]} + {lora_A.shape[0]} -> {new_down.shape[0]}")

    return result


def to_nunchaku(
    input_lora: str | dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
    strength: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Convert LoRA weights to Nunchaku format.

    Parameters
    ----------
    input_lora : str or dict[str, torch.Tensor]
        Path to a safetensors file or a LoRA weight dictionary.
    model_state_dict : dict[str, torch.Tensor]
        Current model state dict with SVD weights.
    strength : float
        LoRA strength multiplier.

    Returns
    -------
    dict[str, torch.Tensor]
        Updated weights for the affected layers.
    """
    # Convert to intermediate format
    if isinstance(input_lora, str):
        lora_dict = to_diffusers(input_lora)
    else:
        lora_dict = to_diffusers(input_lora)

    # Convert to Nunchaku format
    return convert_chroma_lora_to_nunchaku(lora_dict, model_state_dict, strength)
