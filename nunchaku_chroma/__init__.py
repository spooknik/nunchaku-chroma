"""
Nunchaku Chroma - Adds Chroma model support to Nunchaku.

This package provides:
- NunchakuChromaTransformer2DModel: Quantized Chroma transformer for diffusers
- ComfyUI integration: Wrapper and loader nodes for ComfyUI
- LoRA support: Chroma-specific LoRA converters and composition
"""

__version__ = "0.1.0"

from .transformer_chroma import (
    NunchakuChromaTransformer2DModel,
    NunchakuChromaAttention,
    NunchakuChromaTransformerBlock,
    NunchakuChromaSingleTransformerBlock,
    convert_chroma_state_dict,
)

__all__ = [
    "NunchakuChromaTransformer2DModel",
    "NunchakuChromaAttention",
    "NunchakuChromaTransformerBlock",
    "NunchakuChromaSingleTransformerBlock",
    "convert_chroma_state_dict",
    "__version__",
]
