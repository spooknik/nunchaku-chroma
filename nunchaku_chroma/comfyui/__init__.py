"""
ComfyUI integration for Nunchaku Chroma.

This module provides ComfyUI nodes for loading and using Nunchaku Chroma models.
"""

from .loader import NunchakuChromaDiTLoader, DEFAULT_CHROMA_CONFIG
from .lora import NunchakuChromaLoraLoader, NunchakuChromaLoraStack
from .wrapper import ComfyChromaWrapper

__all__ = [
    "NunchakuChromaDiTLoader",
    "NunchakuChromaLoraLoader",
    "NunchakuChromaLoraStack",
    "ComfyChromaWrapper",
    "DEFAULT_CHROMA_CONFIG",
]

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "NunchakuChromaDiTLoader": NunchakuChromaDiTLoader,
    "NunchakuChromaLoraLoader": NunchakuChromaLoraLoader,
    "NunchakuChromaLoraStack": NunchakuChromaLoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NunchakuChromaDiTLoader": "Nunchaku Chroma DiT Loader",
    "NunchakuChromaLoraLoader": "Nunchaku Chroma LoRA Loader",
    "NunchakuChromaLoraStack": "Nunchaku Chroma LoRA Stack",
}
