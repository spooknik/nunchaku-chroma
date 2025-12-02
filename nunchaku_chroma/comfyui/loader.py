"""
This module provides the NunchakuChromaDiTLoader class for loading Nunchaku Chroma models in ComfyUI.
"""

import gc
import logging
import os

import torch


# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Default Chroma config - based on Flux schnell architecture but with Chroma's settings
# The key is "disable_unet_model_creation": True to prevent ComfyUI from creating a real Flux model
DEFAULT_CHROMA_CONFIG = {
    "axes_dim": [16, 56, 56],
    "context_in_dim": 4096,
    "depth": 19,
    "depth_single_blocks": 38,
    "disable_unet_model_creation": True,
    "guidance_embed": False,  # Chroma doesn't use guidance embed
    "hidden_size": 3072,
    "image_model": "flux",
    "in_channels": 64,  # Chroma uses 64 channels (packed latents)
    "mlp_ratio": 4.0,
    "num_heads": 24,
    "out_channels": 64,
    "patch_size": 1,  # Chroma uses patch_size=1
    "qkv_bias": True,
    "theta": 10000,
    "vec_in_dim": 768,
}


class NunchakuChromaDiTLoader:
    """
    Loader for Nunchaku Chroma models.

    This class manages model loading, device selection, attention implementation,
    and CPU offload for efficient inference.

    Attributes
    ----------
    transformer : NunchakuChromaTransformer2DModel or None
        The loaded transformer model.
    model_path : str or None
        Path to the loaded model.
    device : torch.device or None
        Device on which the model is loaded.
    data_type : str or None
        Data type used for inference.
    """

    def __init__(self):
        """
        Initialize the NunchakuChromaDiTLoader.

        Sets up internal state and selects the default torch device.
        """
        self.transformer = None
        self.model_path = None
        self.device = None
        self.data_type = None

        # Import here to avoid issues when not running in ComfyUI
        import comfy.model_management
        self.device = comfy.model_management.get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for the node interface.
        """
        import folder_paths
        from nunchaku.utils import is_turing

        safetensor_files = folder_paths.get_filename_list("diffusion_models")

        ngpus = torch.cuda.device_count()

        all_turing = True
        for i in range(torch.cuda.device_count()):
            if not is_turing(f"cuda:{i}"):
                all_turing = False

        if all_turing:
            attention_options = ["nunchaku-fp16"]  # turing GPUs do not support flashattn2
            dtype_options = ["float16"]
        else:
            attention_options = ["nunchaku-fp16", "flash-attention2"]
            dtype_options = ["bfloat16", "float16"]

        return {
            "required": {
                "model_path": (
                    safetensor_files,
                    {"tooltip": "The Nunchaku Chroma model."},
                ),
                "attention": (
                    attention_options,
                    {
                        "default": attention_options[0],
                        "tooltip": (
                            "Attention implementation. The default implementation is `flash-attention2`. "
                            "`nunchaku-fp16` use FP16 attention, offering ~1.2× speedup. "
                            "Note that 20-series GPUs can only use `nunchaku-fp16`."
                        ),
                    },
                ),
                "device_id": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": ngpus - 1,
                        "step": 1,
                        "display": "number",
                        "lazy": True,
                        "tooltip": "The GPU device ID to use for the model.",
                    },
                ),
                "data_type": (
                    dtype_options,
                    {
                        "default": dtype_options[0],
                        "tooltip": "Specifies the model's data type. Default is `bfloat16`. "
                        "For 20-series GPUs, which do not support `bfloat16`, use `float16` instead.",
                    },
                ),
            },
            "optional": {
                "i2f_mode": (
                    ["enabled", "always"],
                    {
                        "default": "enabled",
                        "tooltip": "The GEMM implementation for 20-series GPUs"
                        "— this option is only applicable to these GPUs.",
                    },
                )
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Chroma DiT Loader"

    def load_model(
        self,
        model_path: str,
        attention: str,
        device_id: int,
        data_type: str,
        **kwargs,
    ):
        """
        Load a Nunchaku Chroma model with the specified configuration.

        Parameters
        ----------
        model_path : str
            Path to the model directory or safetensors file.
        attention : str
            Attention implementation to use ("nunchaku-fp16" or "flash-attention2").
        device_id : int
            GPU device ID to use.
        data_type : str
            Data type for inference ("bfloat16" or "float16").
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tuple
            A tuple containing the loaded and patched model.
        """
        import comfy.model_management
        import comfy.model_patcher
        import folder_paths
        from comfy.supported_models import FluxSchnell
        from nunchaku_chroma import NunchakuChromaTransformer2DModel
        from .wrapper import ComfyChromaWrapper

        device = torch.device(f"cuda:{device_id}")

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_path)

        # Check if the device_id is valid
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid device_id: {device_id}. Only {torch.cuda.device_count()} GPUs available.")

        # Get the GPU properties
        gpu_properties = torch.cuda.get_device_properties(device_id)
        gpu_memory = gpu_properties.total_memory / (1024**2)  # Convert to MiB
        gpu_name = gpu_properties.name
        logger.debug(f"GPU {device_id} ({gpu_name}) Memory: {gpu_memory} MiB")

        if (
            self.model_path != model_path
            or self.device != device
            or self.data_type != data_type
        ):
            if self.transformer is not None:
                model_size = comfy.model_management.module_size(self.transformer)
                transformer = self.transformer
                self.transformer = None
                transformer.to("cpu")
                del transformer
                gc.collect()
                comfy.model_management.cleanup_models_gc()
                comfy.model_management.soft_empty_cache()
                comfy.model_management.free_memory(model_size, device)

            self.transformer = NunchakuChromaTransformer2DModel.from_pretrained(
                model_path,
                device=device,
                torch_dtype=torch.float16 if data_type == "float16" else torch.bfloat16,
            )
            self.model_path = model_path
            self.device = device
            self.data_type = data_type

        transformer = self.transformer

        # Set attention implementation
        if attention == "nunchaku-fp16":
            transformer.set_attention_impl("nunchaku-fp16")
        else:
            assert attention == "flash-attention2"
            transformer.set_attention_impl("flashattn2")

        # Use FluxSchnell as the base model config since Chroma is based on FLUX schnell
        model_config = FluxSchnell(DEFAULT_CHROMA_CONFIG)
        model_config.set_inference_dtype(torch.bfloat16, None)
        model_config.custom_operations = None
        model = model_config.get_model({})
        model.diffusion_model = ComfyChromaWrapper(transformer, config=DEFAULT_CHROMA_CONFIG)
        model = comfy.model_patcher.ModelPatcher(model, device, device_id)
        return (model,)
