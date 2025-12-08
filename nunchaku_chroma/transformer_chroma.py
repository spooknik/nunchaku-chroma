"""
This module provides Nunchaku ChromaTransformer2DModel and its building blocks in Python.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_chroma import (
    ChromaSingleTransformerBlock,
    ChromaTransformer2DModel,
    ChromaTransformerBlock,
)
from diffusers.models.transformers.transformer_flux import FluxAttention
from huggingface_hub import utils
from torch import nn
from torch.nn import GELU

from nunchaku.ops.fused import fused_gelu_mlp
from nunchaku.utils import get_precision, pad_tensor, check_hardware_compatibility
from nunchaku.models.attention import NunchakuBaseAttention, NunchakuFeedForward
from nunchaku.models.attention_processors.flux import NunchakuFluxFA2Processor, NunchakuFluxFP16AttnProcessor
from nunchaku.models.embeddings import NunchakuFluxPosEmbed, pack_rotemb
from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.models.utils import fuse_linears
from nunchaku.models.transformers.utils import NunchakuModelLoaderMixin


class NunchakuChromaAttention(NunchakuBaseAttention):
    """
    Nunchaku-optimized ChromaAttention module with quantized and fused QKV projections.

    Chroma reuses FluxAttention, so we can reuse the same attention processor implementations.

    Parameters
    ----------
    other : FluxAttention
        The original FluxAttention module to wrap and quantize.
    processor : str, optional
        The attention processor to use ("flashattn2" or "nunchaku-fp16").
    skip_quantize_to_out : bool, optional
        If True, keep the to_out projection as a regular nn.Linear instead of quantizing.
        This supports mixed precision models where some layers are kept at full precision.
    skip_quantize_to_add_out : bool, optional
        If True, keep the to_add_out projection as a regular nn.Linear instead of quantizing.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, other: FluxAttention, processor: str = "flashattn2", skip_quantize_to_out: bool = False, skip_quantize_to_add_out: bool = False, **kwargs):
        super(NunchakuChromaAttention, self).__init__(processor)
        self.head_dim = other.head_dim
        self.inner_dim = other.inner_dim
        self.query_dim = other.query_dim
        self.use_bias = other.use_bias
        self.dropout = other.dropout
        self.out_dim = other.out_dim
        self.context_pre_only = other.context_pre_only
        self.pre_only = other.pre_only
        self.heads = other.heads
        self.added_kv_proj_dim = other.added_kv_proj_dim
        self.added_proj_bias = other.added_proj_bias

        self.norm_q = other.norm_q
        self.norm_k = other.norm_k

        # Fuse the QKV projections for efficiency.
        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)

        if not self.pre_only:
            self.to_out = other.to_out
            if skip_quantize_to_out:
                # Keep as regular nn.Linear for mixed precision models
                pass
            else:
                self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

        if self.added_kv_proj_dim is not None:
            self.norm_added_q = other.norm_added_q
            self.norm_added_k = other.norm_added_k

            # Fuse the additional QKV projections.
            with torch.device("meta"):
                add_qkv_proj = fuse_linears([other.add_q_proj, other.add_k_proj, other.add_v_proj])
            self.add_qkv_proj = SVDQW4A4Linear.from_linear(add_qkv_proj, **kwargs)
            if skip_quantize_to_add_out:
                # Keep as regular nn.Linear for mixed precision models
                self.to_add_out = other.to_add_out
            else:
                self.to_add_out = SVDQW4A4Linear.from_linear(other.to_add_out, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor = None,
        **kwargs,
    ):
        """
        Forward pass for NunchakuChromaAttention.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor.
        encoder_hidden_states : torch.Tensor, optional
            Encoder hidden states for cross-attention.
        attention_mask : torch.Tensor, optional
            Attention mask.
        image_rotary_emb : tuple or torch.Tensor, optional
            Rotary embeddings for image/text tokens.
        **kwargs
            Additional arguments.

        Returns
        -------
        Output of the attention processor.
        """
        return self.processor(
            attn=self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

    def set_processor(self, processor: str):
        """
        Set the attention processor.

        Parameters
        ----------
        processor : str
            Name of the processor ("flashattn2" or "nunchaku-fp16").

            - ``"flashattn2"``: Standard FlashAttention-2.
            - ``"nunchaku-fp16"``: Uses FP16 attention accumulation, up to 1.2× faster.

        Raises
        ------
        ValueError
            If the processor is not supported.
        """
        if processor == "flashattn2":
            self.processor = NunchakuFluxFA2Processor()
        elif processor == "nunchaku-fp16":
            self.processor = NunchakuFluxFP16AttnProcessor()
        else:
            raise ValueError(f"Processor {processor} is not supported")


class NunchakuChromaTransformerBlock(ChromaTransformerBlock):
    """
    Nunchaku-optimized ChromaTransformerBlock with quantized attention and feedforward layers.

    Parameters
    ----------
    block : ChromaTransformerBlock
        The original block to wrap and quantize.
    skip_quantize_to_out : bool, optional
        If True, keep the attention to_out projection as regular nn.Linear.
    skip_quantize_to_add_out : bool, optional
        If True, keep the attention to_add_out projection as regular nn.Linear.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, block: ChromaTransformerBlock, skip_quantize_to_out: bool = False, skip_quantize_to_add_out: bool = False, **kwargs):
        super(ChromaTransformerBlock, self).__init__()

        # Chroma's pruned norm layers don't have silu+linear, keep them as-is
        self.norm1 = block.norm1
        self.norm1_context = block.norm1_context

        self.attn = NunchakuChromaAttention(block.attn, skip_quantize_to_out=skip_quantize_to_out, skip_quantize_to_add_out=skip_quantize_to_add_out, **kwargs)
        self.norm2 = block.norm2
        self.norm2_context = block.norm2_context
        self.ff = NunchakuFeedForward(block.ff, **kwargs)
        self.ff_context = NunchakuFeedForward(block.ff_context, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the transformer block.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states.
        encoder_hidden_states : torch.Tensor
            Encoder hidden states for cross-attention.
        temb : torch.Tensor
            Time/conditioning embedding (pre-sliced for img/txt).
        image_rotary_emb : tuple of torch.Tensor, optional
            Rotary embeddings for image/text tokens.
        attention_mask : torch.Tensor, optional
            Attention mask.
        joint_attention_kwargs : dict, optional
            Additional attention arguments (not supported).

        Returns
        -------
        tuple
            (encoder_hidden_states, hidden_states) after block processing.

        Raises
        ------
        NotImplementedError
            If joint_attention_kwargs is provided.
        """
        if joint_attention_kwargs is not None and len(joint_attention_kwargs) > 0:
            raise NotImplementedError("joint_attention_kwargs is not supported")

        # Chroma splits temb into img and txt parts
        temb_img, temb_txt = temb[:, :6], temb[:, 6:]
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb_img)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb_txt
        )

        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class NunchakuChromaSingleTransformerBlock(ChromaSingleTransformerBlock):
    """
    Nunchaku-optimized single transformer block with quantized attention and MLP.

    Parameters
    ----------
    block : ChromaSingleTransformerBlock
        The original block to wrap and quantize.
    skip_quantize_to_out : bool, optional
        If True, keep the attention to_out projection as regular nn.Linear.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, block: ChromaSingleTransformerBlock, skip_quantize_to_out: bool = False, **kwargs):
        super(ChromaSingleTransformerBlock, self).__init__()
        self.mlp_hidden_dim = block.mlp_hidden_dim

        # Chroma's pruned norm layer - keep as-is
        self.norm = block.norm

        # MLP path: dim -> mlp_hidden_dim -> dim (split projections like FLUX)
        self.mlp_fc1 = SVDQW4A4Linear.from_linear(block.proj_mlp, **kwargs)
        self.act_mlp = block.act_mlp
        # mlp_fc2: mlp_hidden_dim -> dim (NOT concatenated input)
        self.mlp_fc2 = SVDQW4A4Linear.from_linear(block.proj_out, in_features=self.mlp_hidden_dim, **kwargs)
        # For int4, shift the activation of mlp_fc2 to make it unsigned.
        self.mlp_fc2.act_unsigned = self.mlp_fc2.precision != "nvfp4"

        # Attention with separate output projection
        self.attn = NunchakuChromaAttention(block.attn, **kwargs)
        # attn.to_out: dim -> dim
        if skip_quantize_to_out:
            # Keep as regular nn.Linear for mixed precision models
            # Create a new Linear layer with the correct dimensions
            self.attn.to_out = nn.Linear(block.proj_mlp.in_features, block.proj_out.out_features, bias=block.proj_out.bias is not None)
        else:
            self.attn.to_out = SVDQW4A4Linear.from_linear(block.proj_out, in_features=block.proj_mlp.in_features, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the single transformer block.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states.
        temb : torch.Tensor
            Time or conditioning embedding.
        image_rotary_emb : tuple of torch.Tensor, optional
            Rotary embeddings for tokens.
        attention_mask : torch.Tensor, optional
            Attention mask.
        joint_attention_kwargs : dict, optional
            Additional attention arguments.

        Returns
        -------
        torch.Tensor
            Output hidden states after block processing.
        """
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        # MLP path: dim -> mlp_hidden_dim -> dim
        if isinstance(self.act_mlp, GELU):
            # Use fused GELU MLP for efficiency.
            mlp_output = fused_gelu_mlp(norm_hidden_states, self.mlp_fc1, self.mlp_fc2)
        else:
            # Fallback to original MLP.
            mlp_hidden_states = self.mlp_fc1(norm_hidden_states)
            mlp_hidden_states = self.act_mlp(mlp_hidden_states)
            mlp_output = self.mlp_fc2(mlp_hidden_states)

        # Attention path (includes to_out projection)
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            **joint_attention_kwargs,
        )

        # ADD outputs (not concatenate) - FLUX-style split architecture
        hidden_states = attn_output + mlp_output
        gate = gate.unsqueeze(1)
        hidden_states = gate * hidden_states

        # Residual connection
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class NunchakuChromaTransformer2DModel(ChromaTransformer2DModel, NunchakuModelLoaderMixin):
    """
    Nunchaku-optimized ChromaTransformer2DModel with LoRA support.

    Attributes
    ----------
    comfy_lora_meta_list : list
        List of (lora_path, strength) tuples for tracking loaded LoRAs in ComfyUI.
    comfy_lora_sd_list : list
        List of state dicts for each loaded LoRA.
    _original_in_channels : int
        Original input channel count before any LoRA modifications.
    _base_state_dict : dict
        Original state dict for resetting LoRA effects.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ComfyUI LoRA tracking
        self.comfy_lora_meta_list = []
        self.comfy_lora_sd_list = []
        self._original_in_channels = None
        self._base_state_dict = None

    def _patch_model(self, non_quantized_layers: set = None, **kwargs):
        """
        Patch the model with quantized transformer blocks.

        Parameters
        ----------
        non_quantized_layers : set, optional
            Set of layer prefixes that should remain as regular nn.Linear layers
            instead of being quantized. Example: {"transformer_blocks.0.out_proj"}
        **kwargs
            Additional arguments for quantization.

        Returns
        -------
        self : NunchakuChromaTransformer2DModel
            The patched model.
        """
        non_quantized_layers = non_quantized_layers or set()

        self.pos_embed = NunchakuFluxPosEmbed(dim=self.inner_dim, theta=10000, axes_dim=self.pos_embed.axes_dim)
        for i, block in enumerate(self.transformer_blocks):
            # Check if out_proj for this block should be kept as non-quantized
            skip_to_out = f"transformer_blocks.{i}.out_proj" in non_quantized_layers
            skip_to_add_out = f"transformer_blocks.{i}.out_proj_context" in non_quantized_layers
            self.transformer_blocks[i] = NunchakuChromaTransformerBlock(
                block, skip_quantize_to_out=skip_to_out, skip_quantize_to_add_out=skip_to_add_out, **kwargs
            )
        for i, block in enumerate(self.single_transformer_blocks):
            # Check if out_proj for this block should be kept as non-quantized
            skip_to_out = f"single_transformer_blocks.{i}.out_proj" in non_quantized_layers
            self.single_transformer_blocks[i] = NunchakuChromaSingleTransformerBlock(
                block, skip_quantize_to_out=skip_to_out, **kwargs
            )
        return self

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a pretrained NunchakuChromaTransformer2DModel from a safetensors file.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the safetensors file. It can be a local file or a remote HuggingFace path.
        **kwargs
            Additional arguments (e.g., device, torch_dtype).

        Returns
        -------
        NunchakuChromaTransformer2DModel
            The loaded and quantized model.

        Raises
        ------
        NotImplementedError
            If offload is requested.
        AssertionError
            If the file is not a safetensors file.
        """
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)

        if offload:
            raise NotImplementedError("Offload is not supported for ChromaTransformer2DModel")

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"
        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))

        # Check hardware compatibility (INT4 vs FP4)
        if quantization_config:
            check_hardware_compatibility(quantization_config, device)

        rank = quantization_config.get("rank", 32)
        transformer = transformer.to(torch_dtype)

        # Detect non-quantized layers (layers with .weight but no .qweight)
        non_quantized_layers = detect_non_quantized_layers(model_state_dict)

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"
        transformer._patch_model(precision=precision, rank=rank, non_quantized_layers=non_quantized_layers)

        transformer = transformer.to_empty(device=device)

        # Reinitialize the time_text_embed.mod_proj buffer which is lost during to_empty()
        # because it's registered with persistent=False
        # The mod_proj is a computed value (sinusoidal timestep embedding)
        from diffusers.models.embeddings import get_timestep_embedding

        out_dim = transformer.time_text_embed.mod_proj.shape[0]
        num_channels = transformer.time_text_embed.time_proj.num_channels
        transformer.time_text_embed.mod_proj = get_timestep_embedding(
            torch.arange(out_dim) * 1000, 2 * num_channels, flip_sin_to_cos=True, downscale_freq_shift=0
        ).to(device=device, dtype=torch_dtype)

        converted_state_dict = convert_chroma_state_dict(model_state_dict, non_quantized_layers)

        state_dict = transformer.state_dict()

        for k in state_dict.keys():
            if k not in converted_state_dict:
                if ".wcscales" in k:
                    converted_state_dict[k] = torch.ones_like(state_dict[k])
                else:
                    raise KeyError(f"Missing key in converted state dict: {k}")

        # Load state dict with assign=True to handle dtype mismatches for non-quantized layers
        # This allows the model to accept weights of different dtypes and assign them directly

        # Load the wtscale from the converted state dict first (before load_state_dict).
        for n, m in transformer.named_modules():
            if isinstance(m, SVDQW4A4Linear):
                if m.wtscale is not None:
                    m.wtscale = converted_state_dict.pop(f"{n}.wtscale", 1.0)

        transformer.load_state_dict(converted_state_dict, assign=True)

        # Store original in_channels and base state for LoRA support
        transformer._original_in_channels = transformer.config.in_channels
        transformer._base_state_dict = {k: v.clone() for k, v in converted_state_dict.items()}

        return transformer

    def set_attention_impl(self, impl: str):
        """
        Set the attention implementation for all transformer blocks.

        Parameters
        ----------
        impl : str
            Attention implementation to use. Supported values:

            - ``"flashattn2"`` (default): Standard FlashAttention-2.
            - ``"nunchaku-fp16"``: Uses FP16 attention accumulation, up to 1.2× faster.

        """
        for block in self.transformer_blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'set_processor'):
                block.attn.set_processor(impl)
        for block in self.single_transformer_blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'set_processor'):
                block.attn.set_processor(impl)

    def _init_lora_state(self, store_originals: bool = True):
        """
        Initialize LoRA tracking state if not already done.

        Parameters
        ----------
        store_originals : bool, optional
            Whether to store original SVD weights for reset capability.
            Set to False to save memory if you don't need reset_lora() or
            set_lora_strength(). Default is True.
        """
        if not hasattr(self, '_original_in_channels'):
            self._original_in_channels = self.config.in_channels

        # ComfyUI LoRA tracking lists
        if not hasattr(self, 'comfy_lora_meta_list'):
            self.comfy_lora_meta_list = []
        if not hasattr(self, 'comfy_lora_sd_list'):
            self.comfy_lora_sd_list = []

        # Store original SVD weights for reset (only if requested and not already stored)
        if not hasattr(self, '_original_svd_weights'):
            self._original_svd_weights = None

        if store_originals and self._original_svd_weights is None:
            self._original_svd_weights = {}
            for name, module in self.named_modules():
                if isinstance(module, SVDQW4A4Linear):
                    self._original_svd_weights[f"{name}.proj_down"] = module.proj_down.data.clone()
                    self._original_svd_weights[f"{name}.proj_up"] = module.proj_up.data.clone()

    def update_lora_params(self, lora: str | dict[str, torch.Tensor], strength: float = 1.0, store_originals: bool = True):
        """
        Update the model with new LoRA parameters.

        This method merges LoRA weights with the existing SVD low-rank branch.
        The LoRA weights are converted to Nunchaku format and merged with the
        existing proj_down/proj_up tensors.

        Parameters
        ----------
        lora : str or dict[str, torch.Tensor]
            Either a path to a safetensors file, or a state dict containing LoRA weights.
            Keys should be in the format: ``"layer_name.lora_A.weight"`` and
            ``"layer_name.lora_B.weight"``.
        strength : float, optional
            LoRA strength multiplier. Default is 1.0.
        store_originals : bool, optional
            Whether to store original SVD weights for reset capability.
            Set to False to save memory if you don't need ``reset_lora()`` or
            ``set_lora_strength()``. Default is True.

        Notes
        -----
        Setting ``store_originals=False`` saves approximately 200-400MB of GPU memory
        but disables the ability to reset or adjust LoRA strength after application.
        """
        from nunchaku.lora.chroma.diffusers_converter import to_diffusers
        from nunchaku.lora.chroma.nunchaku_converter import convert_chroma_lora_to_nunchaku

        logger = logging.getLogger(__name__)
        self._init_lora_state(store_originals=store_originals)

        # Handle file path input
        if isinstance(lora, str):
            state_dict = to_diffusers(lora)
        else:
            state_dict = lora

        # Store for dynamic strength adjustment
        self._current_lora_dict = state_dict
        self._current_strength = strength

        # First reset to original weights
        self._restore_original_weights()

        # Get current model state dict (after reset to original)
        model_state_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, SVDQW4A4Linear):
                model_state_dict[f"{name}.proj_down"] = module.proj_down.data
                model_state_dict[f"{name}.proj_up"] = module.proj_up.data

        # Convert and merge LoRA
        try:
            updated_weights = convert_chroma_lora_to_nunchaku(state_dict, model_state_dict, strength=strength)

            # Apply updated weights
            for key, value in updated_weights.items():
                parts = key.rsplit(".", 1)
                if len(parts) == 2:
                    module_path, param_name = parts
                    # Find the module
                    module = self
                    for part in module_path.split("."):
                        if part.isdigit():
                            module = module[int(part)]
                        else:
                            module = getattr(module, part)

                    if hasattr(module, param_name):
                        param = getattr(module, param_name)
                        param.data = value.to(param.device, param.dtype)

            logger.info(f"Applied Chroma LoRA: updated {len(updated_weights)} weight tensors")
        except Exception as e:
            logger.error(f"Failed to apply Chroma LoRA: {e}")
            raise

    def _restore_original_weights(self):
        """Restore original SVD weights before applying new LoRA."""
        if not hasattr(self, '_original_svd_weights') or self._original_svd_weights is None:
            return False

        for key, value in self._original_svd_weights.items():
            parts = key.rsplit(".", 1)
            if len(parts) == 2:
                module_path, param_name = parts
                module = self
                for part in module_path.split("."):
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)

                if hasattr(module, param_name):
                    param = getattr(module, param_name)
                    param.data = value.clone().to(param.device, param.dtype)

        return True

    def set_lora_strength(self, strength: float = 1.0):
        """
        Adjust LoRA strength without reloading from disk.

        This re-merges the currently loaded LoRA with a new strength value.
        Requires that a LoRA has already been applied via ``update_lora_params()``
        with ``store_originals=True``.

        Parameters
        ----------
        strength : float, optional
            New LoRA strength multiplier. Default is 1.0.
        """
        logger = logging.getLogger(__name__)

        if not hasattr(self, '_original_svd_weights') or self._original_svd_weights is None:
            logger.warning(
                "Cannot adjust LoRA strength: original weights not stored. "
                "Use update_lora_params(..., store_originals=True) to enable this feature."
            )
            return

        if not hasattr(self, '_current_lora_dict') or self._current_lora_dict is None:
            logger.warning("No LoRA currently loaded. Call update_lora_params() first.")
            return

        # Re-apply with new strength
        self.update_lora_params(self._current_lora_dict, strength=strength)

    def reset_lora(self):
        """
        Reset all LoRA parameters to their default state.

        This restores the original SVD weights that were saved when the model
        was first loaded. Requires that LoRA was applied with ``store_originals=True``.
        """
        logger = logging.getLogger(__name__)
        self._init_lora_state(store_originals=False)  # Don't create originals if they don't exist

        if not self._restore_original_weights():
            logger.warning(
                "Cannot reset LoRA: original weights not stored. "
                "Use update_lora_params(..., store_originals=True) to enable reset."
            )
            return

        # Clear stored LoRA state
        self._current_lora_dict = None
        self._current_strength = None

        logger.info("Reset Chroma model to original SVD weights (LoRA removed)")

    def reset_x_embedder(self):
        """
        Reset the x_embedder module if the input channel count has changed.
        """
        self._init_lora_state()
        # For Chroma, x_embedder is typically not modified by LoRA
        pass

    def update_lora_params_multi(self, loras: list[tuple[str | dict[str, torch.Tensor], float]]):
        """
        Update the model with multiple composed LoRA parameters.

        This method composes multiple LoRAs with their respective strengths into a single
        LoRA representation, then applies it to the model.

        Parameters
        ----------
        loras : list of tuple
            List of (lora, strength) tuples where:
            - lora: Either a path to a safetensors file or a LoRA state dict
            - strength: Float strength/scale factor for that LoRA

        Examples
        --------
        >>> model.update_lora_params_multi([
        ...     ("lora1.safetensors", 0.8),
        ...     ("lora2.safetensors", 0.5),
        ... ])
        """
        from nunchaku.lora.chroma.compose import compose_lora

        logger = logging.getLogger(__name__)

        if len(loras) == 0:
            self.reset_lora()
            return

        # Compose all LoRAs into one
        composed_lora = compose_lora(loras)

        # Store for reference (use composed result with effective strength=1.0)
        self._current_lora_list = loras

        # Apply the composed LoRA with strength=1.0 (strengths already baked in)
        self.update_lora_params(composed_lora, strength=1.0)

        lora_info = ", ".join([f"{l[0] if isinstance(l[0], str) else 'dict'}@{l[1]}" for l in loras])
        logger.info(f"Applied {len(loras)} composed Chroma LoRAs: {lora_info}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass for the NunchakuChromaTransformer2DModel.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states of shape (batch_size, image_sequence_length, in_channels).
        encoder_hidden_states : torch.Tensor, optional
            Conditional embeddings (e.g., from text).
        timestep : torch.LongTensor, optional
            Denoising step.
        img_ids : torch.Tensor, optional
            Image token IDs.
        txt_ids : torch.Tensor, optional
            Text token IDs.
        attention_mask : torch.Tensor, optional
            Attention mask.
        joint_attention_kwargs : dict, optional
            Additional attention arguments.
        controlnet_block_samples : any, optional
            ControlNet residuals for transformer blocks.
        controlnet_single_block_samples : any, optional
            ControlNet residuals for single transformer blocks.
        return_dict : bool, optional
            Whether to return a Transformer2DModelOutput (default: True).
        controlnet_blocks_repeat : bool, optional
            Whether to repeat controlnet blocks.

        Returns
        -------
        Transformer2DModelOutput or tuple
            Output sample tensor or output tuple.
        """
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000

        input_vec = self.time_text_embed(timestep)
        pooled_temb = self.distilled_guidance_layer(input_vec)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        assert image_rotary_emb.shape[2] == 1 * (txt_tokens + img_tokens)
        # [1, tokens, head_dim / 2, 1, 2] (sincos)
        image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]
        rotary_emb_single = image_rotary_emb

        rotary_emb_txt = pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
        rotary_emb_img = pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
        rotary_emb_single = pack_rotemb(pad_tensor(rotary_emb_single, 256, 1))

        for index_block, block in enumerate(self.transformer_blocks):
            # Chroma-specific temb slicing
            img_offset = 3 * len(self.single_transformer_blocks)
            txt_offset = img_offset + 6 * len(self.transformer_blocks)
            img_modulation = img_offset + 6 * index_block
            text_modulation = txt_offset + 6 * index_block
            temb = torch.cat(
                (
                    pooled_temb[:, img_modulation : img_modulation + 6],
                    pooled_temb[:, text_modulation : text_modulation + 6],
                ),
                dim=1,
            )

            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
                attention_mask=attention_mask,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # Controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            # Chroma-specific temb slicing for single blocks
            start_idx = 3 * index_block
            temb = pooled_temb[:, start_idx : start_idx + 3]

            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=rotary_emb_single,
                attention_mask=attention_mask,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # Controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        # Chroma uses the last 2 elements of pooled_temb for output norm
        temb = pooled_temb[:, -2:]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def detect_non_quantized_layers(state_dict: dict[str, torch.Tensor]) -> set[str]:
    """
    Detect layers in the state dict that are not quantized.

    A layer is considered non-quantized if it has a .weight key but no corresponding .qweight key.
    This is used to support mixed precision models where some layers are kept at full precision.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        The model state dict to analyze.

    Returns
    -------
    set[str]
        Set of layer prefixes that are non-quantized (e.g., {"transformer_blocks.0.out_proj"}).
    """
    # Collect all layer prefixes that have .weight
    layers_with_weight = set()
    layers_with_qweight = set()

    for k in state_dict.keys():
        if k.endswith(".weight"):
            # Extract the layer prefix (everything before .weight)
            prefix = k[:-7]  # Remove ".weight"
            layers_with_weight.add(prefix)
        elif k.endswith(".qweight"):
            # Extract the layer prefix (everything before .qweight)
            prefix = k[:-8]  # Remove ".qweight"
            layers_with_qweight.add(prefix)

    # Non-quantized layers are those with .weight but no .qweight
    # Only consider transformer block layers that could be quantized
    non_quantized = set()
    for prefix in layers_with_weight:
        if prefix not in layers_with_qweight:
            # Only track layers in transformer_blocks or single_transformer_blocks
            # that are candidates for quantization (out_proj, qkv_proj, mlp_fc, out_proj_context)
            if (prefix.startswith("transformer_blocks.") or prefix.startswith("single_transformer_blocks.")):
                if any(pattern in prefix for pattern in [".out_proj", ".qkv_proj", ".mlp_fc", ".out_proj_context"]):
                    non_quantized.add(prefix)

    return non_quantized


def convert_chroma_state_dict(state_dict: dict[str, torch.Tensor], non_quantized_layers: set[str] = None) -> dict[str, torch.Tensor]:
    """
    Convert a state dict from the quantized Chroma format to NunchakuChromaTransformer2DModel format.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        The original state dict.
    non_quantized_layers : set[str], optional
        Set of layer prefixes that are non-quantized. This is used for informational purposes
        and does not affect the key conversion logic since non-quantized layers simply have
        different keys (.weight/.bias instead of .qweight/.wscales/etc.).

    Returns
    -------
    dict[str, torch.Tensor]
        The converted state dict compatible with NunchakuChromaTransformer2DModel.
    """
    non_quantized_layers = non_quantized_layers or set()
    new_state_dict = {}
    for k, v in state_dict.items():
        if "single_transformer_blocks." in k:
            if ".qkv_proj." in k:
                new_k = k.replace(".qkv_proj.", ".attn.to_qkv.")
            elif ".norm_q." in k or ".norm_k." in k:
                new_k = k.replace(".norm_k.", ".attn.norm_k.")
                new_k = new_k.replace(".norm_q.", ".attn.norm_q.")
            elif ".mlp_fc1." in k:
                # mlp_fc1 stays as mlp_fc1
                new_k = k
            elif ".mlp_fc2." in k:
                # mlp_fc2 stays as mlp_fc2
                new_k = k
            elif ".out_proj." in k:
                # out_proj -> attn.to_out (attention output projection)
                new_k = k.replace(".out_proj.", ".attn.to_out.")
            else:
                new_k = k
            new_k = new_k.replace(".lora_down", ".proj_down")
            new_k = new_k.replace(".lora_up", ".proj_up")
            if ".smooth_orig" in k:
                new_k = new_k.replace(".smooth_orig", ".smooth_factor_orig")
            elif ".smooth" in k:
                new_k = new_k.replace(".smooth", ".smooth_factor")
            new_state_dict[new_k] = v
        elif "transformer_blocks." in k:
            if ".mlp_context_fc1" in k:
                new_k = k.replace(".mlp_context_fc1.", ".ff_context.net.0.proj.")
            elif ".mlp_context_fc2" in k:
                new_k = k.replace(".mlp_context_fc2.", ".ff_context.net.2.")
            elif ".mlp_fc1" in k:
                new_k = k.replace(".mlp_fc1.", ".ff.net.0.proj.")
            elif ".mlp_fc2" in k:
                new_k = k.replace(".mlp_fc2.", ".ff.net.2.")
            elif ".qkv_proj_context." in k:
                new_k = k.replace(".qkv_proj_context.", ".attn.add_qkv_proj.")
            elif ".qkv_proj." in k:
                new_k = k.replace(".qkv_proj.", ".attn.to_qkv.")
            elif ".norm_q." in k or ".norm_k." in k:
                new_k = k.replace(".norm_k.", ".attn.norm_k.")
                new_k = new_k.replace(".norm_q.", ".attn.norm_q.")
            elif ".norm_added_q." in k or ".norm_added_k." in k:
                new_k = k.replace(".norm_added_k.", ".attn.norm_added_k.")
                new_k = new_k.replace(".norm_added_q.", ".attn.norm_added_q.")
            elif ".out_proj." in k:
                new_k = k.replace(".out_proj.", ".attn.to_out.0.")
            elif ".out_proj_context." in k:
                new_k = k.replace(".out_proj_context.", ".attn.to_add_out.")
            else:
                new_k = k
            new_k = new_k.replace(".lora_down", ".proj_down")
            new_k = new_k.replace(".lora_up", ".proj_up")
            if ".smooth_orig" in k:
                new_k = new_k.replace(".smooth_orig", ".smooth_factor_orig")
            elif ".smooth" in k:
                new_k = new_k.replace(".smooth", ".smooth_factor")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    return new_state_dict
