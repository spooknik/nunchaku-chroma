"""
Compose multiple LoRA weights into a single LoRA for Chroma models.

This script merges several LoRA safetensors files into one, applying individual strength values to each.

**Example Usage:**

.. code-block:: bash

    python -m nunchaku.lora.chroma.compose \\
        -i lora1.safetensors lora2.safetensors \\
        -s 0.8 1.0 \\
        -o composed_lora.safetensors

**Arguments:**

- ``-i``, ``--input-paths``: Input LoRA safetensors files (one or more).
- ``-s``, ``--strengths``: Strength value for each LoRA (must match number of inputs).
- ``-o``, ``--output-path``: Output path for the composed LoRA safetensors file.

This will merge ``lora1.safetensors`` (strength 0.8) and ``lora2.safetensors`` (strength 1.0) into ``composed_lora.safetensors``.

**Main Function**

:func:`compose_lora`
"""

import argparse
import os

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .diffusers_converter import to_diffusers


def compose_lora(
    loras: list[tuple[str | dict[str, torch.Tensor], float]], output_path: str | None = None
) -> dict[str, torch.Tensor]:
    """
    Compose multiple LoRA weights into a single LoRA representation.

    Parameters
    ----------
    loras : list of (str or dict[str, torch.Tensor], float)
        Each tuple contains:
            - Path to a LoRA safetensors file or a LoRA weights dictionary.
            - Strength/scale factor for that LoRA.
    output_path : str, optional
        Path to save the composed LoRA weights as a safetensors file. If None, does not save.

    Returns
    -------
    dict[str, torch.Tensor]
        The composed LoRA weights in diffusers format.

    Notes
    -----
    - Converts all input LoRAs to Diffusers format.
    - Handles QKV projection fusion for attention layers.
    - Applies strength scaling to LoRA weights.
    - Concatenates multiple LoRAs along appropriate dimensions.

    Examples
    --------
    >>> lora_paths = [("lora1.safetensors", 0.8), ("lora2.safetensors", 0.6)]
    >>> composed = compose_lora(lora_paths, "composed_lora.safetensors")
    >>> lora_dicts = [({"layer.weight": torch.randn(10, 20)}, 1.0)]
    >>> composed = compose_lora(lora_dicts)
    """
    if len(loras) == 0:
        return {}

    if len(loras) == 1:
        lora, strength = loras[0]
        lora_dict = to_diffusers(lora)
        # Apply strength to lora_A weights
        for k, v in lora_dict.items():
            if "lora_A" in k and v.ndim == 2:
                lora_dict[k] = v * strength
        if output_path is not None:
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            save_file(lora_dict, output_path)
        return lora_dict

    composed = {}
    for lora, strength in loras:
        lora_dict = to_diffusers(lora)

        for k, v in list(lora_dict.items()):
            if v.ndim == 1:
                # Handle 1D tensors (biases, norms, etc.)
                previous_tensor = composed.get(k, None)
                if previous_tensor is None:
                    # Skip scaling for norm layers
                    if "norm_q" in k or "norm_k" in k or "norm_added_q" in k or "norm_added_k" in k:
                        composed[k] = v
                    else:
                        composed[k] = v * strength
                else:
                    if not ("norm_q" in k or "norm_k" in k or "norm_added_q" in k or "norm_added_k" in k):
                        composed[k] = previous_tensor + v * strength
            elif v.ndim == 2:
                # Handle QKV fusion for single blocks
                if ".to_q." in k or ".add_q_proj." in k:
                    if "lora_B" in k:
                        continue  # Skip lora_B, handle with lora_A

                    # Get Q, K, V projections
                    q_a = v
                    k_a = lora_dict.get(k.replace(".to_q.", ".to_k.").replace(".add_q_proj.", ".add_k_proj."))
                    v_a = lora_dict.get(k.replace(".to_q.", ".to_v.").replace(".add_q_proj.", ".add_v_proj."))

                    q_b = lora_dict.get(k.replace("lora_A", "lora_B"))
                    k_b = lora_dict.get(
                        k.replace("lora_A", "lora_B").replace(".to_q.", ".to_k.").replace(".add_q_proj.", ".add_k_proj.")
                    )
                    v_b = lora_dict.get(
                        k.replace("lora_A", "lora_B").replace(".to_q.", ".to_v.").replace(".add_q_proj.", ".add_v_proj.")
                    )

                    # Skip if any component is missing
                    if any(x is None for x in [k_a, v_a, q_b, k_b, v_b]):
                        continue

                    # Pad to same rank if needed
                    max_rank = max(q_a.shape[0], k_a.shape[0], v_a.shape[0])
                    q_a = F.pad(q_a, (0, 0, 0, max_rank - q_a.shape[0]))
                    k_a = F.pad(k_a, (0, 0, 0, max_rank - k_a.shape[0]))
                    v_a = F.pad(v_a, (0, 0, 0, max_rank - v_a.shape[0]))
                    q_b = F.pad(q_b, (0, max_rank - q_b.shape[1]))
                    k_b = F.pad(k_b, (0, max_rank - k_b.shape[1]))
                    v_b = F.pad(v_b, (0, max_rank - v_b.shape[1]))

                    # Fuse QKV
                    if torch.isclose(q_a, k_a).all() and torch.isclose(q_a, v_a).all():
                        lora_a = q_a
                        lora_b = torch.cat((q_b, k_b, v_b), dim=0)
                    else:
                        # Stack the lora_A matrices
                        lora_a_group = (q_a, k_a, v_a)
                        new_shape_a = [sum([_.shape[0] for _ in lora_a_group]), q_a.shape[1]]
                        lora_a = torch.zeros(new_shape_a, dtype=q_a.dtype, device=q_a.device)
                        start_dim = 0
                        for tensor in lora_a_group:
                            lora_a[start_dim : start_dim + tensor.shape[0]] = tensor
                            start_dim += tensor.shape[0]

                        # Stack the lora_B matrices as block diagonal
                        lora_b_group = (q_b, k_b, v_b)
                        new_shape_b = [sum([_.shape[0] for _ in lora_b_group]), sum([_.shape[1] for _ in lora_b_group])]
                        lora_b = torch.zeros(new_shape_b, dtype=q_b.dtype, device=q_b.device)
                        start_dims = (0, 0)
                        for tensor in lora_b_group:
                            end_dims = (start_dims[0] + tensor.shape[0], start_dims[1] + tensor.shape[1])
                            lora_b[start_dims[0] : end_dims[0], start_dims[1] : end_dims[1]] = tensor
                            start_dims = end_dims

                    # Apply strength
                    lora_a = lora_a * strength

                    # Create fused keys
                    new_k_a = k.replace(".to_q.", ".to_qkv.").replace(".add_q_proj.", ".add_qkv_proj.")
                    new_k_b = new_k_a.replace("lora_A", "lora_B")

                    for kk, vv, dim in ((new_k_a, lora_a, 0), (new_k_b, lora_b, 1)):
                        previous_lora = composed.get(kk, None)
                        composed[kk] = vv if previous_lora is None else torch.cat([previous_lora, vv], dim=dim)

                elif ".to_k." in k or ".to_v." in k or ".add_k_proj." in k or ".add_v_proj." in k:
                    continue  # Handled above with to_q
                else:
                    # Regular LoRA weights
                    if "lora_A" in k:
                        v = v * strength

                    previous_lora = composed.get(k, None)
                    if previous_lora is None:
                        composed[k] = v
                    else:
                        # Concatenate along appropriate dimension
                        dim = 0 if "lora_A" in k else 1
                        composed[k] = torch.cat([previous_lora, v], dim=dim)

    if output_path is not None:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        save_file(composed, output_path)

    return composed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose multiple Chroma LoRAs into one")
    parser.add_argument(
        "-i", "--input-paths", type=str, nargs="*", required=True, help="paths to the lora safetensors files"
    )
    parser.add_argument("-s", "--strengths", type=float, nargs="*", required=True, help="strengths for each lora")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="path to the output safetensors file")
    args = parser.parse_args()
    assert len(args.input_paths) == len(args.strengths), "Number of input paths must match number of strengths"
    compose_lora(list(zip(args.input_paths, args.strengths)), args.output_path)
