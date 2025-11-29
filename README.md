# Nunchaku Chroma

Adds [Chroma](https://github.com/lodestones/Chroma) model support to [Nunchaku](https://github.com/mit-han-lab/nunchaku) for fast 4-bit quantized inference.

## Overview

Chroma is a FLUX-based diffusion model that requires specific handling:
- Uses **packed latents** (64 channels) instead of unpacked (16 channels)
- Uses `patch_size=1` instead of FLUX's `patch_size=2`
- Architecture: 19 double transformer blocks + 38 single transformer blocks

Since Chroma reuses FLUX's CUDA kernels, **no recompilation is needed** - only Python files need to be added to your existing nunchaku installation.

## Installation

### Prerequisites

- Python 3.10+
- [Nunchaku](https://github.com/mit-han-lab/nunchaku) already installed
- A quantized Chroma model (`.safetensors` file)

### Quick Install (ComfyUI)

The simplest way to install is using the standalone installer:

```bash
git clone https://github.com/your-username/nunchaku-chroma.git
cd nunchaku-chroma
python install.py C:\ComfyUI   # Windows
python install.py ~/ComfyUI    # Linux/Mac
```

Or with auto-detection:

```bash
python install.py --auto-detect
```

Use `--dry-run` to see what would be changed without making modifications.

### Alternative: Install as Python Package

```bash
pip install nunchaku-chroma
```

Or install from source:

```bash
git clone https://github.com/your-username/nunchaku-chroma.git
cd nunchaku-chroma
pip install -e .
```

Then run the patcher:

```bash
python -m nunchaku_chroma.patcher
python -m nunchaku_chroma.patcher --comfyui /path/to/ComfyUI-nunchaku  # Optional
```

### Verify Installation

```bash
python -m nunchaku_chroma.patcher --verify
```

## Usage

### Diffusers API

```python
from diffusers import ChromaPipeline
from nunchaku import NunchakuChromaTransformer2DModel
import torch

# Load quantized transformer
transformer = NunchakuChromaTransformer2DModel.from_pretrained(
    "path/to/quantized-chroma.safetensors",
    torch_dtype=torch.bfloat16,
)

# Create pipeline
pipe = ChromaPipeline.from_pretrained(
    "lodestones/Chroma",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")

# Generate
image = pipe(
    "A beautiful sunset over mountains",
    num_inference_steps=20,
    guidance_scale=4.0,
).images[0]

image.save("output.png")
```

### ComfyUI

After installing:

1. Place your quantized Chroma model in `ComfyUI/models/diffusion_models/`
2. In ComfyUI, use the **"Nunchaku Chroma DiT Loader"** node
3. Connect to a standard FLUX-compatible sampling workflow

## LoRA Support

Chroma LoRAs are fully supported with automatic format conversion and strength control.

### In ComfyUI

1. Place your Chroma LoRA files in `ComfyUI/models/loras/`
2. Use the **"Nunchaku Chroma LoRA Loader"** node after the model loader
3. Adjust the **strength** slider (0.0 to 2.0) to control LoRA intensity
4. You can chain multiple LoRA nodes, or use **"Nunchaku Chroma LoRA Stack"** for multiple LoRAs in one node

### Diffusers API

```python
from nunchaku import NunchakuChromaTransformer2DModel

# Load quantized model
transformer = NunchakuChromaTransformer2DModel.from_pretrained(
    "path/to/quantized-chroma.safetensors",
    torch_dtype=torch.bfloat16,
)

# Apply single LoRA with strength
transformer.update_lora_params("path/to/lora.safetensors", strength=0.8)

# Adjust strength dynamically (without reloading)
transformer.set_lora_strength(0.5)

# Reset to base model
transformer.reset_lora()

# Apply multiple LoRAs with different strengths
transformer.update_lora_params_multi([
    ("lora1.safetensors", 0.8),
    ("lora2.safetensors", 0.5),
])
```

### Composing Multiple LoRAs

You can pre-compose multiple LoRAs into a single file:

```bash
python -m nunchaku.lora.chroma.compose \
    -i lora1.safetensors lora2.safetensors \
    -s 0.8 0.5 \
    -o composed_lora.safetensors
```

### How LoRA Works

The LoRA implementation:

1. **Converts** ComfyUI/Kohya format (`lora_unet_double_blocks_*`) to diffusers format
2. **Applies strength** by scaling the LoRA weights before merging
3. **Merges** LoRA weights with the existing SVD low-rank branch by concatenation
4. **Preserves** the original quantization compensation while adding the LoRA effect

This approach is different from standard LoRA application - instead of adding LoRA as a separate computation, we concatenate the LoRA weights with the existing SVD decomposition weights. This maintains inference speed while supporting LoRA effects.

### Supported LoRA Formats

- **ComfyUI/Kohya format**: `lora_unet_double_blocks_*`, `lora_unet_single_blocks_*`
- **Diffusers format**: `transformer_blocks.*.lora_A.weight`, etc.

## File Structure

The installer adds the following files:

| Location | File | Purpose |
|----------|------|---------|
| nunchaku | `models/transformers/transformer_chroma.py` | Core Chroma transformer with LoRA support |
| nunchaku | `lora/chroma/__init__.py` | Module exports |
| nunchaku | `lora/chroma/diffusers_converter.py` | Converts ComfyUI LoRA format to diffusers |
| nunchaku | `lora/chroma/nunchaku_converter.py` | Merges LoRA with SVD branch |
| nunchaku | `lora/chroma/compose.py` | Multi-LoRA composition utility |
| ComfyUI-nunchaku | `wrappers/chroma.py` | Latent format adapter (16ch ↔ 64ch) |
| ComfyUI-nunchaku | `wrappers/lora/` | LoRA converter copies for wrapper imports |
| ComfyUI-nunchaku | `nodes/models/chroma.py` | Model loader node |
| ComfyUI-nunchaku | `nodes/lora/chroma.py` | LoRA loader nodes |

## Troubleshooting

### "mat1 and mat2 shapes cannot be multiplied"

This usually means latent packing is not working correctly. Make sure you're using the ComfyUI wrapper which handles the conversion between 16-channel and 64-channel formats.

### "missing required positional argument 'y'"

The wrapper's forward method needs `y=None` and `guidance=None` as optional keyword arguments. This is handled in the included wrapper.

### "No module named 'nunchaku.lora.chroma'"

The LoRA converter files weren't installed. Re-run the installer:

```bash
python install.py C:\ComfyUI
```

### Node doesn't appear in ComfyUI

1. Make sure nunchaku has Chroma support: `python -m nunchaku_chroma.patcher --verify`
2. Make sure ComfyUI-nunchaku was patched: Check for `wrappers/chroma.py` and `nodes/models/chroma.py`
3. Restart ComfyUI after patching

### LoRA has no effect

1. Check that the LoRA format is supported (ComfyUI/Kohya or diffusers format)
2. Verify the LoRA was trained for Chroma (not FLUX - they have different architectures)
3. Check ComfyUI logs for any conversion warnings

## License

MIT License
