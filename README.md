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

Chroma LoRAs are fully supported with automatic format conversion.

### In ComfyUI

1. Place your Chroma LoRA files in `ComfyUI/models/loras/`
2. Use the **"Nunchaku Chroma LoRA Loader"** node after the model loader
3. You can chain multiple LoRA nodes, or use **"Nunchaku Chroma LoRA Stack"** for multiple LoRAs in one node

### How LoRA Works

The LoRA implementation:

1. **Converts** ComfyUI/Kohya format (`lora_unet_double_blocks_*`) to diffusers format
2. **Merges** LoRA weights with the existing SVD low-rank branch by concatenation
3. **Preserves** the original quantization compensation while adding the LoRA effect

This approach is different from standard LoRA application - instead of adding LoRA as a separate computation, we concatenate the LoRA weights with the existing SVD decomposition weights. This maintains inference speed while supporting LoRA effects.

### Supported LoRA Formats

- **ComfyUI/Kohya format**: `lora_unet_double_blocks_*`, `lora_unet_single_blocks_*`
- **Diffusers format**: `transformer_blocks.*.lora_A.weight`, etc.

## File Structure

The installer adds the following files:

| Location | File | Purpose |
|----------|------|---------|
| nunchaku | `models/transformers/transformer_chroma.py` | Core Chroma transformer with LoRA support |
| nunchaku | `lora/chroma/diffusers_converter.py` | Converts ComfyUI LoRA format to diffusers |
| nunchaku | `lora/chroma/nunchaku_converter.py` | Merges LoRA with SVD branch |
| ComfyUI-nunchaku | `wrappers/chroma.py` | Latent format adapter (16ch ↔ 64ch) |
| ComfyUI-nunchaku | `wrappers/lora/` | LoRA converter copies for wrapper imports |
| ComfyUI-nunchaku | `nodes/models/chroma.py` | Model loader node |
| ComfyUI-nunchaku | `nodes/lora/chroma.py` | LoRA loader nodes |

## Folder Structure:

```
nunchaku-chroma/
├── install.py                    # Installer script
├── nunchaku_chroma/
│   ├── __init__.py
│   ├── transformer_chroma.py     # Main transformer with LoRA support
│   ├── patcher.py
│   ├── lora/
│   │   ├── __init__.py
│   │   ├── diffusers_converter.py   # Converts ComfyUI LoRA → diffusers format
│   │   └── nunchaku_converter.py    # Merges LoRA with SVD branch
│   └── comfyui/
│       ├── __init__.py
│       ├── wrapper.py            # ComfyUI wrapper with LoRA loading
│       ├── loader.py             # DiT loader node
│       └── lora.py               # LoRA loader nodes
```


