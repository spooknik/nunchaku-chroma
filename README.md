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

### Install nunchaku-chroma

```bash
pip install nunchaku-chroma
```

Or install from source:

```bash
git clone https://github.com/your-username/nunchaku-chroma.git
cd nunchaku-chroma
pip install -e .
```

### Patch your nunchaku installation

After installing, run the patcher to add Chroma support:

```bash
python -m nunchaku_chroma.patcher
```

This will:
1. Copy `transformer_chroma.py` to your nunchaku installation
2. Update the `__init__.py` files to export `NunchakuChromaTransformer2DModel`

### Optional: Patch ComfyUI-nunchaku

If you use ComfyUI, you can also patch ComfyUI-nunchaku:

```bash
python -m nunchaku_chroma.patcher --comfyui /path/to/ComfyUI-nunchaku
```

### Verify installation

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

After patching ComfyUI-nunchaku:

1. Place your quantized Chroma model in `ComfyUI/models/diffusion_models/`
2. In ComfyUI, use the **"Nunchaku Chroma DiT Loader"** node
3. Connect to a standard FLUX-compatible sampling workflow

### LoRA Support

Chroma LoRAs are supported! In ComfyUI:

1. Place your Chroma LoRA files in `ComfyUI/models/loras/`
2. Use the **"Nunchaku Chroma LoRA Loader"** node after the model loader
3. You can chain multiple LoRA nodes, or use **"Nunchaku Chroma LoRA Stack"** for multiple LoRAs in one node

Note: Chroma LoRAs use the same format as FLUX LoRAs (`lora_unet_double_blocks_*` and `lora_unet_single_blocks_*`).

## Patcher Options

```
python -m nunchaku_chroma.patcher [options]

Options:
    --verify        Only verify if Chroma support is installed
    --comfyui PATH  Also patch ComfyUI-nunchaku at the specified path
    --dry-run       Show what would be done without making changes
```

## How it Works

The patcher adds the following files:

| Location | File | Purpose |
|----------|------|---------|
| nunchaku | `models/transformers/transformer_chroma.py` | Core Chroma transformer implementation with LoRA support |
| ComfyUI-nunchaku | `wrappers/chroma.py` | ComfyUI latent format adapter (packs 16ch → 64ch) |
| ComfyUI-nunchaku | `nodes/models/chroma.py` | ComfyUI loader node |
| ComfyUI-nunchaku | `nodes/lora/chroma.py` | ComfyUI LoRA loader nodes |

It also updates the `__init__.py` files to export `NunchakuChromaTransformer2DModel`.

## Troubleshooting

### "mat1 and mat2 shapes cannot be multiplied"

This usually means latent packing is not working correctly. Make sure you're using the ComfyUI wrapper which handles the conversion between 16-channel and 64-channel formats.

### "missing required positional argument 'y'"

The wrapper's forward method needs `y=None` and `guidance=None` as optional keyword arguments. This is handled in the included wrapper.

### Node doesn't appear in ComfyUI

1. Make sure nunchaku has Chroma support: `python -m nunchaku_chroma.patcher --verify`
2. Make sure ComfyUI-nunchaku was patched: Check for `wrappers/chroma.py` and `nodes/models/chroma.py`
3. Restart ComfyUI after patching

## License

MIT License
