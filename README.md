# Nunchaku Chroma

Repackaged [Nunchaku](https://github.com/mit-han-lab/nunchaku) with built-in [Chroma](https://github.com/lodestones/Chroma) model support for fast 4-bit quantized inference.

## Overview

Chroma is a FLUX-based diffusion model that requires specific handling:
- Uses **packed latents** (64 channels) instead of unpacked (16 channels)
- Uses `patch_size=1` instead of FLUX's `patch_size=2`
- Architecture: 19 double transformer blocks + 38 single transformer blocks

This package provides `nunchaku-chroma`, a drop-in replacement for `nunchaku` with Chroma support built-in. Since Chroma reuses FLUX's CUDA kernels, **no recompilation is needed**.

## Installation

### From GitHub Releases (Recommended)

Download the wheel for your platform, Python version, and PyTorch version:

```bash
# Example: Python 3.12, PyTorch 2.7, Linux
pip install https://github.com/spooknik/nunchaku-chroma/releases/download/v1.0.2/nunchaku_chroma-1.0.2+torch2.7-cp312-cp312-linux_x86_64.whl

# Example: Python 3.12, PyTorch 2.8, Windows
pip install https://github.com/spooknik/nunchaku-chroma/releases/download/v1.0.2/nunchaku_chroma-1.0.2+torch2.8-cp312-cp312-win_amd64.whl
```

### Supported Configurations

| Platform | Python | PyTorch |
|----------|--------|---------|
| Linux x86_64 | 3.10, 3.11, 3.12, 3.13 | 2.7, 2.8, 2.9, 2.10 |
| Windows AMD64 | 3.10, 3.11, 3.12, 3.13 | 2.7, 2.8, 2.9, 2.10 |

### Finding Your Configuration

```python
import sys
import torch

print(f"Python: cp{sys.version_info.major}{sys.version_info.minor}")
print(f"PyTorch: torch{'.'.join(torch.__version__.split('.')[:2])}")
print(f"Platform: {'linux_x86_64' if sys.platform == 'linux' else 'win_amd64'}")
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

1. Install `nunchaku-chroma` wheel (see above)
2. Install [ComfyUI-nunchaku](https://github.com/nunchaku-tech/ComfyUI-nunchaku)
3. Place your quantized Chroma model in `ComfyUI/models/diffusion_models/`
4. Use the **"Nunchaku Chroma DiT Loader"** node

## LoRA Support

Chroma LoRAs are fully supported with automatic format conversion and strength control.

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

### Supported LoRA Formats

- **ComfyUI/Kohya format**: `lora_unet_double_blocks_*`, `lora_unet_single_blocks_*`
- **Diffusers format**: `transformer_blocks.*.lora_A.weight`, etc.

## Building Wheels Locally

To build wheels yourself:

```bash
git clone https://github.com/spooknik/nunchaku-chroma.git
cd nunchaku-chroma

# Build single wheel
python build_wheels.py --single torch2.7 cp312 linux_x86_64

# Build all wheels
python build_wheels.py

# Built wheels are in dist/
```

## Troubleshooting

### "mat1 and mat2 shapes cannot be multiplied"

This usually means latent packing is not working correctly. Make sure you're using the ComfyUI wrapper which handles the conversion between 16-channel and 64-channel formats.

### "missing required positional argument 'y'"

The wrapper's forward method needs `y=None` and `guidance=None` as optional keyword arguments. This is handled in the included wrapper.

### "No module named 'nunchaku.lora.chroma'"

Make sure you installed `nunchaku-chroma`, not the original `nunchaku` package.

### LoRA has no effect

1. Check that the LoRA format is supported (ComfyUI/Kohya or diffusers format)
2. Verify the LoRA was trained for Chroma (not FLUX - they have different architectures)
3. Check ComfyUI logs for any conversion warnings

## License

MIT License
