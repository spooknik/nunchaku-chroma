# Nunchaku Chroma

Adds [Chroma](https://github.com/lodestones/Chroma) model support to [Nunchaku](https://github.com/mit-han-lab/nunchaku) for fast 4-bit quantized inference.

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU

### Install from PyPI

```bash
pip install nunchaku nunchaku-chroma
```

### Install from source

```bash
pip install nunchaku
git clone https://github.com/spooknik/nunchaku-chroma.git
cd nunchaku-chroma
pip install -e .
```

## Usage

### Diffusers API

```python
from diffusers import ChromaPipeline
from nunchaku_chroma import NunchakuChromaTransformer2DModel
import torch

# Load quantized transformer
transformer = NunchakuChromaTransformer2DModel.from_pretrained(
    "path/to/quantized-chroma.safetensors",
    device="cuda",
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

### LoRA Support

```python
from nunchaku_chroma import NunchakuChromaTransformer2DModel

# Load quantized model
transformer = NunchakuChromaTransformer2DModel.from_pretrained(
    "path/to/quantized-chroma.safetensors",
    device="cuda",
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

### ComfyUI

For ComfyUI integration, use the [comfyui-nunchaku-chroma](https://github.com/spooknik/comfyui-nunchaku-chroma) custom node:

```bash
# Install dependencies
pip install nunchaku nunchaku-chroma

# Install the ComfyUI node
cd ComfyUI/custom_nodes
git clone https://github.com/spooknik/comfyui-nunchaku-chroma.git

# Restart ComfyUI
```

The following nodes will be available:
- **Nunchaku Chroma DiT Loader** - Load quantized Chroma models
- **Nunchaku Chroma LoRA Loader** - Apply a single LoRA
- **Nunchaku Chroma LoRA Stack** - Apply multiple LoRAs at once

## Supported LoRA Formats

- **ComfyUI/Kohya format**: `lora_unet_double_blocks_*`, `lora_unet_single_blocks_*`
- **Diffusers format**: `transformer_blocks.*.lora_A.weight`, etc.

## License

Apache-2.0
