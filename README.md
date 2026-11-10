# mindor-misc-filter

Inference-only [MISCFilter](https://github.com/ChengxuLiu/MISCFilter) (Motion-adaptive Integral Sampling with deformable Conv Filter, CVPR 2024) deblurring kernels, packaged as a minimal `pip`-installable module.

The upstream repository ships as a research codebase with training infrastructure and inline CUDA code launched through legacy helpers. This package extracts only the inference path and bridges the CUDA kernel through CuPy's `RawKernel` so downstream projects can run MISCFilter with just `torch` and `cupy-cudaXYx`.

## Install

```bash
pip install git+https://github.com/hanyeol/mindor-misc-filter.git@v0.1.0
```

You must also install a CuPy build matching your CUDA driver — this package deliberately leaves CuPy out of its declared dependencies because the right variant depends on the host CUDA version:

```bash
# For CUDA 12:
pip install "cupy-cuda12x>=13.6,<14"
```

## Usage

```python
import torch
from mindor_misc_filter import build_misc_filter_model

# Load the checkpoint file for the chosen variant.
# The official Google Drive archive bundles all three variants:
#   RealBlur_J.pth, RealBlur_R.pth, GoPro.pth
checkpoint = torch.load("RealBlur_J.pth", map_location="cpu")
model = build_misc_filter_model(checkpoint).to("cuda")

# image: (1, 3, H, W) float32 in [0, 1], H and W multiples of 8.
output = model(image)  # same shape, deblurred
```

``build_misc_filter_model`` wires MISCFilter's inline CUDA kernel into CuPy's RawKernel launcher the first time it is called. Missing CuPy raises ``ImportError``.

## Variants

| Variant      | Training data                  |
| ------------ | ------------------------------ |
| `RealBlur_J` | Real-world JPEG shots          |
| `RealBlur_R` | Real-world RAW checkpoint      |
| `GoPro`      | Synthetic GoPro motion blur    |

Unlike HI-Diff the architecture is identical across variants; only the checkpoint changes.

## Compatibility

- **PyTorch**: `>=2.0`.
- **GPU**: **NVIDIA CUDA only**. The deformable kernel is a CuPy `RawKernel` with no CPU or MPS fallback.
- **CuPy**: a `cupy-cudaXYx` matching the host CUDA major is required at runtime.
- **Input**: ``(B, 3, H, W)`` float32 tensors with H and W divisible by 8.
- **Precision**: fp32 only — the deformable sampling path produces NaNs in fp16.

See [NOTICE.md](NOTICE.md) for upstream attribution and the full list of modifications.
