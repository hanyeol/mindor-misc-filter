# Attribution

The MISCFilter kernels in this package are derived from the official
[ChengxuLiu/MISCFilter](https://github.com/ChengxuLiu/MISCFilter) implementation
(CVPR 2024) at upstream revision
[`aca59a88`](https://github.com/ChengxuLiu/MISCFilter/tree/aca59a8884cdf91f894dd0e7ca9f2527e4c9fd74).
The cleanup follows the inference-only port in
[porfeng/ComfyUI-PhotoDeblur](https://github.com/porfeng/ComfyUI-PhotoDeblur).

File provenance:

| This package                                      | Upstream path                       |
| ------------------------------------------------- | ----------------------------------- |
| `mindor_misc_filter/misc_filter/net.py`           | `models/MISCFilterNet.py`           |
| `mindor_misc_filter/misc_filter/kernel.py`        | `models/MISCKernel_cuda.py`         |
| `mindor_misc_filter/misc_filter/doconv.py`        | `models/doconv_pytorch.py`          |
| `mindor_misc_filter/misc_filter/layers.py`        | `models/layers.py`                  |

Modifications from upstream:

- Renamed every source file to a short, lowercase form
  (`MISCFilterNet.py` → `net.py`, `MISCKernel_cuda.py` → `kernel.py`,
  `doconv_pytorch.py` → `doconv.py`) and rewrote the corresponding intra-package
  imports.
- Replaced upstream's legacy pyinn-style `cupy_launch` helper with a
  `cupy.RawKernel` + `cupy.cuda.ExternalStream` bridge installed by
  `build_misc_filter_model` at first call. The inline CUDA C in `kernel.py`
  is kept verbatim; only the Python-side launcher changes.
- Added a single public builder (`build_misc_filter_model`) that constructs
  the `MISCKernelNet` architecture, loads the chosen variant's checkpoint,
  and returns an eval-mode wrapper.
- Flattened the directory layout into a single `mindor_misc_filter/` package
  so the kernels become importable as
  ``from mindor_misc_filter import build_misc_filter_model`` without
  `sys.path` manipulation.

Upstream license: MIT. See [LICENSE](LICENSE).
