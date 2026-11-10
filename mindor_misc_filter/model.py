from typing import Any, Dict, Mapping
import os
import torch
from torch import Tensor

class MiscFilterModel(torch.nn.Module):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()

        self.net = net

    def forward(self, image: Tensor) -> Tensor:
        predictions, _ = self.net(image)
        return predictions[0]

def build_misc_filter_model(checkpoint: Mapping[str, Any]) -> torch.nn.Module:
    """Build a MISCFilter inference model and load ``checkpoint``.

    ``checkpoint`` is a state dict for the ``RealBlur_J`` / ``RealBlur_R`` /
    ``GoPro`` variant, or the full checkpoint dict (with a ``state_dict``
    key). ``module.`` prefixes left over from DataParallel training are
    stripped automatically.

    Requires an NVIDIA GPU with CUDA 12 and a matching CuPy install — the
    deformable kernel is a CuPy ``RawKernel`` with no CPU or MPS fallback.
    ``ImportError`` is raised if CuPy is missing.

    Returns an eval-mode ``torch.nn.Module`` on the CPU with gradients
    disabled; move it to the target CUDA device after construction.
    """
    _configure_misc_kernel()

    from .misc_filter.net import MISCKernelNet

    net = MISCKernelNet(inference=False)
    net.load_state_dict(_strip_module_prefix(checkpoint, "state_dict"), strict=True)

    return MiscFilterModel(net).requires_grad_(False).eval()

def _configure_misc_kernel() -> None:
    # Bridge upstream MISCFilter's inline CUDA C through CuPy's RawKernel.
    # Upstream originally launched the kernel via the pyinn `cupy_launch`
    # helper; we swap in a `cupy.RawKernel` so no compiled extension is
    # needed. The `--std=c++11` option matches the kernel's language.
    import cupy
    import numpy as np

    from .misc_filter import kernel as kernels

    cuda_path = os.environ.get("CUDA_PATH")

    if cuda_path and hasattr(os, "add_dll_directory"):
        for suffix in ( "bin", "bin/x64" ):
            directory = os.path.join(cuda_path, suffix)

            if os.path.isdir(directory):
                os.add_dll_directory(directory)

    @cupy.memoize(for_each_device=True)
    def launch(name: str, source: str):
        kernel = cupy.RawKernel(source, name, options=( "--std=c++11", ))

        def call(*, grid, block, args, stream):
            packed = ( np.int32(args[0]), *( np.uint64(ptr) for ptr in args[1:] ) )
            kernel(grid, block, packed, stream=cupy.cuda.ExternalStream(stream.ptr))

        return call

    kernels.cupy_launch = launch

def _strip_module_prefix(checkpoint: Mapping[str, Any], key: str) -> Dict[str, Any]:
    # MISCFilter wraps weights under 'state_dict'; some are already flat.
    # DataParallel training also prepends 'module.' to every key — drop it so
    # strict=True holds.
    state_dict: Dict[str, Any] = {}

    for name, value in checkpoint.get(key, checkpoint).items():
        state_dict[name.removeprefix("module.")] = value

    return state_dict
