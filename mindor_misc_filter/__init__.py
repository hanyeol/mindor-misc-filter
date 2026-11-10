"""Inference-only MISCFilter (Motion-adaptive Integral Sampling with
deformable Conv Filter) deblurring.

Public entry point:
  * ``build_misc_filter_model`` — construct a MISCFilter inference model from
    an already-loaded checkpoint.

The architecture classes (``MISCKernelNet``, the DO-Conv primitives, the CUDA
kernel launcher …) are intentionally not re-exported — downstream code should
route every use through ``build_misc_filter_model`` so the CuPy kernel setup
stays encapsulated.
"""

from .model import build_misc_filter_model

__all__ = [ "build_misc_filter_model" ]
