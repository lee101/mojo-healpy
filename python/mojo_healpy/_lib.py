from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_HEALPY_LIB", os.path.join(ROOT, "dist", "libmojo-healpy.so"))

I = ctypes.c_int64

_SIGNATURES = {
    "mhp_pix2ang": ([I, I, I, I, I, I], None),
    "mhp_ang2pix": ([I, I, I, I, I, I], None),
    "mhp_pix2vec": ([I, I, I, I, I, I, I], None),
    "mhp_vec2pix": ([I, I, I, I, I, I, I], None),
    "mhp_order_convert": ([I, I, I, I, I], None),
    "mhp_alm2map": ([I, I, I, I, I], None),
    "mhp_alm2map_gpu": ([I, I, I, I, I], I),
    "mhp_map2alm": ([I, I, I, I, I, I, I, I, I, I], None),
    "mhp_alm2cl": ([I, I, I, I, I, I], None),
    "mhp_almxfl": ([I, I, I, I, I], None),
}

_instance: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _instance
    if _instance is None:
        if not os.path.exists(LIB):
            raise RuntimeError(f"Mojo library not found at {LIB}; run `pixi run build`")
        _instance = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_instance, name)
            function.argtypes = argtypes
            function.restype = restype
    return _instance


def addr(array: np.ndarray, dtype: np.dtype, *, writable: bool = False) -> int:
    """Return an FFI address only for the exact buffer layout Mojo expects."""
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"FFI buffer must have dtype {np.dtype(dtype)}, got {array.dtype}")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("FFI buffers must be aligned and C-contiguous")
    if writable and not array.flags.writeable:
        raise ValueError("FFI output buffer must be writable")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise ValueError("FFI buffer pointer must be non-null")
    return address
