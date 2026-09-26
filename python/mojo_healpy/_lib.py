from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_HEALPY_LIB", os.path.join(ROOT, "dist", "libmojo-healpy.so"))

I = ctypes.c_int64

_SIGNATURES = {
    "mhp_pix2ang": ([I] * 8, None),
    "mhp_ang2pix": ([I] * 8, None),
    "mhp_pix2vec": ([I] * 9, None),
    "mhp_vec2pix": ([I] * 9, None),
    "mhp_order_convert": ([I] * 7, None),
    "mhp_alm2map": ([I] * 7, None),
    "mhp_alm2map_gpu": ([I] * 5, I),
    "mhp_map2alm_analyze": ([I] * 7, None),
    "mhp_map2alm_reduce": ([I] * 4, None),
    "mhp_map2alm_residual": ([I] * 3, None),
    "mhp_map2alm_update": ([I] * 3, None),
    "mhp_alm2cl": ([I] * 6, None),
    "mhp_almxfl": ([I] * 5, None),
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


ELEMENT_PARALLEL_MIN = 65_536
ELEMENTS_PER_WORKER = 16_384
MAX_ELEMENT_WORKERS = 8

TRANSFORM_PARALLEL_MIN = 4_096
PIXELS_PER_WORKER = 128
MAX_TRANSFORM_WORKERS = 32


def run_element_chunks(function, arguments, count, *, per_worker, max_workers):
    """Call a range-taking kernel once, or split it across worker threads.

    Every HEALPix geometry and harmonic kernel costs tens of cycles of
    transcendental work per 24-32 bytes of traffic, so these are
    compute-bound and scale with cores. ctypes drops the GIL for the foreign
    call, so this is genuine fan-out; chunks write disjoint element slices.
    """
    workers = 1
    if count >= ELEMENT_PARALLEL_MIN:
        workers = max(1, min(max_workers, -(-count // per_worker)))
    if workers == 1:
        function(*arguments, 0, count)
        return
    bounds = [i * count // workers for i in range(workers + 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(
            pool.map(
                lambda c: function(*arguments, bounds[c], bounds[c + 1]),
                range(workers),
            )
        )


def run_pixel_chunks(function, arguments, count):
    """Fan a pixel-range kernel out over worker threads."""
    workers = 1
    if count >= TRANSFORM_PARALLEL_MIN:
        workers = max(1, min(MAX_TRANSFORM_WORKERS, -(-count // PIXELS_PER_WORKER)))
    if workers == 1:
        function(*arguments, 0, count)
        return
    bounds = [i * count // workers for i in range(workers + 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(
            pool.map(
                lambda c: function(*arguments, bounds[c], bounds[c + 1]),
                range(workers),
            )
        )


def run_indexed_chunks(function, make_arguments, count, tasks):
    """Run a range-taking kernel once per task index, on worker threads.

    `make_arguments(task)` supplies the leading arguments for that task, so
    each worker can address its own accumulator slice. The element range for
    task `t` is `[count * t // tasks, count * (t + 1) // tasks)`.
    """
    if tasks <= 1:
        function(*make_arguments(0), 0, count)
        return
    bounds = [i * count // tasks for i in range(tasks + 1)]
    with ThreadPoolExecutor(max_workers=tasks) as pool:
        list(
            pool.map(
                lambda t: function(
                    *make_arguments(t), bounds[t], bounds[t + 1]
                ),
                range(tasks),
            )
        )


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
