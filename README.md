# mojo-healpy

HEALPix pixelization and scalar spherical harmonics implemented in
[Mojo](https://www.modular.com/mojo), with a NumPy-facing Python API shaped like
[healpy](https://healpy.readthedocs.io/).

This is a focused port of the compute-heavy core, not a reimplementation of every
healpy feature. Use it as:

```python
import mojo_healpy as hp
```

The covered function names and argument order match healpy, so code using this subset
normally only needs that import change.

## Covered subset

Pixelization:

- `nside2npix`, `npix2nside`, `isnsideok`, `isnpixok`
- `nside2resol`, `nside2pixarea`
- `pix2ang`, `ang2pix`, `pix2vec`, `vec2pix` in RING and NESTED ordering
- `ring2nest`, `nest2ring`, `reorder`
- `ang2vec`, `vec2ang`

Scalar spherical harmonics:

- `Alm.getsize`, `Alm.getlmax`, `Alm.getidx`, `Alm.getlm`
- `alm2map` and `map2alm`, including rectangular `mmax` and iterative residual
  correction
- `alm2cl`, including cross-spectra and collections of scalar fields
- `almxfl`, `gauss_beam`, and `anafast`

The port does not cover polarized spin-2 T/E/B transforms, pixel or ring quadrature
weight files, pixel-window convolution, neighbour/disc queries, map resolution changes,
rotations, plotting, or FITS I/O. Unsupported weighting and polarization modes raise
`NotImplementedError` instead of silently producing scalar results. The transform code is
a direct scalar implementation; unlike healpy, it does not call libsharp.

Parity is tested against the real `healpy` package. The test suite exhaustively
checks all pixel centers through `nside=16`, checks random directions through `nside=64`,
and compares synthesis, analysis, spectra, filtering, indexing, ordering, broadcasting,
and validation behavior. These tests cover every function listed above; they do not imply
coverage of unlisted healpy APIs or optional modes.

## Install and run

The repository pins the Mojo nightly used to build it and installs healpy for parity
testing:

```bash
pixi install
pixi run build
pixi run test
```

The shared library is written to `dist/libmojo-healpy.so`.

## Usage

```python
import numpy as np
import mojo_healpy as hp

nside = 8
pixels = hp.ang2pix(
    nside,
    np.array([0.0, 45.0, 120.0]),
    np.array([0.0, 30.0, -20.0]),
    lonlat=True,
)
lon, lat = hp.pix2ang(nside, pixels, lonlat=True)

lmax = 6
alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
alm[hp.Alm.getidx(lmax, 2, 0)] = 1.0
sky = hp.alm2map(alm, nside, lmax=lmax)
recovered = hp.map2alm(sky, lmax=lmax, iter=3)
power = hp.alm2cl(recovered)

print(pixels)
print(sky.shape, power.shape)
```

Run the complete example directly with the environment:

```bash
pixi run python -c 'import mojo_healpy as hp; print(hp.pix2ang(1, 0))'
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux x86-64.
Times are the best of five runs, except the transforms, which use three runs.
`healpy / Mojo` above 1 means Mojo is faster.

| kernel | mojo-healpy | healpy | healpy / Mojo |
| --- | ---: | ---: | ---: |
| `ang2pix`, 2M directions | 32.688 ms | 116.797 ms | 3.57x |
| `pix2ang`, 2M pixels | 31.034 ms | 93.634 ms | 3.02x |
| `ring2nest`, 2M pixels | 33.498 ms | 85.068 ms | 2.54x |
| `almxfl`, `lmax=2048` | 26.258 ms | 36.324 ms | 1.38x |
| `alm2cl`, `lmax=2048` | 8.751 ms | 21.804 ms | 2.49x |
| `alm2map`, `nside=32`, `lmax=32` | 17.498 ms | 28.052 ms | 1.60x |
| `map2alm`, `nside=32`, `lmax=32`, `iter=0` | 19.133 ms | 24.023 ms | 1.26x |

These are the direct output of `pixi run bench` on this machine and include Python
validation and FFI overhead on both sides. Results vary with system load.

CPU remains the default. An optional accelerator implementation can be requested with
`device="gpu"`. GPU allocation or launch failure raises `RuntimeError`; it never silently
returns a CPU measurement or result.

## How it works

Python validates shapes, lengths, scalar bounds, dtypes, contiguity, alignment, and
writability before passing raw buffer addresses across a small C ABI using `ctypes`.
The ABI carries integer addresses, scalar sizes, and flags; Mojo reconstructs mutable
`UnsafePointer` values and writes into caller-owned arrays. Python keeps every NumPy
buffer alive for the duration of the synchronous call. Already-contiguous inputs of the
required dtype remain zero-copy across the boundary.

Pixel arrays are contiguous `int64`; maps and angles are contiguous `float64`.
Harmonic coefficients are NumPy `complex128`, which the kernels access as interleaved
real and imaginary `float64` values. Coefficients use healpy's packed m-major layout:
`m * (2*lmax + 1 - m) // 2 + l`.

RING conversion follows the HEALPix latitude-ring equations. NESTED conversion uses
face-local coordinates with bit-interleaved x/y indices. Harmonic synthesis evaluates
normalized associated Legendre functions by recurrence and reconstructs the real field
from stored non-negative m modes. Analysis applies equal-area HEALPix quadrature and,
when requested, repeats synthesis and residual analysis just as healpy's unweighted
iterative transform does. Large independent geometry and transform ranges use thresholded
CPU parallelism; packed-alm filtering, spectra, residuals, and reductions use SIMD with
scalar remainder loops.

MIT licensed.
