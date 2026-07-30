"""Single shared-library compilation unit."""

from kernels import (
    ang2pix_kernel,
    order_convert_kernel,
    pix2ang_kernel,
    pix2vec_kernel,
    vec2pix_kernel,
)
from filter import almxfl_kernel
from gpu_synthesis import alm2map_gpu
from harmonics import alm2map_kernel, map2alm_kernel
from power import alm2cl_kernel


@export("mhp_pix2ang")
def mhp_pix2ang(
    nside: Int, pixels: Int, theta: Int, phi: Int, n: Int, nest: Int
) abi("C"):
    pix2ang_kernel(nside, pixels, theta, phi, n, nest)


@export("mhp_ang2pix")
def mhp_ang2pix(
    nside: Int, theta: Int, phi: Int, pixels: Int, n: Int, nest: Int
) abi("C"):
    ang2pix_kernel(nside, theta, phi, pixels, n, nest)


@export("mhp_pix2vec")
def mhp_pix2vec(
    nside: Int, pixels: Int, x: Int, y: Int, z: Int, n: Int, nest: Int
) abi("C"):
    pix2vec_kernel(nside, pixels, x, y, z, n, nest)


@export("mhp_vec2pix")
def mhp_vec2pix(
    nside: Int, x: Int, y: Int, z: Int, pixels: Int, n: Int, nest: Int
) abi("C"):
    vec2pix_kernel(nside, x, y, z, pixels, n, nest)


@export("mhp_order_convert")
def mhp_order_convert(
    nside: Int, source: Int, destination: Int, n: Int, nested_output: Int
) abi("C"):
    order_convert_kernel(nside, source, destination, n, nested_output)


@export("mhp_alm2cl")
def mhp_alm2cl(
    first: Int, second: Int, cls: Int, lmax: Int, mmax: Int, lmax_out: Int
) abi("C"):
    alm2cl_kernel(first, second, cls, lmax, mmax, lmax_out)


@export("mhp_alm2map")
def mhp_alm2map(
    alms: Int, maps: Int, nside: Int, lmax: Int, mmax: Int
) abi("C"):
    alm2map_kernel(alms, maps, nside, lmax, mmax)


@export("mhp_alm2map_gpu")
def mhp_alm2map_gpu(
    alms: Int, maps: Int, nside: Int, lmax: Int, mmax: Int
) abi("C") -> Int:
    var npix = 12 * nside * nside
    var nalm = (mmax + 1) * (2 * lmax + 2 - mmax) // 2
    if 8 * (npix + 2 * nalm) >= 2_000_000_000:
        return 0
    try:
        alm2map_gpu(alms, maps, nside, lmax, mmax)
        return 1
    except:
        return 0


@export("mhp_map2alm")
def mhp_map2alm(
    maps: Int,
    alms: Int,
    residual: Int,
    correction: Int,
    partials: Int,
    nside: Int,
    lmax: Int,
    mmax: Int,
    iterations: Int,
    analyze_tasks: Int,
) abi("C"):
    map2alm_kernel(
        maps,
        alms,
        residual,
        correction,
        partials,
        nside,
        lmax,
        mmax,
        iterations,
        analyze_tasks,
    )


@export("mhp_almxfl")
def mhp_almxfl(
    alms: Int, filters: Int, result: Int, lmax: Int, mmax: Int
) abi("C"):
    almxfl_kernel(alms, filters, result, lmax, mmax)
