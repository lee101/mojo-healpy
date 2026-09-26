"""Single shared-library compilation unit."""

from harmonics import (
    alm2map_kernel,
    map2alm_analyze_kernel,
    map2alm_reduce_kernel,
    map2alm_residual_kernel,
    map2alm_update_kernel,
)
from kernels import (
    ang2pix_kernel,
    order_convert_kernel,
    pix2ang_kernel,
    pix2vec_kernel,
    vec2pix_kernel,
)
from filter import almxfl_kernel
from gpu_synthesis import alm2map_gpu
from power import alm2cl_kernel


@export("mhp_pix2ang")
def mhp_pix2ang(
    nside: Int,
    pixels: Int,
    theta: Int,
    phi: Int,
    n: Int,
    nest: Int,
    start: Int,
    stop: Int,
) abi("C"):
    pix2ang_kernel(nside, pixels, theta, phi, n, nest, start, stop)


@export("mhp_ang2pix")
def mhp_ang2pix(
    nside: Int,
    theta: Int,
    phi: Int,
    pixels: Int,
    n: Int,
    nest: Int,
    start: Int,
    stop: Int,
) abi("C"):
    ang2pix_kernel(nside, theta, phi, pixels, n, nest, start, stop)


@export("mhp_pix2vec")
def mhp_pix2vec(
    nside: Int,
    pixels: Int,
    x: Int,
    y: Int,
    z: Int,
    n: Int,
    nest: Int,
    start: Int,
    stop: Int,
) abi("C"):
    pix2vec_kernel(nside, pixels, x, y, z, n, nest, start, stop)


@export("mhp_vec2pix")
def mhp_vec2pix(
    nside: Int,
    x: Int,
    y: Int,
    z: Int,
    pixels: Int,
    n: Int,
    nest: Int,
    start: Int,
    stop: Int,
) abi("C"):
    vec2pix_kernel(nside, x, y, z, pixels, n, nest, start, stop)


@export("mhp_order_convert")
def mhp_order_convert(
    nside: Int,
    source: Int,
    destination: Int,
    n: Int,
    nested_output: Int,
    start: Int,
    stop: Int,
) abi("C"):
    order_convert_kernel(nside, source, destination, n, nested_output, start, stop)


@export("mhp_alm2cl")
def mhp_alm2cl(
    first: Int, second: Int, cls: Int, lmax: Int, mmax: Int, lmax_out: Int
) abi("C"):
    alm2cl_kernel(first, second, cls, lmax, mmax, lmax_out)


@export("mhp_alm2map")
def mhp_alm2map(
    alms: Int,
    maps: Int,
    nside: Int,
    lmax: Int,
    mmax: Int,
    start: Int,
    stop: Int,
) abi("C"):
    alm2map_kernel(alms, maps, nside, lmax, mmax, start, stop)


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


@export("mhp_map2alm_analyze")
def mhp_map2alm_analyze(
    maps: Int,
    partials: Int,
    nside: Int,
    lmax: Int,
    mmax: Int,
    start: Int,
    stop: Int,
) abi("C"):
    map2alm_analyze_kernel(maps, partials, nside, lmax, mmax, start, stop)


@export("mhp_map2alm_reduce")
def mhp_map2alm_reduce(
    partials: Int, alms: Int, ncoeff: Int, tasks: Int
) abi("C"):
    map2alm_reduce_kernel(partials, alms, ncoeff, tasks)


@export("mhp_map2alm_residual")
def mhp_map2alm_residual(maps: Int, residual: Int, n: Int) abi("C"):
    map2alm_residual_kernel(maps, residual, n)


@export("mhp_map2alm_update")
def mhp_map2alm_update(alms: Int, correction: Int, ncoeff: Int) abi("C"):
    map2alm_update_kernel(alms, correction, ncoeff)


@export("mhp_almxfl")
def mhp_almxfl(
    alms: Int, filters: Int, result: Int, lmax: Int, mmax: Int
) abi("C"):
    almxfl_kernel(alms, filters, result, lmax, mmax)
