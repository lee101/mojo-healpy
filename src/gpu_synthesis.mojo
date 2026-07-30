"""Optional GPU spherical-harmonic synthesis."""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import floor, sqrt

from harmonics import INV_SQRT_4PI, alm_index
from kernels import FPtr, PI, TWOPI, fp, pix_to_zphi


def sincos_f64(value: Float64) -> Tuple[Float64, Float64]:
    var x = value - floor((value + PI) / TWOPI) * TWOPI
    var cos_sign = 1.0
    if x > 0.5 * PI:
        x = PI - x
        cos_sign = -1.0
    elif x < -0.5 * PI:
        x = -PI - x
        cos_sign = -1.0
    var x2 = x * x
    var sine = x * (
        1.0
        + x2
        * (
            -1.0 / 6.0
            + x2
            * (
                1.0 / 120.0
                + x2
                * (
                    -1.0 / 5040.0
                    + x2
                    * (
                        1.0 / 362880.0
                        + x2
                        * (
                            -1.0 / 39916800.0
                            + x2
                            * (
                                1.0 / 6227020800.0
                                + x2
                                * (
                                    -1.0 / 1307674368000.0
                                    + x2
                                    * (
                                        1.0 / 355687428096000.0
                                        + x2 * (-1.0 / 121645100408832000.0)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    var cosine = cos_sign * (
        1.0
        + x2
        * (
            -1.0 / 2.0
            + x2
            * (
                1.0 / 24.0
                + x2
                * (
                    -1.0 / 720.0
                    + x2
                    * (
                        1.0 / 40320.0
                        + x2
                        * (
                            -1.0 / 3628800.0
                            + x2
                            * (
                                1.0 / 479001600.0
                                + x2
                                * (
                                    -1.0 / 87178291200.0
                                    + x2
                                    * (
                                        1.0 / 20922789888000.0
                                        + x2 * (-1.0 / 6402373705728000.0)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    return (sine, cosine)


def synthesize_gpu(
    alms: FPtr,
    maps: FPtr,
    nside: Int,
    lmax: Int,
    mmax: Int,
    npix: Int,
):
    var pixel = global_idx.x
    if pixel >= npix:
        return
    var z, phi = pix_to_zphi(nside, pixel, False)
    var sintheta = sqrt(max(0.0, 1.0 - z * z))
    var lambda_mm = INV_SQRT_4PI
    var value = 0.0
    for m in range(mmax + 1):
        if m > 0:
            lambda_mm *= (
                -sqrt(Float64(2 * m + 1) / Float64(2 * m)) * sintheta
            )
        var s, c = sincos_f64(Float64(m) * phi)
        var factor = 1.0 if m == 0 else 2.0
        var idx = alm_index(lmax, m, m)
        value += factor * lambda_mm * (
            alms[2 * idx] * c - alms[2 * idx + 1] * s
        )
        if m >= lmax:
            continue
        var previous2 = lambda_mm
        var previous = z * sqrt(Float64(2 * m + 3)) * lambda_mm
        idx = alm_index(lmax, m + 1, m)
        value += factor * previous * (
            alms[2 * idx] * c - alms[2 * idx + 1] * s
        )
        for ell in range(m + 2, lmax + 1):
            var e = Float64(ell)
            var mm = Float64(m * m)
            var alpha = sqrt((e * e - mm) / (4.0 * e * e - 1.0))
            var em1 = e - 1.0
            var beta = sqrt(
                (em1 * em1 - mm) / (4.0 * em1 * em1 - 1.0)
            )
            var current = (z * previous - beta * previous2) / alpha
            idx = alm_index(lmax, ell, m)
            value += factor * current * (
                alms[2 * idx] * c - alms[2 * idx + 1] * s
            )
            previous2 = previous
            previous = current
    maps[pixel] = value


def alm2map_gpu(
    alm_address: Int,
    map_address: Int,
    nside: Int,
    lmax: Int,
    mmax: Int,
) raises:
    var npix = 12 * nside * nside
    var nalm = (mmax + 1) * (2 * lmax + 2 - mmax) // 2
    var ctx = DeviceContext()
    var device_alms = ctx.enqueue_create_buffer[DType.float64](2 * nalm)
    var device_maps = ctx.enqueue_create_buffer[DType.float64](npix)
    ctx.enqueue_copy(device_alms, fp(alm_address))
    comptime BLOCK_SIZE = 256
    ctx.enqueue_function[synthesize_gpu](
        device_alms.unsafe_ptr(),
        device_maps.unsafe_ptr(),
        nside,
        lmax,
        mmax,
        npix,
        grid_dim=(npix + BLOCK_SIZE - 1) // BLOCK_SIZE,
        block_dim=BLOCK_SIZE,
    )
    ctx.enqueue_copy(fp(map_address), device_maps)
    ctx.synchronize()
