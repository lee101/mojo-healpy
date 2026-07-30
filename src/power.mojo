"""Power spectra from packed alm arrays."""

from std.sys.info import simd_width_of
from std.sys.intrinsics import strided_load

from kernels import fp


def alm2cl_kernel(
    first_address: Int,
    second_address: Int,
    cl_address: Int,
    lmax: Int,
    mmax: Int,
    lmax_out: Int,
):
    var first = fp(first_address)
    var second = fp(second_address)
    var cls = fp(cl_address)
    comptime W = simd_width_of[DType.float64]()
    var ell = 0
    var vector_end = ((lmax_out + 1) // W) * W
    while ell < vector_end:
        cls.store(ell, SIMD[DType.float64, W](0.0))
        ell += W
    while ell <= lmax_out:
        cls[ell] = 0.0
        ell += 1

    for m in range(min(mmax, lmax_out) + 1):
        var factor = 1.0 if m == 0 else 2.0
        ell = m
        vector_end = m + ((lmax_out + 1 - m) // W) * W
        while ell < vector_end:
            var idx = m * (2 * lmax + 1 - m) // 2 + ell
            var first_real = strided_load[simd_width=W](
                first + 2 * idx, 2
            )
            var first_imag = strided_load[simd_width=W](
                first + 2 * idx + 1, 2
            )
            var second_real = strided_load[simd_width=W](
                second + 2 * idx, 2
            )
            var second_imag = strided_load[simd_width=W](
                second + 2 * idx + 1, 2
            )
            var sums = (
                first_real * second_real + first_imag * second_imag
            ) * factor
            cls.store(ell, cls.load[width=W](ell) + sums)
            ell += W
        while ell <= lmax_out:
            var idx = m * (2 * lmax + 1 - m) // 2 + ell
            cls[ell] += factor * (
                first[2 * idx] * second[2 * idx]
                + first[2 * idx + 1] * second[2 * idx + 1]
            )
            ell += 1

    for ell_value in range(lmax_out + 1):
        cls[ell_value] /= Float64(2 * ell_value + 1)
