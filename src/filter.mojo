"""Packed-alm scalar filtering."""

from std.sys.info import simd_width_of

from kernels import fp


def almxfl_kernel(
    alm_address: Int,
    filter_address: Int,
    result_address: Int,
    lmax: Int,
    mmax: Int,
):
    var alms = fp(alm_address)
    var filters = fp(filter_address)
    var result = fp(result_address)
    comptime W = simd_width_of[DType.float64]()
    for m in range(mmax + 1):
        var ell = m
        var vector_end = m + ((lmax + 1 - m) // W) * W
        while ell < vector_end:
            var idx = m * (2 * lmax + 1 - m) // 2 + ell
            var values = alms.load[width=2 * W](2 * idx)
            var factors = filters.load[width=W](ell)
            result.store(2 * idx, values * factors.interleave(factors))
            ell += W
        while ell <= lmax:
            var idx = m * (2 * lmax + 1 - m) // 2 + ell
            result[2 * idx] = alms[2 * idx] * filters[ell]
            result[2 * idx + 1] = alms[2 * idx + 1] * filters[ell]
            ell += 1
