"""Scalar spherical-harmonic synthesis, analysis, and spectra."""

from std.algorithm import parallelize
from std.math import cos, sin, sqrt
from std.sys.info import simd_width_of

from kernels import FPtr, PI, fp, pix_to_zphi

comptime INV_SQRT_4PI = 0.282094791773878143474039725780386292
comptime TRANSFORM_PARALLEL_THRESHOLD = 4096


def alm_index(lmax: Int, ell: Int, m: Int) -> Int:
    return m * (2 * lmax + 1 - m) // 2 + ell


def synthesize_range(
    alms: FPtr,
    maps: FPtr,
    nside: Int,
    lmax: Int,
    mmax: Int,
    start: Int,
    end: Int,
):
    for pixel in range(start, end):
        var z, phi = pix_to_zphi(nside, pixel, False)
        var sintheta = sqrt(max(0.0, 1.0 - z * z))
        var lambda_mm = INV_SQRT_4PI
        var value = 0.0
        for m in range(mmax + 1):
            if m > 0:
                lambda_mm *= (
                    -sqrt(Float64(2 * m + 1) / Float64(2 * m)) * sintheta
                )
            var c = cos(Float64(m) * phi)
            var s = sin(Float64(m) * phi)
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
                var beta = sqrt((em1 * em1 - mm) / (4.0 * em1 * em1 - 1.0))
                var current = (z * previous - beta * previous2) / alpha
                idx = alm_index(lmax, ell, m)
                value += factor * current * (
                    alms[2 * idx] * c - alms[2 * idx + 1] * s
                )
                previous2 = previous
                previous = current
        maps[pixel] = value


def synthesize(alms: FPtr, maps: FPtr, nside: Int, lmax: Int, mmax: Int):
    var npix = 12 * nside * nside
    if npix < TRANSFORM_PARALLEL_THRESHOLD:
        synthesize_range(alms, maps, nside, lmax, mmax, 0, npix)
        return

    comptime TASKS = 32

    @parameter
    def worker(task: Int):
        var start = task * npix // TASKS
        var end = (task + 1) * npix // TASKS
        synthesize_range(alms, maps, nside, lmax, mmax, start, end)

    parallelize[worker](TASKS, TASKS)


def analyze_range(
    maps: FPtr,
    alms: FPtr,
    nside: Int,
    lmax: Int,
    mmax: Int,
    start: Int,
    end: Int,
):
    var npix = 12 * nside * nside
    var weight = 4.0 * PI / Float64(npix)
    for pixel in range(start, end):
        var z, phi = pix_to_zphi(nside, pixel, False)
        var sintheta = sqrt(max(0.0, 1.0 - z * z))
        var lambda_mm = INV_SQRT_4PI
        var sample = maps[pixel] * weight
        for m in range(mmax + 1):
            if m > 0:
                lambda_mm *= (
                    -sqrt(Float64(2 * m + 1) / Float64(2 * m)) * sintheta
                )
            var c = cos(Float64(m) * phi)
            var s = sin(Float64(m) * phi)
            var idx = alm_index(lmax, m, m)
            alms[2 * idx] += sample * lambda_mm * c
            alms[2 * idx + 1] -= sample * lambda_mm * s
            if m >= lmax:
                continue
            var previous2 = lambda_mm
            var previous = z * sqrt(Float64(2 * m + 3)) * lambda_mm
            idx = alm_index(lmax, m + 1, m)
            alms[2 * idx] += sample * previous * c
            alms[2 * idx + 1] -= sample * previous * s
            for ell in range(m + 2, lmax + 1):
                var e = Float64(ell)
                var mm = Float64(m * m)
                var alpha = sqrt((e * e - mm) / (4.0 * e * e - 1.0))
                var em1 = e - 1.0
                var beta = sqrt((em1 * em1 - mm) / (4.0 * em1 * em1 - 1.0))
                var current = (z * previous - beta * previous2) / alpha
                idx = alm_index(lmax, ell, m)
                alms[2 * idx] += sample * current * c
                alms[2 * idx + 1] -= sample * current * s
                previous2 = previous
                previous = current


def analyze_serial(maps: FPtr, alms: FPtr, nside: Int, lmax: Int, mmax: Int):
    var nalm = (mmax + 1) * (2 * lmax + 2 - mmax) // 2
    for i in range(2 * nalm):
        alms[i] = 0.0
    analyze_range(
        maps, alms, nside, lmax, mmax, 0, 12 * nside * nside
    )


def analyze_parallel(
    maps: FPtr,
    alms: FPtr,
    partials: FPtr,
    nside: Int,
    lmax: Int,
    mmax: Int,
    tasks: Int,
):
    var npix = 12 * nside * nside
    var ncoeff = 2 * (mmax + 1) * (2 * lmax + 2 - mmax) // 2

    @parameter
    def worker(task: Int):
        var local = partials + task * ncoeff
        for i in range(ncoeff):
            local[i] = 0.0
        var start = task * npix // tasks
        var end = (task + 1) * npix // tasks
        analyze_range(maps, local, nside, lmax, mmax, start, end)

    parallelize[worker](tasks, tasks)

    comptime W = simd_width_of[DType.float64]()
    var i = 0
    var vector_end = (ncoeff // W) * W
    while i < vector_end:
        var sums = SIMD[DType.float64, W](0.0)
        for task in range(tasks):
            sums += partials.load[width=W](task * ncoeff + i)
        alms.store(i, sums)
        i += W
    while i < ncoeff:
        var total = 0.0
        for task in range(tasks):
            total += partials[task * ncoeff + i]
        alms[i] = total
        i += 1


def alm2map_kernel(
    alm_address: Int, map_address: Int, nside: Int, lmax: Int, mmax: Int
):
    synthesize(fp(alm_address), fp(map_address), nside, lmax, mmax)


def map2alm_kernel(
    map_address: Int,
    alm_address: Int,
    residual_address: Int,
    correction_address: Int,
    partial_address: Int,
    nside: Int,
    lmax: Int,
    mmax: Int,
    iterations: Int,
    analyze_tasks: Int,
):
    var maps = fp(map_address)
    var alms = fp(alm_address)
    var npix = 12 * nside * nside
    var nalm = (mmax + 1) * (2 * lmax + 2 - mmax) // 2
    if analyze_tasks > 1:
        analyze_parallel(
            maps,
            alms,
            fp(partial_address),
            nside,
            lmax,
            mmax,
            analyze_tasks,
        )
    else:
        analyze_serial(maps, alms, nside, lmax, mmax)
    for _ in range(iterations):
        var residual = fp(residual_address)
        var correction = fp(correction_address)
        synthesize(alms, residual, nside, lmax, mmax)
        comptime W = simd_width_of[DType.float64]()
        var i = 0
        var vector_end = (npix // W) * W
        while i < vector_end:
            residual.store(
                i,
                maps.load[width=W](i) - residual.load[width=W](i),
            )
            i += W
        while i < npix:
            residual[i] = maps[i] - residual[i]
            i += 1
        if analyze_tasks > 1:
            analyze_parallel(
                residual,
                correction,
                fp(partial_address),
                nside,
                lmax,
                mmax,
                analyze_tasks,
            )
        else:
            analyze_serial(residual, correction, nside, lmax, mmax)
        i = 0
        vector_end = ((2 * nalm) // W) * W
        while i < vector_end:
            alms.store(
                i,
                alms.load[width=W](i) + correction.load[width=W](i),
            )
            i += W
        while i < 2 * nalm:
            alms[i] += correction[i]
            i += 1
