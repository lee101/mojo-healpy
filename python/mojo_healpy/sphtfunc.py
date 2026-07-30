from __future__ import annotations

import math

import numpy as np

from ._lib import addr, lib
from .pixelfunc import npix2nside, nside2npix

UNSEEN = -1.6375e30


class Alm:
    @staticmethod
    def getsize(lmax, mmax=None):
        lmax = int(lmax)
        mmax = lmax if mmax is None or int(mmax) < 0 else int(mmax)
        if lmax < 0 or mmax < 0 or mmax > lmax:
            return -1
        return (mmax + 1) * (2 * lmax + 2 - mmax) // 2

    @staticmethod
    def getlmax(s, mmax=None):
        size = int(s)
        if mmax is None or int(mmax) < 0:
            candidate = int((math.sqrt(8 * size + 1) - 3) // 2)
            return candidate if Alm.getsize(candidate) == size else -1
        mmax = int(mmax)
        numerator = 2 * size
        denominator = mmax + 1
        if numerator % denominator:
            return -1
        candidate = (numerator // denominator + mmax - 2) // 2
        return candidate if Alm.getsize(candidate, mmax) == size else -1

    @staticmethod
    def getidx(lmax, l, m):
        l, m = np.broadcast_arrays(l, m)
        result = m * (2 * int(lmax) + 1 - m) // 2 + l
        return int(result) if result.ndim == 0 else result.astype(np.int64)

    @staticmethod
    def getlm(lmax, i=None):
        lmax = int(lmax)
        indices = np.arange(Alm.getsize(lmax)) if i is None else np.asarray(i)
        scalar = indices.ndim == 0
        flat = indices.reshape(-1)
        starts = np.array([Alm.getidx(lmax, m, m) for m in range(lmax + 1)])
        m = np.searchsorted(starts, flat, side="right") - 1
        l = flat - starts[m] + m
        l, m = l.reshape(indices.shape), m.reshape(indices.shape)
        if scalar:
            return int(l), int(m)
        return l.astype(np.int64), m.astype(np.int64)


def _limits(size, lmax, mmax):
    if lmax is None:
        lmax = Alm.getlmax(size, mmax)
        if lmax < 0:
            raise ValueError("Wrong alm size for the requested mmax")
    lmax = int(lmax)
    mmax = lmax if mmax is None else int(mmax)
    if lmax < 0 or mmax < 0 or mmax > lmax:
        raise ValueError("lmax and mmax must satisfy 0 <= mmax <= lmax")
    if Alm.getsize(lmax, mmax) != size:
        raise ValueError("Wrong alm size for the requested lmax and mmax")
    return lmax, mmax


def _alm_array(alm):
    return np.ascontiguousarray(alm, dtype=np.complex128)


def alm2map(
    alms,
    nside,
    lmax=None,
    mmax=None,
    pixwin=False,
    fwhm=0.0,
    sigma=None,
    pol=True,
    inplace=False,
    verbose=True,
    device="cpu",
):
    del inplace, verbose
    if pixwin:
        raise NotImplementedError("pixel-window convolution is not covered")
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    if sigma is not None or fwhm != 0.0:
        width = float(sigma) if sigma is not None else float(fwhm) / math.sqrt(8 * math.log(2))
    else:
        width = 0.0
    values = np.asarray(alms)
    if values.ndim not in (1, 2):
        raise ValueError("alms must be a one- or two-dimensional array")
    if np.ndim(nside) != 0:
        raise ValueError("nside must be scalar")
    single = values.ndim == 1
    if not single and pol:
        raise NotImplementedError("polarized spin-2 alm2map is not covered; pass pol=False")
    rows = values.reshape(1, -1) if single else values
    input_lmax, input_mmax = _limits(rows.shape[-1], lmax, mmax)
    if width:
        rows = np.asarray(
            [almxfl(row, gauss_beam(width * math.sqrt(8 * math.log(2)), input_lmax), input_mmax) for row in rows]
        )
    result = np.empty((rows.shape[0], nside2npix(nside)), dtype=np.float64)
    for source, target in zip(rows, result):
        source = _alm_array(source)
        function = (
            lib().mhp_alm2map_gpu if device == "gpu" else lib().mhp_alm2map
        )
        status = function(
            addr(source, np.complex128),
            addr(target, np.float64, writable=True),
            int(nside),
            input_lmax,
            input_mmax,
        )
        if device == "gpu" and status != 1:
            raise RuntimeError("GPU synthesis failed or exceeded its 2 GB safety limit")
    return result[0] if single else result


def map2alm(
    maps,
    lmax=None,
    mmax=None,
    iter=3,
    pol=True,
    use_weights=False,
    datapath=None,
    gal_cut=0,
    use_pixel_weights=False,
    verbose=True,
):
    del datapath, verbose
    if use_weights or use_pixel_weights:
        raise NotImplementedError("ring and pixel quadrature weights are not covered")
    if gal_cut:
        raise NotImplementedError("gal_cut is not covered")
    if isinstance(iter, (bool, np.bool_)) or int(iter) != iter or int(iter) < 0:
        raise ValueError("iter must be a non-negative integer")
    values = np.asarray(maps)
    if values.ndim not in (1, 2):
        raise ValueError("maps must be a one- or two-dimensional array")
    if values.dtype.kind == "c":
        raise TypeError("maps must be real-valued")
    single = values.ndim == 1
    if not single and pol:
        raise NotImplementedError("polarized spin-2 map2alm is not covered; pass pol=False")
    rows = values.reshape(1, -1) if single else values
    nside = npix2nside(rows.shape[-1])
    lmax = 3 * nside - 1 if lmax is None else int(lmax)
    mmax = lmax if mmax is None else int(mmax)
    result = np.empty((rows.shape[0], Alm.getsize(lmax, mmax)), dtype=np.complex128)
    for source, target in zip(rows, result):
        source = np.ascontiguousarray(source, dtype=np.float64)
        invalid = (source == UNSEEN) | ~np.isfinite(source)
        if np.any(invalid):
            source = source.copy()
            source[invalid] = 0.0
        residual = (
            np.empty(source.size, dtype=np.float64) if int(iter) > 0 else None
        )
        correction = (
            np.empty(target.size, dtype=np.complex128) if int(iter) > 0 else None
        )
        max_tasks = (128 * 1024 * 1024) // target.nbytes
        analyze_tasks = (
            min(32, max_tasks) if source.size >= 4096 and max_tasks >= 2 else 1
        )
        partials = (
            np.empty((analyze_tasks, target.size), dtype=np.complex128)
            if analyze_tasks > 1
            else None
        )
        lib().mhp_map2alm(
            addr(source, np.float64),
            addr(target, np.complex128, writable=True),
            0 if residual is None else addr(residual, np.float64, writable=True),
            0 if correction is None else addr(correction, np.complex128, writable=True),
            0 if partials is None else addr(partials, np.complex128, writable=True),
            nside,
            lmax,
            mmax,
            int(iter),
            analyze_tasks,
        )
    return result[0] if single else result


def _spectrum(first, second, lmax, mmax, lmax_out):
    first = _alm_array(first)
    second = _alm_array(second)
    result = np.empty(lmax_out + 1, dtype=np.float64)
    lib().mhp_alm2cl(
        addr(first, np.complex128),
        addr(second, np.complex128),
        addr(result, np.float64, writable=True),
        lmax,
        mmax,
        lmax_out,
    )
    return result


def alm2cl(alms1, alms2=None, lmax=None, mmax=None, lmax_out=None, nspec=None):
    first = np.asarray(alms1)
    if first.ndim not in (1, 2):
        raise ValueError("alm arrays must be one- or two-dimensional")
    first = first.reshape(1, -1) if first.ndim == 1 else first
    second_given = alms2 is not None
    second = first if alms2 is None else np.asarray(alms2)
    if second.ndim not in (1, 2):
        raise ValueError("alm arrays must be one- or two-dimensional")
    second = second.reshape(1, -1) if second.ndim == 1 else second
    lmax, mmax = _limits(first.shape[-1], lmax, mmax)
    if second.shape[-1] != first.shape[-1]:
        raise ValueError("alm arrays must have the same size")
    lmax_out = lmax if lmax_out is None else min(int(lmax_out), lmax)
    if lmax_out < 0:
        raise ValueError("lmax_out must be non-negative")
    if first.shape[0] == second.shape[0] and not second_given:
        pairs = [
            (i, i + offset)
            for offset in range(first.shape[0])
            for i in range(first.shape[0] - offset)
        ]
    elif first.shape[0] == second.shape[0]:
        pairs = [(i, j) for i in range(first.shape[0]) for j in range(second.shape[0])]
    else:
        raise ValueError("alm collections must contain the same number of arrays")
    if nspec is not None:
        pairs = pairs[: int(nspec)]
    spectra = np.asarray(
        [_spectrum(first[i], second[j], lmax, mmax, lmax_out) for i, j in pairs]
    )
    return spectra[0] if spectra.shape[0] == 1 else spectra


def almxfl(alm, fl, mmax=None, inplace=False):
    original = np.asarray(alm)
    if inplace and (original.dtype != np.complex128 or not original.flags.c_contiguous):
        raise ValueError("inplace=True requires a C-contiguous complex128 array")
    values = _alm_array(alm)
    lmax, mmax = _limits(values.size, None, mmax)
    raw_filters = np.asarray(fl)
    if raw_filters.dtype.kind == "c":
        raise TypeError("filter must be real-valued")
    filters = np.ascontiguousarray(raw_filters, dtype=np.float64)
    if filters.size < lmax + 1:
        raise ValueError("filter must contain at least lmax + 1 values")
    result = values if inplace and np.asarray(alm).dtype == np.complex128 else np.empty_like(values)
    lib().mhp_almxfl(
        addr(values, np.complex128),
        addr(filters, np.float64),
        addr(result, np.complex128, writable=True),
        lmax,
        mmax,
    )
    return result


def gauss_beam(fwhm, lmax=512, pol=False):
    if pol:
        raise NotImplementedError("polarized beam transfer functions are not covered")
    ell = np.arange(int(lmax) + 1, dtype=np.float64)
    sigma = float(fwhm) / math.sqrt(8.0 * math.log(2.0))
    return np.exp(-0.5 * ell * (ell + 1.0) * sigma**2)


def anafast(
    map1,
    map2=None,
    nspec=None,
    lmax=None,
    mmax=None,
    iter=3,
    alm=False,
    pol=True,
    use_weights=False,
    datapath=None,
    gal_cut=0,
    use_pixel_weights=False,
):
    first = map2alm(
        map1, lmax, mmax, iter, pol, use_weights, datapath, gal_cut, use_pixel_weights
    )
    second = None
    if map2 is not None:
        second = map2alm(
            map2, lmax, mmax, iter, pol, use_weights, datapath, gal_cut, use_pixel_weights
        )
    spectra = alm2cl(first, second, lmax, mmax, nspec=nspec)
    if not alm:
        return spectra
    return (spectra, first) if second is None else (spectra, first, second)
