from __future__ import annotations

import math

import numpy as np

from ._lib import (
    ELEMENTS_PER_WORKER,
    MAX_ELEMENT_WORKERS,
    addr,
    lib,
    run_element_chunks,
)


def _scalar(array: np.ndarray):
    return array[()] if array.ndim == 0 else array


def _valid_nside(nside, nest=False):
    values = np.asarray(nside)
    if values.dtype.kind not in "biuf":
        return np.zeros(values.shape, dtype=bool)
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0)
    in_range = (safe >= 1) & (safe <= 2**29)
    integers = np.where(in_range, safe, 0).astype(np.int64)
    valid = finite & in_range & (values == integers)
    if nest:
        valid &= (integers & (integers - 1)) == 0
    return valid


def isnsideok(nside, nest=False):
    result = _valid_nside(nside, nest=nest)
    return bool(result) if result.ndim == 0 else result


def nside2npix(nside):
    if not np.all(_valid_nside(nside)):
        raise ValueError(f"{nside} is not a valid nside parameter")
    result = 12 * np.asarray(nside, dtype=np.int64) ** 2
    return int(result) if result.ndim == 0 else result


def isnpixok(npix):
    values = np.asarray(npix)
    if values.dtype.kind not in "biuf":
        result = np.zeros(values.shape, dtype=bool)
        return bool(result) if result.ndim == 0 else result
    with np.errstate(invalid="ignore"):
        nside = np.sqrt(values / 12.0)
    finite = np.isfinite(values)
    representable = finite & (values >= 0) & (values <= np.iinfo(np.int64).max)
    safe_values = np.where(representable, values, 0)
    integers = safe_values.astype(np.int64)
    safe_nside = np.where(
        representable & (nside <= 2**29), np.floor(nside), 0
    ).astype(np.int64)
    result = (
        representable
        & (values == integers)
        & (values > 0)
        & (nside == np.floor(nside))
        & (safe_nside <= 2**29)
        & (12 * safe_nside**2 == values)
    )
    return bool(result) if result.ndim == 0 else result


def npix2nside(npix):
    if not np.all(isnpixok(npix)):
        raise ValueError(f"Wrong pixel number (it is not 12*nside**2)")
    result = np.sqrt(np.asarray(npix) / 12).astype(np.int64)
    return int(result) if result.ndim == 0 else result


def nside2pixarea(nside, degrees=False):
    area = 4.0 * np.pi / np.asarray(nside2npix(nside), dtype=np.float64)
    if degrees:
        area = np.degrees(1.0) ** 2 * area
    return float(area) if area.ndim == 0 else area


def nside2resol(nside, arcmin=False):
    result = np.sqrt(nside2pixarea(nside))
    if arcmin:
        result = np.degrees(result) * 60.0
    return float(result) if np.ndim(result) == 0 else result


def _dispatch_geometry(nside, inputs, output_count, symbol, nest):
    nside_array = np.asarray(nside)
    if nside_array.ndim == 0:
        arrays = np.broadcast_arrays(*[np.asarray(v) for v in inputs])
        if not _valid_nside(nside_array, nest=nest):
            raise ValueError("nside must be a positive power of two for NESTED ordering")
        shape = arrays[0].shape
        flat_inputs = [
            np.ascontiguousarray(
                value,
                dtype=np.int64
                if i == 0 and symbol in ("mhp_pix2ang", "mhp_pix2vec")
                else np.float64,
            ).ravel()
            for i, value in enumerate(arrays)
        ]
        output_dtype = (
            np.int64 if symbol in ("mhp_ang2pix", "mhp_vec2pix") else np.float64
        )
        outputs = [
            np.empty(flat_inputs[0].size, dtype=output_dtype)
            for _ in range(output_count)
        ]
        if flat_inputs[0].size == 0:
            return [value.reshape(shape) for value in outputs]
        arguments = [int(nside_array)]
        arguments.extend(addr(value, value.dtype) for value in flat_inputs)
        arguments.extend(addr(value, value.dtype, writable=True) for value in outputs)
        arguments.extend((flat_inputs[0].size, int(nest)))
        run_element_chunks(
            getattr(lib(), symbol),
            arguments,
            flat_inputs[0].size,
            per_worker=ELEMENTS_PER_WORKER,
            max_workers=MAX_ELEMENT_WORKERS,
        )
        return [value.reshape(shape) for value in outputs]

    arrays = np.broadcast_arrays(nside_array, *[np.asarray(v) for v in inputs])
    nsides = arrays[0].astype(np.int64)
    if not np.all(_valid_nside(nsides, nest=nest)):
        raise ValueError("nside must be a positive power of two for NESTED ordering")
    shape = nsides.shape
    flat_inputs = [
        np.ascontiguousarray(v, dtype=np.int64 if i == 0 and symbol in ("mhp_pix2ang", "mhp_pix2vec") else np.float64).ravel()
        for i, v in enumerate(arrays[1:])
    ]
    output_dtype = np.int64 if symbol in ("mhp_ang2pix", "mhp_vec2pix") else np.float64
    outputs = [np.empty(nsides.size, dtype=output_dtype) for _ in range(output_count)]
    flat_nsides = nsides.ravel()
    for value in np.unique(flat_nsides):
        mask = flat_nsides == value
        packed_inputs = [np.ascontiguousarray(v[mask]) for v in flat_inputs]
        packed_outputs = [np.empty(mask.sum(), dtype=output_dtype) for _ in outputs]
        arguments = [int(value)]
        arguments.extend(addr(v, v.dtype) for v in packed_inputs)
        arguments.extend(addr(v, v.dtype, writable=True) for v in packed_outputs)
        arguments.extend((int(mask.sum()), int(nest)))
        run_element_chunks(
            getattr(lib(), symbol),
            arguments,
            int(mask.sum()),
            per_worker=ELEMENTS_PER_WORKER,
            max_workers=MAX_ELEMENT_WORKERS,
        )
        for target, packed in zip(outputs, packed_outputs):
            target[mask] = packed
    return [value.reshape(shape) for value in outputs]


def pix2ang(nside, ipix, nest=False, lonlat=False):
    pixels = np.asarray(ipix)
    if pixels.dtype.kind not in "biuf" or np.any(~np.isfinite(pixels)):
        raise ValueError("ipix must contain finite integers")
    if np.any(pixels != np.floor(pixels)):
        raise ValueError("ipix must contain integers")
    if np.any(pixels < 0) or np.any(pixels >= np.asarray(nside2npix(nside))):
        raise ValueError("ipix must be in [0, 12*nside**2)")
    theta, phi = _dispatch_geometry(nside, [ipix], 2, "mhp_pix2ang", nest)
    if lonlat:
        return _scalar(np.degrees(phi)), _scalar(90.0 - np.degrees(theta))
    return _scalar(theta), _scalar(phi)


def ang2pix(
    nside, theta, phi, nest=False, lonlat=False, latauto=False, latbounce=True
):
    theta_array, phi_array = np.broadcast_arrays(theta, phi)
    if lonlat:
        lon = np.asarray(theta_array, dtype=np.float64)
        lat = np.asarray(phi_array, dtype=np.float64)
        if latauto:
            lat = (lat + 180.0) % 360.0 - 180.0
            crossed = np.abs(lat) > 90.0
            if not latbounce:
                lon = lon + np.where(crossed, 180.0, 0.0)
            lat = np.where(lat > 90.0, 180.0 - lat, lat)
            lat = np.where(lat < -90.0, -180.0 - lat, lat)
        if np.any(np.abs(lat) > 90.0):
            raise ValueError("THETA is out of range [0,pi]")
        theta_array = np.radians(90.0 - lat)
        phi_array = np.radians(lon)
    elif np.any(theta_array < 0.0) or np.any(theta_array > np.pi):
        raise ValueError("THETA is out of range [0,pi]")
    if np.any(~np.isfinite(theta_array)) or np.any(~np.isfinite(phi_array)):
        raise ValueError("angles must be finite")
    (pixels,) = _dispatch_geometry(
        nside, [theta_array, phi_array], 1, "mhp_ang2pix", nest
    )
    return _scalar(pixels)


def pix2vec(nside, ipix, nest=False):
    pixels = np.asarray(ipix)
    if (
        pixels.dtype.kind not in "biuf"
        or np.any(~np.isfinite(pixels))
        or np.any(pixels != np.floor(pixels))
    ):
        raise ValueError("ipix must contain finite integers")
    if np.any(pixels < 0) or np.any(pixels >= np.asarray(nside2npix(nside))):
        raise ValueError("ipix must be in [0, 12*nside**2)")
    x, y, z = _dispatch_geometry(nside, [ipix], 3, "mhp_pix2vec", nest)
    return _scalar(x), _scalar(y), _scalar(z)


def vec2pix(nside, x, y, z, nest=False):
    vectors = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    )
    if any(np.any(~np.isfinite(component)) for component in vectors):
        raise ValueError("vector components must be finite")
    if np.any(np.linalg.norm(np.stack(vectors, axis=-1), axis=-1) == 0):
        raise ValueError("zero-length vector")
    (pixels,) = _dispatch_geometry(nside, [x, y, z], 1, "mhp_vec2pix", nest)
    return _scalar(pixels)


def ring2nest(nside, ipix):
    return _order_convert(nside, ipix, True)


def nest2ring(nside, ipix):
    return _order_convert(nside, ipix, False)


def _order_convert(nside, ipix, nested_output):
    if not np.isscalar(nside):
        raise ValueError("nside must be scalar for ordering conversion")
    if not isnsideok(nside, nest=True):
        raise ValueError("nside must be a positive power of two")
    source = np.ascontiguousarray(ipix, dtype=np.int64)
    original = np.asarray(ipix)
    if (
        original.dtype.kind not in "biuf"
        or np.any(~np.isfinite(original))
        or np.any(original != source)
    ):
        raise ValueError("ipix must contain finite integers")
    if np.any(source < 0) or np.any(source >= nside2npix(nside)):
        raise ValueError("ipix must be in [0, 12*nside**2)")
    destination = np.empty_like(source)
    if source.size == 0:
        return _scalar(destination)
    run_element_chunks(
        lib().mhp_order_convert,
        [
            int(nside),
            addr(source, np.int64),
            addr(destination, np.int64, writable=True),
            source.size,
            int(nested_output),
        ],
        source.size,
        per_worker=ELEMENTS_PER_WORKER,
        max_workers=MAX_ELEMENT_WORKERS,
    )
    return _scalar(destination)


def reorder(map_in, inp=None, out=None, r2n=None, n2r=None):
    if r2n is not None:
        if r2n:
            inp, out = "RING", "NESTED"
    if n2r is not None:
        if n2r:
            inp, out = "NESTED", "RING"
    inp = "RING" if inp is None else inp.upper()
    out = "RING" if out is None else out.upper()
    values = np.asarray(map_in)
    nside = npix2nside(values.shape[-1])
    if inp == out:
        return values.copy()
    indices = np.arange(values.shape[-1], dtype=np.int64)
    if inp.startswith("R") and out.startswith("N"):
        return values[..., nest2ring(nside, indices)]
    if inp.startswith("N") and out.startswith("R"):
        return values[..., ring2nest(nside, indices)]
    raise ValueError("inp and out must be 'RING' or 'NESTED'")


def ang2vec(theta, phi, lonlat=False):
    theta, phi = np.broadcast_arrays(theta, phi)
    if lonlat:
        theta, phi = np.radians(90.0 - phi), np.radians(theta)
    sintheta = np.sin(theta)
    result = np.stack(
        (sintheta * np.cos(phi), sintheta * np.sin(phi), np.cos(theta)), axis=-1
    )
    return result


def vec2ang(vectors, lonlat=False):
    values = np.asarray(vectors, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("vectors must have a final dimension of length 3")
    if np.any(~np.isfinite(values)):
        raise ValueError("vector components must be finite")
    norm = np.linalg.norm(values, axis=-1)
    if np.any(norm == 0):
        raise ValueError("zero-length vector")
    theta = np.arccos(np.clip(values[..., 2] / norm, -1.0, 1.0))
    phi = np.mod(np.arctan2(values[..., 1], values[..., 0]), 2.0 * np.pi)
    if lonlat:
        return np.degrees(phi), 90.0 - np.degrees(theta)
    return theta, phi
