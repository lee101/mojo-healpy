import numpy as np
import pytest

import healpy as hp
import mojo_healpy as mh


@pytest.mark.parametrize("nside", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("nest", [False, True])
def test_pix2ang_exhaustive_matches_healpy(nside, nest):
    pixels = np.arange(hp.nside2npix(nside))
    got = mh.pix2ang(nside, pixels, nest=nest)
    expected = hp.pix2ang(nside, pixels, nest=nest)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("nside", [1, 2, 8, 64])
@pytest.mark.parametrize("nest", [False, True])
def test_ang2pix_random_matches_healpy(nside, nest):
    rng = np.random.default_rng(1234 + nside)
    theta = np.arccos(rng.uniform(-1.0, 1.0, 10_000))
    phi = rng.uniform(-20.0, 20.0, theta.size)
    assert np.array_equal(
        mh.ang2pix(nside, theta, phi, nest=nest),
        hp.ang2pix(nside, theta, phi, nest=nest),
    )


def test_scalar_and_broadcasting_match_healpy():
    assert mh.ang2pix(16, np.pi / 2, 0) == hp.ang2pix(16, np.pi / 2, 0)
    nside = np.array([1, 2, 4, 8, 16])
    got = mh.ang2pix(nside, np.pi / 2, 0)
    expected = hp.ang2pix(nside, np.pi / 2, 0)
    assert np.array_equal(got, expected)


def test_lonlat_roundtrip_matches_healpy():
    lon = np.array([-720.0, -45.0, 0.0, 123.0, 810.0])
    lat = np.array([-80.0, -10.0, 0.0, 45.0, 89.0])
    got = mh.ang2pix(32, lon, lat, lonlat=True)
    expected = hp.ang2pix(32, lon, lat, lonlat=True)
    assert np.array_equal(got, expected)
    got_angles = mh.pix2ang(32, got, lonlat=True)
    expected_angles = hp.pix2ang(32, expected, lonlat=True)
    np.testing.assert_allclose(got_angles, expected_angles, atol=1e-12)


@pytest.mark.parametrize("nest", [False, True])
def test_vector_conversions_match_healpy(nest):
    pixels = np.arange(hp.nside2npix(16))
    got = mh.pix2vec(16, pixels, nest=nest)
    expected = hp.pix2vec(16, pixels, nest=nest)
    np.testing.assert_allclose(got, expected, rtol=0, atol=2e-14)
    assert np.array_equal(
        mh.vec2pix(16, *got, nest=nest),
        hp.vec2pix(16, *expected, nest=nest),
    )


@pytest.mark.parametrize("nside", [1, 2, 4, 16])
def test_order_conversions_match_healpy(nside):
    pixels = np.arange(hp.nside2npix(nside))
    assert np.array_equal(mh.ring2nest(nside, pixels), hp.ring2nest(nside, pixels))
    assert np.array_equal(mh.nest2ring(nside, pixels), hp.nest2ring(nside, pixels))


@pytest.mark.parametrize("count", [65_535, 65_539])
def test_geometry_parallel_threshold_and_tail_match_healpy(count):
    rng = np.random.default_rng(count)
    theta = np.arccos(rng.uniform(-1.0, 1.0, count))
    phi = rng.uniform(-8.0, 8.0, count)
    assert np.array_equal(
        mh.ang2pix(512, theta, phi),
        hp.ang2pix(512, theta, phi),
    )


def test_parallel_order_conversion_tail_matches_healpy():
    rng = np.random.default_rng(2027)
    pixels = rng.integers(0, hp.nside2npix(512), 65_539)
    assert np.array_equal(
        mh.ring2nest(512, pixels),
        hp.ring2nest(512, pixels),
    )


def test_reorder_matches_healpy_and_preserves_dtype():
    ring = np.arange(hp.nside2npix(8), dtype=np.float32)
    nested = mh.reorder(ring, r2n=True)
    assert nested.dtype == ring.dtype
    assert np.array_equal(nested, hp.reorder(ring, r2n=True))
    stacked = np.stack((ring, -ring))
    assert np.array_equal(
        mh.reorder(stacked, n2r=True), hp.reorder(stacked, n2r=True)
    )


def test_pixelization_utilities_match_healpy():
    for nside in (1, 2, 3, 16, 1024):
        assert mh.isnsideok(nside) == hp.isnsideok(nside)
        assert mh.nside2npix(nside) == hp.nside2npix(nside)
        assert mh.nside2pixarea(nside) == pytest.approx(hp.nside2pixarea(nside))
        assert mh.nside2resol(nside) == pytest.approx(hp.nside2resol(nside))
        assert mh.npix2nside(mh.nside2npix(nside)) == nside
    assert not mh.isnsideok(3, nest=True)
    assert not mh.isnpixok(13)


def test_ang2vec_and_vec2ang_match_healpy():
    theta = np.array([0.1, 1.0, 2.8])
    phi = np.array([-1.0, 0.5, 8.0])
    vectors = mh.ang2vec(theta, phi)
    np.testing.assert_allclose(vectors, hp.ang2vec(theta, phi), atol=1e-15)
    got = mh.vec2ang(vectors)
    expected = hp.vec2ang(vectors)
    np.testing.assert_allclose(got, expected, atol=1e-15)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        mh.pix2ang(4, -1)
    with pytest.raises(ValueError):
        mh.ang2pix(4, -0.1, 0.0)
    with pytest.raises(ValueError):
        mh.vec2pix(4, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        mh.pix2ang(4, 1.5)
    with pytest.raises(ValueError):
        mh.ring2nest(4, [1.5])
    with pytest.raises(ValueError):
        mh.ang2pix(4, np.nan, 0.0)
    with pytest.raises(ValueError):
        mh.vec2ang([0.0, 0.0, 0.0])


def test_nside_int64_boundary_matches_healpy():
    assert mh.isnsideok(2**29)
    assert not mh.isnsideok(2**29 + 1)
    assert not mh.isnsideok(np.nan)
    assert not mh.isnsideok(np.inf)
    assert mh.nside2npix(2**29) == hp.nside2npix(2**29)


def test_empty_geometry_inputs_do_not_cross_ffi_with_null_buffers():
    empty = np.array([], dtype=np.float64)
    assert mh.ang2pix(4, empty, empty).shape == (0,)
    assert mh.pix2ang(4, np.array([], dtype=np.int64))[0].shape == (0,)
    assert mh.ring2nest(4, np.array([], dtype=np.int64)).shape == (0,)
