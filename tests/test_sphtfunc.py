import numpy as np
import pytest

import healpy as hp
import mojo_healpy as mh


def random_alm(lmax, seed=0):
    rng = np.random.default_rng(seed)
    alm = rng.normal(size=hp.Alm.getsize(lmax)) + 1j * rng.normal(
        size=hp.Alm.getsize(lmax)
    )
    _, m = hp.Alm.getlm(lmax)
    alm[m == 0] = alm[m == 0].real
    return alm


@pytest.mark.parametrize("lmax", [0, 1, 3, 8, 16])
def test_alm_index_helpers_match_healpy(lmax):
    assert mh.Alm.getsize(lmax) == hp.Alm.getsize(lmax)
    assert mh.Alm.getlmax(mh.Alm.getsize(lmax)) == lmax
    l, m = mh.Alm.getlm(lmax)
    expected_l, expected_m = hp.Alm.getlm(lmax)
    assert np.array_equal(l, expected_l)
    assert np.array_equal(m, expected_m)
    assert np.array_equal(mh.Alm.getidx(lmax, l, m), np.arange(l.size))


def test_alm_helpers_with_rectangular_mmax():
    size = mh.Alm.getsize(9, 4)
    assert size == hp.Alm.getsize(9, 4)
    assert mh.Alm.getlmax(size, 4) == hp.Alm.getlmax(size, 4) == 9
    assert mh.Alm.getlmax(size + 1, 4) == -1


@pytest.mark.parametrize(("nside", "lmax"), [(2, 3), (4, 5), (8, 8)])
def test_alm2map_matches_healpy(nside, lmax):
    alm = random_alm(lmax, nside)
    got = mh.alm2map(alm, nside, lmax=lmax)
    expected = hp.alm2map(alm, nside, lmax=lmax)
    np.testing.assert_allclose(got, expected, rtol=2e-14, atol=2e-14)


def test_alm2map_rectangular_mmax_matches_healpy():
    lmax, mmax, nside = 9, 4, 8
    rng = np.random.default_rng(5)
    alm = rng.normal(size=hp.Alm.getsize(lmax, mmax)) + 1j * rng.normal(
        size=hp.Alm.getsize(lmax, mmax)
    )
    np.testing.assert_allclose(
        mh.alm2map(alm, nside, lmax=lmax, mmax=mmax),
        hp.alm2map(alm, nside, lmax=lmax, mmax=mmax),
        rtol=3e-14,
        atol=3e-14,
    )


@pytest.mark.parametrize(("nside", "lmax"), [(2, 3), (4, 5), (8, 8)])
def test_map2alm_matches_healpy_uniform_weights(nside, lmax):
    sky = hp.alm2map(random_alm(lmax, nside + 10), nside, lmax=lmax)
    got = mh.map2alm(sky, lmax=lmax, iter=3)
    expected = hp.map2alm(sky, lmax=lmax, iter=3)
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=5e-14)


def test_map2alm_rectangular_mmax_matches_healpy():
    nside, lmax, mmax = 4, 5, 2
    sky = hp.alm2map(random_alm(lmax, 99), nside, lmax=lmax)
    np.testing.assert_allclose(
        mh.map2alm(sky, lmax=lmax, mmax=mmax, iter=1),
        hp.map2alm(sky, lmax=lmax, mmax=mmax, iter=1),
        rtol=2e-13,
        atol=5e-14,
    )


def test_map2alm_does_not_modify_unseen_input():
    sky = np.ones(hp.nside2npix(2))
    sky[3] = mh.UNSEEN
    before = sky.copy()
    mh.map2alm(sky, lmax=2, iter=0)
    assert np.array_equal(sky, before)


def test_alm2cl_auto_cross_and_lmax_out_match_healpy():
    first = random_alm(12, 21)
    second = random_alm(12, 22)
    np.testing.assert_allclose(mh.alm2cl(first), hp.alm2cl(first), atol=2e-15)
    np.testing.assert_allclose(
        mh.alm2cl(first, second, lmax_out=7),
        hp.alm2cl(first, second, lmax_out=7),
        atol=2e-15,
    )


def test_alm2cl_multiple_spectra_order_matches_healpy():
    alms = np.asarray([random_alm(6, seed) for seed in range(3)])
    np.testing.assert_allclose(mh.alm2cl(alms), hp.alm2cl(alms), atol=2e-15)
    np.testing.assert_allclose(
        mh.alm2cl(alms, nspec=4), hp.alm2cl(alms, nspec=4), atol=2e-15
    )


def test_almxfl_and_gauss_beam_match_healpy():
    alm = random_alm(10, 4)
    transfer = np.linspace(0.2, 1.2, 11)
    np.testing.assert_allclose(mh.almxfl(alm, transfer), hp.almxfl(alm, transfer))
    np.testing.assert_allclose(
        mh.gauss_beam(0.03, lmax=100), hp.gauss_beam(0.03, lmax=100)
    )


def test_simd_tails_for_filter_and_spectrum_match_healpy():
    alm = random_alm(13, 41)
    transfer = np.linspace(0.3, 1.1, 14)
    np.testing.assert_allclose(mh.almxfl(alm, transfer), hp.almxfl(alm, transfer))
    np.testing.assert_allclose(mh.alm2cl(alm), hp.alm2cl(alm), atol=2e-15)


def test_parallel_transforms_and_gpu_request_match_healpy():
    nside, lmax = 32, 8
    alm = random_alm(lmax, 77)
    expected_map = hp.alm2map(alm, nside, lmax=lmax)
    np.testing.assert_allclose(
        mh.alm2map(alm, nside, lmax=lmax),
        expected_map,
        rtol=3e-14,
        atol=3e-14,
    )
    np.testing.assert_allclose(
        mh.alm2map(alm, nside, lmax=lmax, device="gpu"),
        expected_map,
        rtol=3e-14,
        atol=3e-14,
    )
    np.testing.assert_allclose(
        mh.map2alm(expected_map, lmax=lmax, iter=1),
        hp.map2alm(expected_map, lmax=lmax, iter=1),
        rtol=2e-13,
        atol=5e-14,
    )


def test_invalid_device_raises():
    with pytest.raises(ValueError):
        mh.alm2map(random_alm(2), 2, lmax=2, device="tpu")


def test_transform_validation_prevents_unsafe_ffi_calls():
    with pytest.raises(ValueError):
        mh.map2alm(np.ones(12), lmax=1, iter=-1)
    with pytest.raises(ValueError):
        mh.map2alm(np.ones(12), lmax=-1)
    with pytest.raises(ValueError):
        mh.alm2cl(random_alm(2), lmax_out=-1)
    with pytest.raises(ValueError):
        mh.almxfl(np.ones(6, dtype=np.complex64), np.ones(3), inplace=True)
    with pytest.raises(ValueError):
        mh.alm2map(np.ones((2, 2, 6), dtype=np.complex128), 2, lmax=2)
    with pytest.raises(ValueError):
        mh.map2alm(np.ones((2, 2, 12)), lmax=1)
    with pytest.raises(TypeError):
        mh.map2alm(np.ones(12, dtype=np.complex128), lmax=1)
    with pytest.raises(TypeError):
        mh.almxfl(random_alm(2), np.ones(3, dtype=np.complex128))


def test_almxfl_inplace_requires_and_preserves_exact_buffer():
    alm = np.ascontiguousarray(random_alm(2))
    expected = hp.almxfl(alm.copy(), np.arange(3.0))
    returned = mh.almxfl(alm, np.arange(3.0), inplace=True)
    assert returned is alm
    np.testing.assert_allclose(alm, expected)


def test_multiple_scalar_maps_supported_with_pol_false():
    alms = np.asarray([random_alm(4, 30), random_alm(4, 31)])
    got = mh.alm2map(alms, 4, lmax=4, pol=False)
    expected = hp.alm2map(alms, 4, lmax=4, pol=False)
    np.testing.assert_allclose(got, expected, rtol=2e-14, atol=2e-14)


def test_anafast_matches_composed_healpy_transform():
    lmax, nside = 5, 4
    sky = hp.alm2map(random_alm(lmax, 40), nside, lmax=lmax)
    got = mh.anafast(sky, lmax=lmax, iter=2)
    expected = hp.anafast(sky, lmax=lmax, iter=2)
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=2e-13)


def test_unsupported_polarization_and_weights_are_explicit():
    alms = np.zeros((3, mh.Alm.getsize(2)), dtype=np.complex128)
    with pytest.raises(NotImplementedError):
        mh.alm2map(alms, 2, lmax=2, pol=True)
    with pytest.raises(NotImplementedError):
        mh.map2alm(np.ones(48), use_weights=True)
