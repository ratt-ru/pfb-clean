import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from pfb.wavelets.wavelets import (dwt, idwt,
                                   promote_axis,
                                   promote_mode,
                                   discrete_wavelet)


def test_promote_mode():
    assert ["s"] == list(promote_mode("s", 1))
    assert ["s", "s", "s"] == list(promote_mode("s", 3))

    assert ["s"] == list(promote_mode(["s"], 1))
    assert ["s"] == list(promote_mode(("s",), 1))

    with pytest.raises(ValueError):
        assert ["s"] == list(promote_mode(["s"], 2))

    assert ["s", "t"] == list(promote_mode(["s", "t"], 2))
    assert ["s", "t"] == list(promote_mode(("s", "t"), 2))

    with pytest.raises(ValueError):
        assert ["s", "t"] == list(promote_mode(["s", "t"], 3))

    with pytest.raises(ValueError):
        assert ["s", "t"] == list(promote_mode(["s", "t"], 1))


def test_promote_axis():
    assert [0] == list(promote_axis(0, 1))
    assert [0] == list(promote_axis([0], 1))
    assert [0] == list(promote_axis((0,), 1))

    with pytest.raises(ValueError):
        assert [0, 1] == list(promote_axis((0, 1), 1))

    assert [0, 1] == list(promote_axis((0, 1), 2))
    assert [0, 1] == list(promote_axis([0, 1], 2))

    assert [0, 1] == list(promote_axis((0, 1), 3))


@pytest.mark.parametrize("wavelet", ["db1", "db5"])
def test_discrete_wavelet(wavelet):
    pfb_wave = discrete_wavelet(wavelet)

    pywt = pytest.importorskip("pywt")
    py_wave = pywt.Wavelet(wavelet)

    # assert py_wave.support_width == pfb_wave.support_width
    assert py_wave.orthogonal == pfb_wave.orthogonal
    assert py_wave.biorthogonal == pfb_wave.biorthogonal
    #assert py_wave.compact_support == pfb_wave.compact_support
    assert py_wave.family_name == pfb_wave.family_name
    assert py_wave.short_family_name == pfb_wave.short_name
    assert py_wave.vanishing_moments_phi == pfb_wave.vanishing_moments_phi
    assert py_wave.vanishing_moments_psi == pfb_wave.vanishing_moments_psi

    assert_array_almost_equal(py_wave.rec_lo, pfb_wave.rec_lo)
    assert_array_almost_equal(py_wave.dec_lo, pfb_wave.dec_lo)
    assert_array_almost_equal(py_wave.rec_hi, pfb_wave.rec_hi)
    assert_array_almost_equal(py_wave.rec_lo, pfb_wave.rec_lo)


def test_dwt():
    data = np.random.random((111, 126))
    dwt(data, "db1", "symmetric", 0)
