import numpy as np
from africanus.model.spi import fit_spi_components
from numpy.testing._private.utils import assert_allclose
from pfb.utils import convolve2gaussres, Gaussian2D
import pytest

pmp = pytest.mark.parametrize

@pmp("nx", [128])
@pmp("ny", [80, 220])
@pmp("nband", [4, 8])
@pmp("alpha", [-0.5, 0.0, 0.5])
def test_convolve2gaussres(nx, ny, nband, alpha):
    freq = np.linspace(0.5e9, 1.5e9, nband)
    ref_freq = freq[0]

    Gausspari = ()
    es = np.linspace(15, 5, nband)
    for v in range(nband):
        Gausspari += ((es[v], es[v], 0.0),)

    x = np.arange(-nx/2, nx/2)
    y = np.arange(-ny/2, ny/2)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    restored = np.zeros((nband, nx, ny))
    for v in range(nband):
        restored[v] = Gaussian2D(xx, yy, Gausspari[v], normalise=False) * (freq[v]/ref_freq)**alpha

    conv_model, _ = convolve2gaussres(restored, xx, yy, Gausspari[0], 8, gausspari=Gausspari)

    I = np.argwhere(conv_model[-1] > 0.05).squeeze()
    Ix = I[:, 0]
    Iy = I[:, 1]

    comps = conv_model[:, Ix, Iy]
    weights = np.ones((nband))

    out = fit_spi_components(comps.T, weights, freq, ref_freq)

    assert_allclose(1+alpha, 1+out[0, :])  # offset for relative difference
    assert_allclose(out[2, :], restored[0, Ix, Iy])