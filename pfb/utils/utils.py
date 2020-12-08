import numpy as np
from scipy.special import digamma, polygamma
from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import norm
import dask
import dask.array as da
from daskms import xds_from_ms, xds_from_table, xds_to_table, Dataset
from pyrap.tables import table
from numpy.testing import assert_array_equal
from time import time

def prox_21(v, sigma, weights, axis=1):
    """
    Computes weighted version of

    prox_{sigma * || . ||_{21}}(v)

    Assumed that v has shape nbasis x nband x ntot where

    nbasis  - number of orthogonal bases
    nband   - number of imaging bands
    ntot    - total number of coefficients for each basis (must be equal)
    """
    l2_norm = norm(v, axis=axis)  # drops axis
    l2_soft = np.maximum(l2_norm - sigma * weights, 0.0)  # l2_norm is always positive
    # l2_norm = np.abs(np.mean(v, axis=axis))  # drops freq axis
    # l2_soft = np.maximum(l2_norm - sigma * weights, 0.0)  # l2_norm is always positive
    mask = l2_norm != 0
    ratio = np.zeros(mask.shape, dtype=v.dtype)
    ratio[mask] = l2_soft[mask] / l2_norm[mask]
    return v * np.expand_dims(ratio, axis=axis)  # restores axis

def test_convolve(R, psf, args):
    x = np.random.randn(args.channels_out, args.nx, args.ny)

    res1 = R.convolve(x)
    res2 = psf.convolve(x)

    max_diff = np.abs(res1 - res2).max()/res1.max()

    print("Max frac diff is %5.5e and precision is %5.5e"%(max_diff, args.precision))

def test_adjoint(R):
    x = np.random.randn(R.nband, R.nx, R.ny)
    y = np.random.randn(R.nrow, R.nchan).astype(np.complex128)

    # y.H R x = x.H R.H y
    lhs = np.vdot(y, R.dot(x))
    rhs = np.vdot(x, R.hdot(y))
    print(" Natural = ", (lhs - rhs)/rhs)

    lhs = np.vdot(y, R.udot(x))
    rhs = np.vdot(x, R.uhdot(y))
    print(" Uniform = ", (lhs - rhs)/rhs)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')



def Gaussian2D(xin, yin, GaussPar=(1., 1., 0.)):
    S0, S1, PA = GaussPar
    Smaj = np.maximum(S0, S1)
    Smin = np.minimum(S0, S1)
    A = np.array([[1. / Smin ** 2, 0],
                  [0, 1. / Smaj ** 2]])

    c, s, t = np.cos, np.sin, np.deg2rad(-PA)
    R = np.array([[c(t), -s(t)],
                  [s(t), c(t)]])
    A = np.dot(np.dot(R.T, A), R)
    sOut = xin.shape
    # only compute the result out to 5 * emaj
    extent = (5 * Smaj)**2
    xflat = xin.squeeze()
    yflat = yin.squeeze()
    ind = np.argwhere(xflat**2 + yflat**2 <= extent).squeeze()
    idx = ind[:, 0]
    idy = ind[:, 1]
    x = np.array([xflat[idx, idy].ravel(), yflat[idx, idy].ravel()])
    R = np.einsum('nb,bc,cn->n', x.T, A, x)
    # need to adjust for the fact that GaussPar corresponds to FWHM
    fwhm_conv = 2*np.sqrt(2*np.log(2))
    tmp = np.exp(-fwhm_conv*R)
    gausskern = np.zeros(xflat.shape, dtype=np.float64)
    gausskern[idx, idy] = tmp
    return np.ascontiguousarray(gausskern.reshape(sOut),
                                dtype=np.float64)


def give_edges(p, q, nx, ny):
    # image overlap edges
    # left edge for x coordinate
    dxl = p - nx//2
    xl = np.maximum(dxl, 0)
    # right edge for x coordinate
    dxu = p + nx//2
    xu = np.minimum(dxu, nx)
    # left edge for y coordinate
    dyl = q - ny//2
    yl = np.maximum(dyl, 0)
    # right edge for y coordinate
    dyu = q + ny//2
    yu = np.minimum(dyu, ny)

    # PSF overlap edges
    
    xlpsf = np.maximum(nx//2 - p, 0)
    xupsf = np.minimum(3*nx//2 - p, nx)
    ylpsf = np.maximum(ny//2 - q, 0)
    yupsf = np.minimum(3*ny//2 - q, ny)


    return slice(xl, xu), slice(yl, yu), slice(xlpsf, xupsf), slice(ylpsf, yupsf)
