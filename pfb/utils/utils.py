import numpy as np
from scipy.special import digamma, polygamma
from scipy.optimize import fmin_l_bfgs_b
from daskms import xds_from_ms, xds_from_table, xds_to_table, Dataset
from pyrap.tables import table
from numpy.testing import assert_array_equal
from time import time
from ducc0.fft import r2c, c2r
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

def prox_21(v, sigma, weights, axis=1):
    """
    Computes weighted version of

    prox_{sigma * || . ||_{21}}(v)

    Assumed that v has shape nbasis x nband x ntot where

    nbasis  - number of orthogonal bases
    nband   - number of imaging bands
    ntot    - total number of coefficients for each basis (must be equal)
    """
    l2_norm = np.linalg.norm(v, axis=axis)  # drops axis
    l2_soft = np.maximum(l2_norm - sigma * weights, 0.0)  # l2_norm is always positive
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

def to4d(data):
    if data.ndim == 4:
        return data
    elif data.ndim == 2:
        return data[None, None]
    elif data.ndim == 3:
        return data[None]
    elif data.ndim == 1:
        return data[None, None, None]
    else:
        raise ValueError("Only arrays with ndim <= 4 can be broadcast to 4D.")

def Gaussian2D(xin, yin, GaussPar=(1., 1., 0.), normalise=True):
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

    if normalise:
        gausskern /= np.sum(gausskern)
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


def get_padding_info(nx, ny, pfrac):
    from ducc0.fft import good_size
    npad_x = int(pfrac * nx)
    nfft = good_size(nx + npad_x, True)
    npad_xl = (nfft - nx)//2
    npad_xr = nfft - nx - npad_xl
    
    npad_y = int(pfrac * ny)
    nfft = good_size(ny + npad_y, True)
    npad_yl = (nfft - ny)//2
    npad_yr = nfft - ny - npad_yl
    padding = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
    unpad_x = slice(npad_xl, -npad_xr)
    unpad_y = slice(npad_yl, -npad_yr)
    return padding, unpad_x, unpad_y

def convolve2gaussres(image, xx, yy, gaussparf, nthreads, gausspari=None, pfrac=0.5, norm_kernel=False):
    """
    Convolves the image to a specified resolution.
    
    Parameters
    ----------
    Image - (nband, nx, ny) array to convolve
    xx/yy - coordinates on the grid in the same units as gaussparf.
    gaussparf - tuple containing Gaussian parameters of desired resolution (emaj, emin, pa).
    gausspari - initial resolution . By default it is assumed that the image is a clean component image with no associated resolution. 
                If beampari is specified, it must be a tuple containing gausspars for each imaging band in the same format.
    nthreads - number of threads to use for the FFT's.
    pfrac - padding used for the FFT based convolution. Will pad by pfrac/2 on both sides of image 
    """
    nband, nx, ny = image.shape
    padding, unpad_x, unpad_y = get_padding_info(nx, ny, pfrac)
    ax = (1, 2)  # axes over which to perform fft
    lastsize = ny + np.sum(padding[-1])

    gausskern = Gaussian2D(xx, yy, gaussparf, normalise=norm_kernel)
    gausskern = np.pad(gausskern[None], padding, mode='constant')
    gausskernhat = r2c(iFs(gausskern, axes=ax), axes=ax, forward=True, nthreads=nthreads, inorm=0)

    image = np.pad(image, padding, mode='constant')
    imhat = r2c(iFs(image, axes=ax), axes=ax, forward=True, nthreads=nthreads, inorm=0)

    # convolve to desired resolution
    if gausspari is None:
        imhat *= gausskernhat
    else:
        for i in range(nband):
            thiskern = Gaussian2D(xx, yy, gausspari[i], normalise=norm_kernel)
            thiskern = np.pad(thiskern[None], padding, mode='constant')
            thiskernhat = r2c(iFs(thiskern, axes=ax), axes=ax, forward=True, nthreads=nthreads, inorm=0)

            convkernhat = np.where(np.abs(thiskernhat)>0.0, gausskernhat/thiskernhat, 0.0)

            imhat[i] *= convkernhat[0]

    image = Fs(c2r(imhat, axes=ax, forward=False, lastsize=lastsize, inorm=2, nthreads=nthreads), axes=ax)[:, unpad_x, unpad_y]

    return image, gausskern[:, unpad_x, unpad_y]


def fitcleanbeam(psf, level=0.5, pixsize=1.0):
    """
    Find the Gaussian that approximates the main lobe of the PSF.
    """
    from skimage.morphology import label
    from scipy.optimize import curve_fit

    nband, nx, ny = psf.shape

    # coordinates
    x = np.arange(-nx/2, nx/2)
    y = np.arange(-ny/2, ny/2)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # model to fit
    def func(xy, emaj, emin, pa):
        Smin = np.minimum(emaj, emin)
        Smaj = np.maximum(emaj, emin)

        A = np.array([[1. / Smin ** 2, 0],
                    [0, 1. / Smaj ** 2]])

        c, s, t = np.cos, np.sin, np.deg2rad(-pa)
        R = np.array([[c(t), -s(t)],
                    [s(t), c(t)]])
        A = np.dot(np.dot(R.T, A), R)
        xy = np.array([x.ravel(), y.ravel()])
        R = np.einsum('nb,bc,cn->n', xy.T, A, xy)
        # GaussPar should corresponds to FWHM
        fwhm_conv = 2*np.sqrt(2*np.log(2))
        return np.exp(-fwhm_conv*R)

    Gausspars = ()
    for v in range(nband):
        # make sure psf is normalised
        psfv = psf[v]/psf[v].max()
        # find regions where psf is non-zero
        mask = np.where(psfv > level, 1.0, 0)

        # label all islands and find center
        islands = label(mask)
        ncenter = islands[nx//2, ny//2]

        # select psf main lobe
        psfv = psfv[islands == ncenter] 
        x = xx[islands == ncenter]
        y = yy[islands == ncenter]
        xy = np.vstack((x, y))
        xdiff = x.max() - x.min()
        ydiff = y.max() - y.min()
        emaj0 = np.maximum(xdiff, ydiff)
        emin0 = np.minimum(xdiff, ydiff)
        p, _ = curve_fit(func, xy, psfv, p0=(emaj0, emin0, 0.0))
        Gausspars += ((p[0]*pixsize, p[1]*pixsize, p[2]),)

    return Gausspars