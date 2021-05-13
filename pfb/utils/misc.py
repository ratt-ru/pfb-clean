import numpy as np
import dask.array as da
from ducc0.fft import r2c, c2r
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

def estimate_data_size(nant, nhr, nsec, nchan, ncorr, nbytes):
    '''
    Estimates size of data in GB where:

    nant    - number of antennas
    nhr     - lebgth of observation in hours
    nsec    - integration time in seconds
    nchan   - number of channels
    ncorr   - number of correlations
    nbytes  - bytes per item (eg. 8 for complex64)
    '''
    nbl = nant * (nant - 1) // 2
    ntime = nhr * 3600 // nsec
    return nbl * ntime * nchan * ncorr * nbytes / 1e9


def kron_matvec(A, b):
    D = len(A)
    N = b.size
    x = b

    for d in range(D):
        Gd = A[d].shape[0]
        NGd = N // Gd
        X = np.reshape(x, (Gd, NGd))
        Z = A[d].dot(X).T
        x = Z.ravel()
    return x


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


def Gaussian2D(xin, yin, GaussPar=(1., 1., 0.), normalise=True, nsigma=5):
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
    extent = (nsigma * Smaj)**2
    xflat = xin.squeeze()
    yflat = yin.squeeze()
    ind = np.argwhere(xflat**2 + yflat**2 <= extent).squeeze()
    idx = ind[:, 0]
    idy = ind[:, 1]
    x = np.array([xflat[idx, idy].ravel(), yflat[idx, idy].ravel()])
    R = np.einsum('nb,bc,cn->n', x.T, A, x)
    # need to adjust for the fact that GaussPar corresponds to FWHM
    fwhm_conv = 2 * np.sqrt(2 * np.log(2))
    tmp = np.exp(-fwhm_conv * R)
    gausskern = np.zeros(xflat.shape, dtype=np.float64)
    gausskern[idx, idy] = tmp

    if normalise:
        gausskern /= np.sum(gausskern)
    return np.ascontiguousarray(gausskern.reshape(sOut),
                                dtype=np.float64)


def give_edges(p, q, nx, ny, nx_psf, ny_psf):
    nx0 = nx_psf//2
    ny0 = ny_psf//2

    # image overlap edges
    # left edge for x coordinate
    dxl = p - nx0
    xl = np.maximum(dxl, 0)

    # right edge for x coordinate
    dxu = p + nx0
    xu = np.minimum(dxu, nx)
    # left edge for y coordinate
    dyl = q - ny0
    yl = np.maximum(dyl, 0)
    # right edge for y coordinate
    dyu = q + ny0
    yu = np.minimum(dyu, ny)

    # PSF overlap edges
    xlpsf = np.maximum(nx0 - p , 0)
    xupsf = np.minimum(nx0 + nx - p, nx_psf)
    ylpsf = np.maximum(ny0 - q, 0)
    yupsf = np.minimum(ny0 + ny - q, ny_psf)

    return slice(xl, xu), slice(yl, yu), \
        slice(xlpsf, xupsf), slice(ylpsf, yupsf)


def get_padding_info(nx, ny, pfrac):
    from ducc0.fft import good_size
    npad_x = int(pfrac * nx)
    nfft = good_size(nx + npad_x, True)
    npad_xl = (nfft - nx) // 2
    npad_xr = nfft - nx - npad_xl

    npad_y = int(pfrac * ny)
    nfft = good_size(ny + npad_y, True)
    npad_yl = (nfft - ny) // 2
    npad_yr = nfft - ny - npad_yl
    padding = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
    unpad_x = slice(npad_xl, -npad_xr)
    unpad_y = slice(npad_yl, -npad_yr)
    return padding, unpad_x, unpad_y


def convolve2gaussres(image, xx, yy, gaussparf, nthreads, gausspari=None,
                      pfrac=0.5, norm_kernel=False):
    """
    Convolves the image to a specified resolution.

    Parameters
    ----------
    Image       - (nband, nx, ny) array to convolve
    xx/yy       - coordinates on the grid in the same units as gaussparf.
    gaussparf   - tuple containing Gaussian parameters of desired resolution
                  (emaj, emin, pa).
    gausspari   - initial resolution . By default it is assumed that the image
                  is a clean component image with no associated resolution.
                  If beampari is specified, it must be a tuple containing
                  gausspars for each imaging band in the same format.
    nthreads    - number of threads to use for the FFT's.
    pfrac       - padding used for the FFT based convolution.
                  Will pad by pfrac/2 on both sides of image
    """
    nband, nx, ny = image.shape
    padding, unpad_x, unpad_y = get_padding_info(nx, ny, pfrac)
    ax = (1, 2)  # axes over which to perform fft
    lastsize = ny + np.sum(padding[-1])

    gausskern = Gaussian2D(xx, yy, gaussparf, normalise=norm_kernel)
    gausskern = np.pad(gausskern[None], padding, mode='constant')
    gausskernhat = r2c(iFs(gausskern, axes=ax), axes=ax, forward=True,
                       nthreads=nthreads, inorm=0)

    image = np.pad(image, padding, mode='constant')
    imhat = r2c(iFs(image, axes=ax), axes=ax, forward=True, nthreads=nthreads,
                inorm=0)

    # convolve to desired resolution
    if gausspari is None:
        imhat *= gausskernhat
    else:
        for i in range(nband):
            thiskern = Gaussian2D(xx, yy, gausspari[i], normalise=norm_kernel)
            thiskern = np.pad(thiskern[None], padding, mode='constant')
            thiskernhat = r2c(iFs(thiskern, axes=ax), axes=ax, forward=True,
                              nthreads=nthreads, inorm=0)

            convkernhat = np.where(np.abs(thiskernhat) > 0.0,
                                   gausskernhat / thiskernhat, 0.0)

            imhat[i] *= convkernhat[0]

    image = Fs(c2r(imhat, axes=ax, forward=False, lastsize=lastsize, inorm=2,
                   nthreads=nthreads), axes=ax)[:, unpad_x, unpad_y]

    return image, gausskern[:, unpad_x, unpad_y]

def chan_to_band_mapping(ms_name, nband=None):
    '''
    Construct dictionaries containing per MS and SPW channel to band mapping.
    Currently assumes we are only imaging field 0 of the first MS.

    Input:
    ms_name     - list of ms names
    nband       - number of imaging bands

    Output:
    freqs           - dict[MS][SPW] chunked dask arrays of the freq to band mapping
    freq_bin_idx    - dict[MS][SPW] chunked dask arrays of bin starting indices
    freq_bin_counts - dict[MS][SPW] chunked dask arrays of counts in each bin
    freq_out        - frequencies of average (LB - should a weighted sum rather be computed?)
    band_mapping    - dict[MS][SPW] identifying imaging bands going into degridder
    chan_chunks     - dict[MS][SPW] specifying dask chunking scheme over channel
    '''
    from daskms import xds_from_storage_ms as xds_from_ms
    from daskms import xds_from_storage_table as xds_from_table
    import dask
    import dask.array as da
    if not isinstance(ms_name, list):
        ms_name = [ms_name]

    # first pass through data to determine freq_mapping
    radec = None
    freqs = {}
    all_freqs = []
    spws = {}
    for ims in ms_name:
        xds = xds_from_ms(ims, chunks={"row": -1}, columns=('TIME',))

        # subtables
        ddids = xds_from_table(ims + "::DATA_DESCRIPTION")
        fields = xds_from_table(ims + "::FIELD")
        spws_table = xds_from_table(ims + "::SPECTRAL_WINDOW")
        pols = xds_from_table(ims + "::POLARIZATION")

        # subtable data
        ddids = dask.compute(ddids)[0]
        fields = dask.compute(fields)[0]
        spws_table = dask.compute(spws_table)[0]
        pols = dask.compute(pols)[0]

        freqs[ims] = {}
        spws[ims] = []
        for ds in xds:
            field = fields[ds.FIELD_ID]

            # check fields match
            if radec is None:
                radec = field.PHASE_DIR.data.squeeze()

            if not np.array_equal(radec, field.PHASE_DIR.data.squeeze()):
                continue

            spw = spws_table[ds.DATA_DESC_ID]
            tmp_freq = spw.CHAN_FREQ.data.squeeze()
            freqs[ims][ds.DATA_DESC_ID] = tmp_freq
            all_freqs.append(list([tmp_freq]))
            spws[ims].append(ds.DATA_DESC_ID)


    # freq mapping
    all_freqs = dask.compute(all_freqs)
    ufreqs = np.unique(all_freqs)  # sorted ascending
    nchan = ufreqs.size
    if nband is None:
        nband = nchan
    else:
       nband = nband

    # bin edges
    fmin = ufreqs[0]
    fmax = ufreqs[-1]
    fbins = np.linspace(fmin, fmax, nband + 1)
    freq_out = np.zeros(nband)
    for band in range(nband):
        indl = ufreqs >= fbins[band]
        # inclusive except for the last one
        indu = ufreqs < fbins[band + 1] + 1e-6
        freq_out[band] = np.mean(ufreqs[indl & indu])

    # chan <-> band mapping
    band_mapping = {}
    chan_chunks = {}
    freq_bin_idx = {}
    freq_bin_counts = {}
    for ims in freqs:
        freq_bin_idx[ims] = {}
        freq_bin_counts[ims] = {}
        band_mapping[ims] = {}
        chan_chunks[ims] = []
        for spw in freqs[ims]:
            freq = np.atleast_1d(dask.compute(freqs[ims][spw])[0])
            band_map = np.zeros(freq.size, dtype=np.int32)
            for band in range(nband):
                indl = freq >= fbins[band]
                indu = freq < fbins[band + 1] + 1e-6
                band_map = np.where(indl & indu, band, band_map)
            # to dask arrays
            bands, bin_counts = np.unique(band_map, return_counts=True)
            band_mapping[ims][spw] = tuple(bands)
            chan_chunks[ims].append({'chan': tuple(bin_counts)})
            freqs[ims][spw] = da.from_array(freq, chunks=tuple(bin_counts))
            bin_idx = np.append(np.array([0]), np.cumsum(bin_counts))[0:-1]
            freq_bin_idx[ims][spw] = da.from_array(bin_idx, chunks=1)
            freq_bin_counts[ims][spw] = da.from_array(bin_counts, chunks=1)

    return freqs, freq_bin_idx, freq_bin_counts, freq_out, band_mapping, chan_chunks

def stitch_images(dirties, nband, band_mapping):
    _, nx, ny = dirties[0].shape
    dirty = np.zeros((nband, nx, ny), dtype=dirties[0].dtype)
    d = 0
    for ims in band_mapping:
        for spw in band_mapping[ims]:
            for b, band in enumerate(band_mapping[ims][spw]):
                dirty[band] += dirties[d][b]
            d += 1
    return dirty

def restore_corrs(vis, ncorr):
    return da.blockwise(_restore_corrs, ('row', 'chan', 'corr'),
                        vis, ('row', 'chan'),
                        ncorr, None,
                        new_axes={"corr": ncorr},
                        dtype=vis.dtype)


def _restore_corrs(vis, ncorr):
    model_vis = np.zeros(vis.shape+(ncorr,), dtype=vis.dtype)
    model_vis[:, :, 0] = vis
    if model_vis.shape[-1] > 1:
        model_vis[:, :, -1] = vis
    return model_vis