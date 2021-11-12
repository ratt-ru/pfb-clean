
import numpy as np
import dask.array as da
from ducc0.wgridder import ms2dirty, dirty2ms
from ducc0.fft import r2c, c2r, c2c, good_size
from pfb.operators.psi import im2coef, coef2im
iFs = np.fft.ifftshift
Fs = np.fft.fftshift


def hessian_wgt_xds(alpha, xdss, waveopts, hessopts,
                    sigmainv, compute=True):
    '''
    Vis space Hessian reduction over dataset
    '''
    hesses = []
    wsum = 0
    for xds in xdss:
        wgt = xds.WEIGHT.data
        uvw = xds.UVW.data
        freq = xds.FREQ.data
        fbin_idx = xds.FBIN_IDX.data
        fbin_counts = xds.FBIN_COUNTS.data
        beam = xds.BEAM.data
        wsum += xds.WSUM.data.sum()

        convim = hessian_wgt(uvw, weight, alpha, beam, freq, fbin_idx,
                             fbin_counts, waveopts, hessopts)
        hesses.append(convim)
    hess = da.stack(hesses).sum(axis=0)/wsum
    if compute:
        return hess.compute() + alpha * sigmainv**2
    else:
        return hess + alpha * sigmainv**2

def _hessian_wgt_impl(uvw, weight, alpha, beam, freq, fbin_idx, fbin_counts,
                      cell=None,
                      wstack=None,
                      epsilon=None,
                      double_accum=None,
                      nthreads=None,
                      pmask=None,
                      padding=None,
                      bases=None,
                      iy=None,
                      sy=None,
                      ntot=None,
                      nlevels=None):
    """
    Tikhonov regularised Hessian of coeffs
    """
    nband, nx, ny = beam.shape
    nmax = alpha.shape[-1]
    alpha_rec = np.zeros_like(alpha)  # no chunking over row for now
    fbin_idx2 = fbin_idx - fbin_idx.min()  # adjust for freq chunking
    for b in range(nband):
        x = coef2im(alpha[b], pmask, bases, padding, iy, sy, nx, ny)

        flow = fbin_idx2[b]
        fhigh = fbin_idx2[b] + fbin_counts[b]
        mvis = dirty2ms(uvw=uvw,
                        freq=freq[flow:fhigh],
                        dirty=beam[b] * x,
                        wgt=weight[:, flow:fhigh],
                        pixsize_x=cell,
                        pixsize_y=cell,
                        epsilon=epsilon,
                        nthreads=nthreads,
                        do_wstacking=wstack)

        im = ms2dirty(uvw=uvw,
                      freq=freq[flow:fhigh],
                      ms=mvis,
                      wgt=weight[:, flow:fhigh],
                      npix_x=nx,
                      npix_y=ny,
                      pixsize_x=cell,
                      pixsize_y=cell,
                      epsilon=epsilon,
                      nthreads=nthreads,
                      do_wstacking=wstack,
                      double_precision_accumulation=double_accum)

        alpha_rec[b] = im2coef(beam[b]*im, pmask, bases, ntot, nmax, nlevels)

    return alpha_rec


def _hessian_wgt(uvw, weight, alpha, beam, freq, fbin_idx, fbin_counts,
                 waveopts, hessopts):
    return _hessian_wgt_impl(uvw[0], weight[0], alpha, beam[0][0], freq,
                             fbin_idx, fbin_counts, **waveopts, **hessopts)

def hessian_wgt(uvw, weight, alpha, beam, freq, fbin_idx, fbin_counts,
                waveopts, hessopts):
    return da.blockwise(_hessian_wgt, ('chan', 'basis', 'nmax')
,                       uvw, ("row", 'three'),
                        weight, ('row', 'chan'),
                        alpha, ('chan', 'basis', 'nmax'),
                        beam, ('chan', 'nx', 'ny'),
                        freq, ('chan'),
                        fbin_idx, ('chan'),
                        fbin_counts, ('chan'),
                        waveopts, None,
                        hessopts, None,
                        align_arrays=False,
                        adjust_chunks={'chan': beam.chunks[0]},
                        dtype=alpha.dtype)


def _hessian_reg_psf(alpha, beam, psfhat,
                     nthreads=None,
                     sigmainv=None,
                     pmask=None,
                     padding_psf=None,
                     unpad_x=None,
                     unpad_y=None,
                     lastsize=None,
                     padding=None,
                     bases=None,
                     iy=None,
                     sy=None,
                     ntot=None,
                     nmax=None,
                     nlevels=None,
                     nx=None,
                     ny=None):
    """
    Tikhonov regularised Hessian of coeffs
    """

    x = coef2im(alpha, pmask, bases, padding, iy, sy, nx, ny)

    xhat = iFs(np.pad(beam*x, padding_psf, mode='constant'), axes=(0, 1))
    xhat = r2c(xhat, axes=(0, 1), nthreads=nthreads,
                forward=True, inorm=0)
    xhat = c2r(xhat * psfhat, axes=(0, 1), forward=False,
                lastsize=lastsize, inorm=2, nthreads=nthreads)
    im = Fs(xhat, axes=(0, 1))[unpad_x, unpad_y]

    alpha_rec = im2coef(beam*im, pmask, bases, ntot, nmax, nlevels)

    return alpha_rec + alpha * sigmainv**2
