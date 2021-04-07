#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import argparse
from pfb.utils.fits import save_fits
import dask
import dask.array as da
import numpy as np
from africanus.util.numba import jit
from astropy.io import fits
import warnings
from africanus.model.spi.dask import fit_spi_components
from africanus.rime import parallactic_angles
from pfb.utils import load_fits, save_fits, data_from_header
from daskms import xds_from_ms, xds_from_table

@jit(nopython=True, nogil=True, cache=True)
def _unflagged_counts(flags, time_idx, out):
    for i in range(time_idx.size):
            ilow = time_idx[i]
            ihigh = time_idx[i+1]
            out[i] = np.sum(~flags[ilow:ihigh])
    return out

def extract_dde_info(args, freqs):
    """
    Computes paralactic angles, antenna scaling and pointing information
    required for beam interpolation. 
    """
    # get ms info required to compute paralactic angles and weighted sum
    nband = freqs.size
    if args.ms is not None:
        utimes = []
        unflag_counts = []
        ant_pos = None
        phase_dir = None
        for ms_name in args.ms:
            # get antenna positions
            ant = xds_from_table(ms_name + '::ANTENNA')[0].compute()
            if ant_pos is None:
                ant_pos = ant['POSITION'].data
            else: # check all are the same
                tmp = ant['POSITION']
                if not np.array_equal(ant_pos, tmp):
                    raise ValueError("Antenna positions not the same across measurement sets")
            
            # get phase center for field
            field = xds_from_table(ms_name + '::FIELD')[0].compute()
            if phase_dir is None:
                phase_dir = field['PHASE_DIR'][args.field].data.squeeze()
            else:
                tmp = field['PHASE_DIR'][args.field].data.squeeze()
                if not np.array_equal(phase_dir, tmp):
                    raise ValueError('Phase direction not the same across measurement sets')

            # get unique times and count flags
            xds = xds_from_ms(ms_name, columns=["TIME", "FLAG_ROW"], group_cols=["FIELD_ID"])[args.field]
            utime, time_idx = np.unique(xds.TIME.data.compute(), return_index=True)
            ntime = utime.size
            # extract subset of times
            if args.sparsify_time > 1:
                I = np.arange(0, ntime, args.sparsify_time)
                utime = utime[I]
                time_idx = time_idx[I]
                ntime = utime.size
            
            utimes.append(utime)
        
            flags = xds.FLAG_ROW.data.compute()
            unflag_count = _unflagged_counts(flags.astype(np.int32), time_idx, np.zeros(ntime, dtype=np.int32))
            unflag_counts.append(unflag_count)

        utimes = np.concatenate(utimes)
        unflag_counts = np.concatenate(unflag_counts)
        ntimes = utimes.size
        
        # compute paralactic angles
        parangles = parallactic_angles(utimes, ant_pos, phase_dir)

        # mean over antanna nant -> 1
        parangles = np.mean(parangles, axis=1, keepdims=True)
        nant = 1

        # beam_cube_dde requirements
        ant_scale = np.ones((nant, nband, 2), dtype=np.float64)
        point_errs = np.zeros((ntimes, nant, nband, 2), dtype=np.float64)

        return (parangles,
                da.from_array(ant_scale, chunks=ant_scale.shape),
                point_errs,
                unflag_counts,
                True)
    else:
        ntimes = 1
        nant = 1
        parangles = np.zeros((ntimes, nant,), dtype=np.float64)    
        ant_scale = np.ones((nant, nband, 2), dtype=np.float64)
        point_errs = np.zeros((ntimes, nant, nband, 2), dtype=np.float64)
        unflag_counts = np.array([1])
        
        return (parangles, ant_scale, point_errs, unflag_counts, False)


def make_power_beam(args, lm_source, freqs, use_dask):
    print("Loading fits beam patterns from %s" % args.beam_model)
    from glob import glob
    paths = glob(args.beam_model + '**_**.fits')
    beam_hdr = None
    if args.corr_type == 'linear':
        corr1 = 'XX'
        corr2 = 'YY'
    elif args.corr_type == 'circular':
        corr1 = 'LL'
        corr2 = 'RR'
    else:
        raise KeyError("Unknown corr_type supplied. Only 'linear' or 'circular' supported")

    for path in paths:
        if corr1.lower() in path[-10::]:
            if 're' in path[-7::]:
                corr1_re = load_fits(path)
                if beam_hdr is None:
                    beam_hdr = fits.getheader(path)
            elif 'im' in path[-7::]:
                corr1_im = load_fits(path)
            else:
                raise NotImplementedError("Only re/im patterns supported")
        elif corr2.lower() in path[-10::]:
            if 're' in path[-7::]:
                corr2_re = load_fits(path)
            elif 'im' in path[-7::]:
                corr2_im = load_fits(path)
            else:
                raise NotImplementedError("Only re/im patterns supported")
    
    # get power beam
    beam_amp = (corr1_re**2 + corr1_im**2 + corr2_re**2 + corr2_im**2)/2.0

    # get cube in correct shape for interpolation code
    beam_amp = np.ascontiguousarray(np.transpose(beam_amp, (1, 2, 0))
                                    [:, :, :, None, None])
    # get cube info
    if beam_hdr['CUNIT1'].lower() != "deg":
        raise ValueError("Beam image units must be in degrees")
    npix_l = beam_hdr['NAXIS1']
    refpix_l = beam_hdr['CRPIX1']
    delta_l = beam_hdr['CDELT1']
    l_min = (1 - refpix_l)*delta_l
    l_max = (1 + npix_l - refpix_l)*delta_l

    if beam_hdr['CUNIT2'].lower() != "deg":
        raise ValueError("Beam image units must be in degrees")
    npix_m = beam_hdr['NAXIS2']
    refpix_m = beam_hdr['CRPIX2']
    delta_m = beam_hdr['CDELT2']
    m_min = (1 - refpix_m)*delta_m
    m_max = (1 + npix_m - refpix_m)*delta_m

    if (l_min > lm_source[:, 0].min() or m_min > lm_source[:, 1].min() or
            l_max < lm_source[:, 0].max() or m_max < lm_source[:, 1].max()):
        raise ValueError("The supplied beam is not large enough")

    beam_extents = np.array([[l_min, l_max], [m_min, m_max]])

    # get frequencies
    if beam_hdr["CTYPE3"].lower() != 'freq':
        raise ValueError(
            "Cubes are assumed to be in format [nchan, nx, ny]")
    nchan = beam_hdr['NAXIS3']
    refpix = beam_hdr['CRPIX3']
    delta = beam_hdr['CDELT3']  # assumes units are Hz
    freq0 = beam_hdr['CRVAL3']
    bfreqs = freq0 + np.arange(1 - refpix, 1 + nchan - refpix) * delta
    if bfreqs[0] > freqs[0] or bfreqs[-1] < freqs[-1]:
        warnings.warn("The supplied beam does not have sufficient "
                        "bandwidth. Beam frequencies:")
        with np.printoptions(precision=2):
            print(bfreqs)

    if use_dask:
        return (da.from_array(beam_amp, chunks=beam_amp.shape),
                da.from_array(beam_extents, chunks=beam_extents.shape), 
                da.from_array(bfreqs, bfreqs.shape))
    else:
        return beam_amp, beam_extents, bfreqs

def interpolate_beam(ll, mm, freqs, args):
    """
    Interpolate beam to image coordinates and optionally compute average
    over time if MS is provoded
    """
    nband = freqs.size
    print("Interpolating beam")
    parangles, ant_scale, point_errs, unflag_counts, use_dask = extract_dde_info(args, freqs)

    lm_source = np.vstack((ll.ravel(), mm.ravel())).T
    beam_amp, beam_extents, bfreqs = make_power_beam(args, lm_source, freqs, use_dask)

    # interpolate beam
    if use_dask:
        from africanus.rime.dask import beam_cube_dde
        lm_source = da.from_array(lm_source, chunks=lm_source.shape)
        freqs = da.from_array(freqs, chunks=freqs.shape)
        # compute ncpu images at a time to avoid memory errors
        ntimes = parangles.shape[0]
        I = np.arange(0, ntimes, args.ncpu)
        nchunks = I.size
        I = np.append(I, ntimes)
        beam_image = np.zeros((ll.size, 1, nband), dtype=beam_amp.dtype)
        for i in range(nchunks):
            ilow = I[i]
            ihigh = I[i+1]
            part_parangles = da.from_array(parangles[ilow:ihigh], chunks=(1, 1))
            part_point_errs = da.from_array(point_errs[ilow:ihigh], chunks=(1, 1, freqs.size, 2))
            # interpolate and remove redundant axes
            part_beam_image = beam_cube_dde(beam_amp, beam_extents, bfreqs,
                                        lm_source, part_parangles, part_point_errs,
                                        ant_scale, freqs).compute()[:, :, 0, :, 0 , 0]
            # weighted sum over time
            beam_image += np.sum(part_beam_image * unflag_counts[None, ilow:ihigh, None], axis=1, keepdims=True)
        # normalise by sum of weights
        beam_image /= np.sum(unflag_counts)
        # remove time axis
        beam_image = beam_image[:, 0, :]
    else:
        from africanus.rime.fast_beam_cubes import beam_cube_dde
        beam_image = beam_cube_dde(beam_amp, beam_extents, bfreqs,
                                    lm_source, parangles, point_errs,
                                    ant_scale, freqs).squeeze()
    
    

    # swap source and freq axes and reshape to image shape
    beam_source = np.transpose(beam_image, axes=(1, 0))
    return beam_source.squeeze().reshape((freqs.size, *ll.shape))


def create_parser():
    p = argparse.ArgumentParser(description='Beam intrepolation tool.',
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('-image', "--image", type=str, required=True)
    p.add_argument('-ms', "--ms", nargs="+", type=str, 
                   help="Mesurement sets used to make the image. \n"
                   "Used to get paralactic angles if doing primary beam correction")
    p.add_argument('-f', "--field", type=int, default=0,
                   help="Field ID")
    p.add_argument('-o', '--output-filename', type=str,
                   help="Path to output directory. \n"
                        "Placed next to input model if outfile not provided.")
    p.add_argument('-bm', '--beam-model', default=None, type=str,
                   help="Fits beam model to use. \n"
                        "It is assumed that the pattern is path_to_beam/"
                        "name_corr_re/im.fits. \n"
                        "Provide only the path up to name "
                        "e.g. /home/user/beams/meerkat_lband. \n"
                        "Patterns mathing corr are determined "
                        "automatically. \n"
                        "Only real and imaginary beam models currently "
                        "supported.")
    p.add_argument('-st', "--sparsify-time", type=int, default=10,
                   help="Used to select a subset of time ")
    p.add_argument('-ncpu', '--ncpu', default=0, type=int,
                   help="Number of threads to use. \n"
                        "Default of zero means use all threads")
    p.add_argument('-ct', '--corr-type', type=str, default='linear',
                   help="Correlation typ i.e. linear or circular. ")
    return p

def main(args):
    # get coord info
    hdr = fits.getheader(args.image)
    l_coord, ref_l = data_from_header(hdr, axis=1)
    l_coord -= ref_l
    m_coord, ref_m = data_from_header(hdr, axis=2)
    m_coord -= ref_m
    if hdr["CTYPE4"].lower() == 'freq':
        freq_axis = 4
    elif hdr["CTYPE3"].lower() == 'freq':
        freq_axis = 3
    else:
        raise ValueError("Freq axis must be 3rd or 4th")
    freqs, ref_freq = data_from_header(hdr, axis=freq_axis)
    
    xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')
    
    # interpolate primary beam to fits header and optionally average over time
    beam_image = interpolate_beam(xx, yy, freqs, args)


    # save power beam
    save_fits(args.output_filename, beam_image, hdr)
    print("Wrote interpolated beam cube to %s \n" % args.output_filename)


    return

if __name__ == "__main__":
    args = create_parser().parse_args()

    if args.ncpu:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.ncpu))
    else:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()

    print(' \n ')
    GD = vars(args)
    print('Input Options:')
    for key in GD.keys():
        print(key, ' = ', GD[key])

    print(' \n ')

    print("Using %i threads" % args.ncpu)

    print(' \n ')

    main(args)