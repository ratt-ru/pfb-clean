import numpy as np
from numpy.testing import assert_array_equal
from scipy.interpolate import interp1d
from pyrap.tables import table
import argparse
from africanus.calibration.utils import chunkify_rows
from africanus.calibration.utils.dask import corrupt_vis
import dask
import dask.array as da
from daskms import xds_from_ms, xds_to_table
from dask.diagnostics import ProgressBar


# Gpath = '/home/landman/Data/MeerKAT/ESO137/caltables/e137-1557347448_sdp_l0-1gc1_primary.G1'
# Bpath = '/home/landman/Data/MeerKAT/ESO137/caltables/e137-1557347448_sdp_l0-1gc1_primary.B1'


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str, required=True,
                   help="Path to measurement set")
    p.add_argument("--writecol", type=str, required=True,
                   help="Column to write corrupted data to.")
    p.add_argument("--applycol", type=str, default=None,
                   help="Column to apply gains to."
                   "If None will simply apply to a column of ones")
    p.add_argument("--readcol", type=str, default='DATA',
                   help="Column to read in. Used to get shape of output.")
    p.add_argument("--Kpath", tye=str, default=None,
                   help="Path to K gain table")
    p.add_argument("--Gpath", tye=str, default=None,
                   help="Path to G gain table")
    p.add_argument("--Bpath", tye=str, default=None,
                   help="Path to B gain table")
    p.add_argument("--nthreads", type=int, default=0)
    p.add_argument("--utimes_per_chunk",  default=32, type=int,
                   help="Number of unique times in each chunk.")
    p.add_argument("--chan_chunks", type=int, default=-1,
                   help="Channel chunks")
    return p


def gain_func(t, nu, Ko, Bo, Go):
    K = np.exp(1.0j*Ko['delay'](t)[:, None]*nu[None, :])
    B = Bo['amp'](nu)[None, :]*np.exp(Bo['phase'](nu)[None, :]*1.0j)
    G = Go['amp'](t)[:, None]*np.exp(Go['phase'](t)[:, None]*1.0j)
    return K * G * B



def main(args):
    # interp K
    Ktab = table(args.Kpath)
    K = Ktab.getcol('FPARAM')
    t = Ktab.getcol('TIME')    
    ant1 = Ktab.getcol('ANTENNA1')
    Ktab.close()
    ndir = 1
    ncorr = 2
    Kdict = {}
    for p in range(ant1.max()+1):
        rows = np.argwhere(ant1==p).squeeze()
        Kdict.setdefault(p, {})
        for corr in range(ncorr):
            Kdict[p].setdefault(corr, {})
            delay = K[rows, 0, corr]
            Kdict[p][corr]['delay'] = interp1d(t[rows], delay, kind='linear', fill_value='extrapolate')


    # interp G    
    Gtab = table(args.Gpath)
    G = Gtab.getcol('CPARAM')
    t = Gtab.getcol('TIME')    
    ant1 = Gtab.getcol('ANTENNA1')
    Gtab.close()
    Gdict = {}
    for p in range(ant1.max()+1):
        rows = np.argwhere(ant1==p).squeeze()
        Gdict.setdefault(p, {})
        for corr in range(ncorr):
            Gdict[p].setdefault(corr, {})
            amp = np.abs(G[rows, 0, corr])
            Gdict[p][corr]['amp'] = interp1d(t[rows], amp, kind='linear', fill_value='extrapolate')
            phase = np.angle(G[rows, 0, corr])
            Gdict[p][corr]['phase'] = interp1d(t[rows], phase, kind='linear', fill_value='extrapolate')

    
    # interp B    
    Btab = table(args.Bpath)
    B = Btab.getcol('CPARAM')
    freq = table(args.Bpath+'::SPECTRAL_WINDOW').getcol('CHAN_FREQ').squeeze()
    ant1 = Btab.getcol('ANTENNA1')
    Btab.close()
    Bdict = {}
    for p in range(ant1.max()+1):
        rows = np.argwhere(ant1==p).squeeze()
        Bdict.setdefault(p, {})
        for corr in range(ncorr):
            Bdict[p].setdefault(corr, {})
            amp = np.abs(B[rows, :, corr])
            Bdict[p][corr]['amp'] = interp1d(freq, amp, kind='linear', fill_value='extrapolate')
            phase = np.angle(B[p, :, corr])
            Bdict[p][corr]['phase'] = interp1d(freq, phase, kind='linear', fill_value='extrapolate')
    
    
    # construct single phenomenalogical gain term for MS
    ms = table(args.ms)
    time = ms.getcol('TIME')
    utime = np.unique(t)
    ntime = utime.size
    ant1max = ms.getcol('ANTENNA1').max()
    ant2max = ms.getcol('ANTENNA2').max()
    nant = np.maximum(ant1max, ant2max)
    ms.close()  
    freq = table(args.ms+'::SPECTRAL_WINDOW').getcol('CHAN_FREQ').squeeze()
    nchan = freq.size
    jones = np.zeros((ntime, nant, nchan, ndir, ncorr), dtype=np.complex128)
    for p in range(nant):
        for c in range(ncorr):
            print(p, c)
            jones[:, p, :, 0, c] = gain_func(utime, freq, Kdict[p][c], Bdict[p][c], Gdict[p][c])

    # apply gains
    row_chunks, tbin_idx, tbin_counts = chunkify_rows(time, args.utimes_per_chunk)
    tbin_idx = da.from_array(tbin_idx, chunks=(args.utimes_per_chunk))
    tbin_counts = da.from_array(tbin_counts, chunks=(args.utimes_per_chunk))

    jones = da.from_array(jones, chunks=(args.utimes_per_chunk, -1, args.chan_chunks, -1, -1))

    cols = (args.readcol,)
    if args.applycol is not None:
        cols += (args.applycol,)

    xds = xds_from_ms(args.ms, group_cols=('FIELD_ID', 'DATA_DESC_ID'), columns=cols, chunks={'row':row_chunks, 'chan':args.chan_chunks})

    data = getattr(xds, args.readcol)
    nrow, nchan, ncorr = data.shape
    
    # reshape the correlation axis if required
    if ncorr > 2:
        data = data.reshape(nrow, nchan, 2, 2)
        reshape_vis = True
    else:
        reshape_vis = False

    if args.applycol is not None:
        model = getattr(args.applycol)
    else:
        model = da.ones_like(data, chunks=data.chunks)

    # add direction axis and reshape if required
    if reshape_vis:
        model = model.reshape(nrow, nchan, 1, 2, 2)
    else:
        model = model.reshape(nrow, nchan, 1, 2)

    ant1 = xds.ANTENNA1.data
    ant2 = xds.ANTENNA2.data

    corrupted_data = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2, jones, model)

    if reshape_vis:
        corrupted_data = corrupted_data.reshape(nrow, nchan, ncorr)

    xds = xds.assign(**{args.writecol: (("row", "chan", "corr"), corrupted_data)})
    write = xds_to_table(xds, args.ms, [args.writecol])
    with ProgressBar():
        write.compute()



if __name__=="__main__":
    args = create_parser().parse_args()

    if args.nthreads:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.nthreads))
    else:
        import multiprocessing
        args.nthreads = multiprocessing.cpu_count()

    if not isinstance(args.ms, list):
        args.ms = [args.ms]

    GD = vars(args)
    print('Input Options:')
    for key in GD.keys():
        print(key, ' = ', GD[key])
    

    main(args)

