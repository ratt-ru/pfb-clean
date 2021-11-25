# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('BACKWARD')

@cli.command()
@click.option('-xds', '--xds', type=str, required=True,
              help="Path to xarray dataset containing data products")
@click.option('-u', '--update', type=str, required=True,
              help='Path to update.fits')
@click.option('-mds', '--mds', type=str,
              help="Path to xarray dataset containing model, dual and "
              "weights from previous iteration.")
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-nb', '--nband', type=int, required=True,
              help="Number of imaging bands")
@click.option('-rchunk', '--row-chunk', type=int, default=-1,
              help="Number of rows in a chunk.")
@click.option('-mask', '--mask',
              help="Path to mask.fits.")
@click.option('-pmask', '--point-mask',
              help="Path to point source mask.fits.")
@click.option('-bases', '--bases', default='self',
              help='Wavelet bases to use. Give as str separated by | eg.'
              '-bases self|db1|db2|db3|db4')
@click.option('-nlevels', '--nlevels', default=3,
              help='Number of wavelet decomposition levels')
@click.option('-hessnorm', '--hessnorm', type=float,
              help="Spectral norm of Hessian approximation")
@click.option('-otype', '--output-type', default='f4',
              help="Data type of output")
@click.option('-eps', '--epsilon', type=float, default=1e-5,
              help='Gridder accuracy')
@click.option('-sinv', '--sigmainv', type=float, default=1e-3,
              help='Standard deviation of assumed GRF prior used '
              'for preconditioning.')
@click.option('-sig21', '--sigma21', type=float, default=1e-3,
              help='Sparsity threshold level.')
@click.option('-niter', '--niter', dtype=int, default=10,
              help='Number of reweighting iterations. '
              'Reweighting will take place after every primal dual run.')
@click.option('--wstack/--no-wstack', default=True)
@click.option('--double-accum/--no-double-accum', default=True)
@click.option('--use-beam/--no-use-beam', default=True)
@click.option('--use-psf/--no-use-psf', default=True)
@click.option('-pdtol', "--pd-tol", type=float, default=1e-5,
              help="Tolerance of conjugate gradient")
@click.option('-pdmaxit', "--pd-maxit", type=int, default=100,
              help="Maximum number of iterations for conjugate gradient")
@click.option('-pdverb', "--pd-verbose", type=int, default=0,
              help="Verbosity of conjugate gradient. "
              "Set to 2 for debugging or zero for silence.")
@click.option('-pdrf', "--pd-report-freq", type=int, default=10,
              help="Report freq for conjugate gradient.")
@click.option('-pmtol', "--pm-tol", type=float, default=1e-5,
              help="Tolerance of conjugate gradient")
@click.option('-pmmaxit', "--pm-maxit", type=int, default=100,
              help="Maximum number of iterations for conjugate gradient")
@click.option('-pmverb', "--pm-verbose", type=int, default=0,
              help="Verbosity of conjugate gradient. "
              "Set to 2 for debugging or zero for silence.")
@click.option('-pmrf', "--pm-report-freq", type=int, default=10,
              help="Report freq for conjugate gradient.")
@click.option('-ha', '--host-address',
              help='Address where the distributed client lives. '
              'Will use a local cluster if no address is provided')
@click.option('-nw', '--nworkers', type=int,
              help='Number of workers for the client.')
@click.option('-ntpw', '--nthreads-per-worker', type=int,
              help='Number of dask threads per worker.')
@click.option('-nvt', '--nvthreads', type=int,
              help="Total number of threads to use for vertical scaling (eg. gridder, fft's etc.)")
@click.option('-mem', '--mem-limit', type=float,
              help="Memory limit in GB. Default uses all available memory")
@click.option('-nthreads', '--nthreads', type=int,
              help="Total available threads. Default uses all available threads")
@click.option('-scheduler', '--scheduler', default='distributed',
              help="Total available threads. Default uses all available threads")
def backward(**kw):
    '''
    Solves

    argmin_x r(x) + (v - x).H U (v - x) / (2 * gamma)

    If a host address is provided the computation can be distributed
    over imaging band and row. When using a distributed scheduler both
    mem-limit and nthreads is per node and have to be specified.

    When using a local cluster, mem-limit and nthreads refer to the global
    memory and threads available, respectively. By default we will use all
    available resources.

    On a local cluster, the default is to use:

        nworkers = nband
        nthreads-per-worker = 1

    They have to be specified in ~.config/dask/jobqueue.yaml in the
    distributed case.

    if LocalCluster:
        nvthreads = nthreads//(nworkers*nthreads_per_worker)
    else:
        nvthreads = nthreads//nthreads-per-worker

    where nvthreads refers to the number of threads used to scale vertically
    (eg. the number threads given to each gridder instance).

    '''
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')

    if args.nworkers is None:
        args.nworkers = args.nband

    OmegaConf.set_struct(args, True)

    with ExitStack() as stack:
        from pfb import set_client
        args = set_client(args, stack, log)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in args.keys():
            print('     %25s = %s' % (key, args[key]), file=log)

        return _backward(**args)

def _backward(**kw):
    args = OmegaConf.create(kw)
    OmegaConf.set_struct(args, True)

    import numpy as np
    import dask
    import dask.array as da
    import xarray
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.utils.fits import load_fits, set_wcs, save_fits, data_from_header
    from pfb.operators.psi import im2coef, coef2im
    from pfb.operators.hessian import hessian_xds
    from pfb.opt.primal_dual import primal_dual
    from astropy.io import fits
    import pywt

    xds = xds_from_zarr(args.xds, chunks={'row':args.row_chunk})
    nband = xds[0].nband
    nx = xds[0].nx
    ny = xds[0].ny
    wsum = 0.0
    for ds in xds:
        wsum += ds.WSUM.values

    try:
        # always only one mds
        mds = xds_from_zarr(args.mds, chunks={'band': 1})[0]
    except Exception as e:
        mds = xarray.Dataset()

    if 'MODEL' in mds:
        model = mds.MODEL.values
        assert model.shape == (nband, nx, ny)
    else:
        model = np.zeros((nband, nx, ny), dtype=xds[0].DIRTY.dtype)

    update = load_fits(args.update).squeeze()
    data = model + update

    if args.mask is not None:
        print("Initialising mask", file=log)
        mask = load_fits(args.mask).squeeze()
        assert mask.shape == (nx, ny)
        mask = mask[None].astype(model)
    else:
        mask = np.ones((1, nx, ny), dtype=model.dtype)

    if args.point_mask is not None:
        print("Initialising point source mask", file=log)
        pmask = load_fits(args.point_mask).squeeze()
        # passing model as mask
        if len(pmask.shape) == 3:
            print("Detected third axis on pmask. "
                  "Initialising pmask from model.", file=log)
            pmask = np.any(pmask, axis=0).astype(model.dtype)

        assert pmask.shape == (nx, ny)
    else:
        pmask = np.ones((nx, ny), dtype=model.dtype)


    # dictionary setup
    print("Setting up dictionary", file=log)
    bases = args.bases.split('|')
    ntots = []
    iys = {}
    sys = {}
    for base in bases:
        if base == 'self':
            y, iy, sy = x0[0].ravel(), 0, 0
        else:
            alpha = pywt.wavedecn(pmask, base, mode='zero',
                                  level=args.nlevels)
            y, iy, sy = pywt.ravel_coeffs(alpha)
        iys[base] = iy
        sys[base] = sy
        ntots.append(y.size)

    # get padding info
    nmax = np.asarray(ntots).max()
    padding = []
    nbasis = len(ntots)
    for i in range(nbasis):
        padding.append(slice(0, ntots[i]))


    # initialise dictionary operators
    bases = da.from_array(np.array(bases, dtype=object), chunks=-1)
    ntots = da.from_array(np.array(ntots, dtype=object), chunks=-1)
    padding = da.from_array(np.array(padding, dtype=object), chunks=-1)
    psi = partial(im2coef, pmask=pmask, bases=bases, ntot=ntots, nmax=nmax,
                  nlevels=args.nlevels)
    psiH = partial(coef2im, pmask=pmask, bases=bases, padding=padding,
                   iy=iys, sy=sys, nx=nx, ny=ny)

    hessopts = {}
    hessopts['cell'] = xds[0].cell_rad
    hessopts['wstack'] = args.wstack
    hessopts['epsilon'] = args.epsilon
    hessopts['double_accum'] = args.double_accum
    hessopts['nthreads'] = args.nvthreads

    if not args.no_use_psf:
        print("Initialising psf", file=log)
        from pfb.operators.psf import psf_convolve_xds
        from ducc0.fft import r2c
        normfact = 1.0

        psf = xds[0].PSF.data
        _, nx_psf, ny_psf = psf.shape

        iFs = np.fft.ifftshift

        npad_xl = (nx_psf - nx)//2
        npad_xr = nx_psf - nx - npad_xl
        npad_yl = (ny_psf - ny)//2
        npad_yr = ny_psf - ny - npad_yl
        padding_psf = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
        unpad_x = slice(npad_xl, -npad_xr)
        unpad_y = slice(npad_yl, -npad_yr)
        lastsize = ny + np.sum(padding_psf[-1])

        # add psfhat to Dataset
        for i, ds in enumerate(xds):
            psf_pad = iFs(ds.PSF.data.compute(), axes=(1, 2))
            psfhat = r2c(psf_pad, axes=(1, 2), forward=True,
                         nthreads=args.nthreads, inorm=0)

            psfhat = da.from_array(psfhat, chunks=(1, -1, -1), name=False)
            ds = ds.assign({'PSFHAT':(('band', 'x_psf', 'y_psfo2'), psfhat)})
            xds[i] = ds

        psfopts = {}
        psfopts['padding'] = padding_psf[1:]
        psfopts['unpad_x'] = unpad_x
        psfopts['unpad_y'] = unpad_y
        psfopts['lastsize'] = lastsize
        psfopts['nthreads'] = args.nvthreads

        hess = partial(psf_convolve_xds, xds=xds, psfopts=psfopts,
                       sigmainv=args.sigmainv, wsum=wsum, mask=mask,
                       compute=True)

    else:
        print("Solving for update using vis space approximation", file=log)
        hess = partial(hessian_xds, xds=xds, hessopts=hessopts,
                       sigmainv=args.sigmainv, wsum=wsum, mask=mask,
                       compute=True)

    if args.hessnorm is None:
        from pfb.opt.power_method import power_method
        L = power_method(hess, (nband, nx, ny), tol=args.pm_tol,
                         maxit=args.pm_maxit, verbosity=args.pm_verbose,
                         report_freq=args.pm_report_freq)
    else:
        L = args.hessnorm

    if 'DUAL' in mds:
        dual = mds.DUAL.values
        assert dual.shape == (nband, nbasis, nmax)
    else:
        dual = np.zeros((nband, nbasis, nmax), dtype=model.dtype)

    if 'WEIGHT' in mds:
        weight = mds.WEIGHT.values
        assert weight.shape == (nbasis, nmax)
    else:
        weight = np.ones((nbasis, nmax), dtype=model.dtype)


    for i in range(args.niter):
        model, dual = primal_dual(hess, data, model, dual, args.sigma21,
                                  psi, psiH, weight, L, prox,
                                  tol=args.pd_tol, maxit=args.pd_maxit,
                                  verbosity=args.pd_verbose,
                                  report_freq=args.pd_report_freq)

        # # reweight
        # l2_norm = np.linalg.norm(psi.hdot(model), axis=1)
        # for m in range(psi.nbasis):
        #     if adapt_sig21:
        #         _, sigmas[m] = expon.fit(l2_norm[m], floc=0.0)
        #         print('basis %i, sigma %f'%sigmas[m], file=log)

        #     weights21[m] = alpha[m]/(alpha[m] + l2_norm[m]) * sigmas[m]/sig_21


    mds.assign(**{'MODEL': (('band', 'x', 'y'), model),
                  'DUAL': (('band', 'basis', 'coef'), dual),
                  'WEIGHT': (('basis', 'coef'), weight)})

    xds_to_zarr(mds, args.mds, columns=['MODEL','DUAL','WEIGHT']).compute()

    # compute apparent residual per dataset
    from pfb.operators.hessian import hessian
    # Required because of https://github.com/ska-sa/dask-ms/issues/171
    xdsw = xds_from_zarr(args.xds, chunks={'band': 1}, columns='DIRTY')
    writes = []
    for ds, dsw in zip(xds, xdsw):
        dirty = ds.DIRTY.data
        wgt = ds.WEIGHT.data
        uvw = ds.UVW.data
        freq = ds.FREQ.data
        fbin_idx = ds.FBIN_IDX.data
        fbin_counts = ds.FBIN_COUNTS.data
        residual = (dirty -
                    hessian(uvw, weight, freq, beam * model, None,
                    fbin_idx, fbin_counts, hessopts))
        dsw = dsw.assign(**{'RESIDUAL': (('band', 'x', 'y'), residual)})
        writes.append(dsw)


    # construct a header from xds attrs
    ra = xds.ra
    dec = xds.dec
    radec = [ra, dec]

    cell_rad = xds.cell_rad
    cell_deg = np.rad2deg(cell_rad)

    freq_out = xds.band.values
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)
    # TODO - add wsum info

    print("Saving results", file=log)
    save_fits(args.output_filename + '_update.fits', model, hdr)
    model_mfs = np.mean(model, axis=0)
    save_fits(args.output_filename + '_update_mfs.fits', model_mfs, hdr_mfs)
    save_fits(args.output_filename + '_residual.fits', residual, hdr)
    residual_mfs = np.sum(residual, axis=0)
    save_fits(args.output_filename + '_residual_mfs.fits', residual_mfs, hdr_mfs)

    print("All done here.", file=log)
