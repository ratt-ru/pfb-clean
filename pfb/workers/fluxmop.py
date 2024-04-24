# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
from functools import partial
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('FLUXMOP')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.fluxmop["inputs"].keys():
    defaults[key.replace("-", "_")] = schema.fluxmop["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.fluxmop)
def fluxmop(**kw):
    '''
    Forward step aka flux mop.

    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'fluxmop_{timestamp}.log')

    if opts.nworkers is None:
        if opts.scheduler=='distributed':
            opts.nworkers = opts.nband
        else:
            opts.nworkers = 1

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _fluxmop(**opts)

def _fluxmop(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import numpy as np
    import numexpr as ne
    import xarray as xr
    import dask
    import dask.array as da
    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from pfb.utils.fits import load_fits, dds2fits_mfs, dds2fits, set_wcs, save_fits
    from pfb.utils.misc import init_mask, dds2cubes
    from pfb.operators.hessian import hessian_xds, hessian_psf_cube
    from pfb.opt.pcg import pcg
    from ducc0.misc import make_noncritical

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    dds_name = f'{basename}_{opts.suffix}.dds'

    dds = xds_from_zarr(dds_name, chunks={'row':-1,
                                          'chan': -1})

    # stitch image space data products
    output_type = dds[0].DIRTY.dtype
    print("Combining slices into cubes", file=log)
    dirty, model, residual, psf, psfhat, beam, wsums, _ = dds2cubes(
                                                               dds,
                                                               opts.nband,
                                                               apparent=False,
                                                               dual=False,
                                                               modelname=opts.model_name)
    fsel = wsums > 0
    wsum = np.sum(wsums)
    psf_mfs = np.sum(psf, axis=0)
    assert (psf_mfs.max() - 1.0) < 2*opts.epsilon
    dirty_mfs = np.sum(dirty, axis=0)
    if residual is None:
        residual = dirty.copy()
        residual_mfs = dirty_mfs.copy()
    else:
        residual_mfs = np.sum(residual, axis=0)

    lastsize = dds[0].y_psf.size

    # for intermediary results (not currently written)
    freq_out = []
    for ds in dds:
        freq_out.append(ds.freq_out)
    freq_out = np.unique(np.array(freq_out))
    nband = opts.nband
    nx = dds[0].x.size
    ny = dds[0].y.size
    ra = dds[0].ra
    dec = dds[0].dec
    radec = [ra, dec]
    cell_rad = dds[0].cell_rad
    cell_deg = np.rad2deg(cell_rad)
    ref_freq = np.mean(freq_out)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)

    # set up vis space Hessian for computing the residual
    # TODO - how to apply beam externally per ds
    hessopts = {}
    hessopts['cell'] = dds[0].cell_rad
    hessopts['do_wgridding'] = opts.do_wgridding
    hessopts['epsilon'] = opts.epsilon
    hessopts['double_accum'] = opts.double_accum
    hessopts['nthreads'] = opts.nvthreads
    hess = partial(hessian_xds, xds=dds, hessopts=hessopts,
                   wsum=wsum, sigmainv=0,
                   mask=np.ones((nx, ny), dtype=output_type),
                   compute=True, use_beam=False)

    # TODO - check coordinates match
    # Add option to interp onto coordinates?
    if opts.mask is not None:
        if opts.mask=='model':
            mask = np.any(model > opts.min_model, axis=0)
            assert mask.shape == (nx, ny)
            mask = mask.astype(output_type)
            print('Using model > 0 to create mask', file=log)
        else:
            mask = load_fits(opts.mask, dtype=output_type).squeeze()
            assert mask.shape == (nx, ny)
            mask = mask.astype(output_type)
            print('Using provided fits mask', file=log)
            if opts.zero_model_outside_mask:
                model[:, mask<1] = 0
                print("Recomputing residual since asked to zero model", file=log)
                convimage = hess(model)
                ne.evaluate('dirty - convimage', out=residual,
                            casting='same_kind')
                ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                            casting='same_kind')
                save_fits(np.mean(model[fsel], axis=0),
                  basename + f'_{opts.suffix}_model_mfs_zeroed.fits',
                  hdr_mfs)
                save_fits(residual_mfs,
                  basename + f'_{opts.suffix}_residual_mfs_zeroed.fits',
                  hdr_mfs)

    else:
        mask = np.ones((nx, ny), dtype=output_type)
        print('Caution - No mask is being applied', file=log)

    if opts.use_psf:
        print("Using image space hessian approximation",
              file=log)
        xout = np.empty(dirty.shape, dtype=dirty.dtype, order='C')
        xout = make_noncritical(xout)
        xpad = np.empty(psf.shape, dtype=dirty.dtype, order='C')
        xpad = make_noncritical(xpad)
        xhat = np.empty(psfhat.shape, dtype=psfhat.dtype)
        xhat = make_noncritical(xhat)
        hess_pcg = partial(hessian_psf_cube,
                           xpad,
                           xhat,
                           xout,
                           beam*mask[None, :, :],
                           psfhat,
                           lastsize,
                           nthreads=opts.nvthreads*opts.nthreads_dask,  # not using dask parallelism
                           sigmainv=opts.sigmainv)
    else:
        print("Using vis space hessian approximation",
              file=log)
        hess_pcg = partial(hessian_xds, xds=dds, hessopts=hessopts,
                   wsum=wsum, sigmainv=opts.sigmainv,
                   mask=mask,
                   compute=True, use_beam=False)

    cgopts = {}
    cgopts['tol'] = opts.cg_tol
    cgopts['maxit'] = opts.cg_maxit
    cgopts['minit'] = opts.cg_minit
    cgopts['verbosity'] = opts.cg_verbose
    cgopts['report_freq'] = opts.cg_report_freq
    cgopts['backtrack'] = opts.backtrack

    print("Solving for update", file=log)
    x0 = np.zeros((nband, nx, ny), dtype=output_type)
    update = pcg(hess_pcg, beam * mask[None, :, :] * residual, x0,
                 tol=opts.cg_tol,
                 maxit=opts.cg_maxit,
                 minit=opts.cg_minit,
                 verbosity=opts.cg_verbose,
                 report_freq=opts.cg_report_freq,
                 backtrack=opts.backtrack)

    modelp = model.copy()
    model += opts.gamma * update


    print("Getting residual", file=log)
    convimage = hess(model)
    ne.evaluate('dirty - convimage', out=residual,
                casting='same_kind')
    ne.evaluate('sum(residual, axis=0)', out=residual_mfs,
                casting='same_kind')


    print("Updating results", file=log)
    dds_out = []
    for ds in dds:
        b = ds.bandid
        r = da.from_array(residual[b]*wsum)
        m = da.from_array(model[b])
        mp = da.from_array(modelp[b])
        u = da.from_array(update[b])
        ds_out = ds.assign(**{'RESIDUAL': (('x', 'y'), r),
                              'MODEL': (('x', 'y'), m),
                              'MODELP': (('x', 'y'), mp),  # to revert in case of failure
                              'UPDATE': (('x', 'y'), u)})
        dds_out.append(ds_out)
    writes = xds_to_zarr(dds_out, dds_name,
                         columns=('RESIDUAL', 'MODEL', 'MODELP', 'UPDATE'),
                         rechunk=True)
    # import ipdb; ipdb.set_trace()
    dask.compute(writes)

    dds = xds_from_zarr(dds_name, chunks={'x': -1, 'y': -1})



    # convert to fits files
    fitsout = []
    if opts.fits_mfs:
        fitsout.append(dds2fits_mfs(dds,
                                    'RESIDUAL',
                                    f'{basename}_{opts.suffix}_mopped',
                                    norm_wsum=True))
        fitsout.append(dds2fits_mfs(dds,
                                    'MODEL',
                                    f'{basename}_{opts.suffix}_mopped',
                                    norm_wsum=False))
        fitsout.append(dds2fits_mfs(dds,
                                    'UPDATE',
                                    f'{basename}_{opts.suffix}_mopped',
                                    norm_wsum=False))

    if opts.fits_cubes:
        fitsout.append(dds2fits(dds,
                                'RESIDUAL',
                                f'{basename}_{opts.suffix}_mopped',
                                norm_wsum=True))
        fitsout.append(dds2fits(dds,
                                'MODEL',
                                f'{basename}_{opts.suffix}_mopped',
                                norm_wsum=False))
        fitsout.append(dds2fits(dds,
                                'UPDATE',
                                f'{basename}_{opts.suffix}_mopped',
                                norm_wsum=False))

    if len(fitsout):
        print("Writing fits", file=log)
        dask.compute(fitsout)

    print("All done here.", file=log)
