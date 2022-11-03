# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('RESTORE')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.restore["inputs"].keys():
    defaults[key] = schema.restore["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.restore)
def restore(**kw):
    '''
    Create fits image data products (eg. restored images).
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    pyscilog.log_to_file(f'{opts.output_filename}_{opts.product}{opts.postfix}.log')
    if opts.nworkers is None:
        opts.nworkers = opts.nband

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        # numpy imports have to happen after this step
        from pfb import set_client
        set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _restore(**opts)

def _restore(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)


    import numpy as np
    from daskms.experimental.zarr import xds_from_zarr
    from pfb.utils.fits import save_fits, add_beampars, set_wcs
    from pfb.utils.misc import Gaussian2D, fitcleanbeam, convolve2gaussres

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    dds_name = f'{basename}{opts.postfix}.dds.zarr'
    mds_name = f'{basename}{opts.postfix}.mds.zarr'

    dds = xds_from_zarr(dds_name)
    # only a single mds (for now)
    mds = xds_from_zarr(mds_name, chunks={'band':1})[0]
    nband = mds.nband
    nx = mds.nx
    ny = mds.ny
    cell_rad = mds.cell_rad
    cell_deg = np.rad2deg(cell_rad)
    freq = mds.freq.data
    nx_psf = dds[0].nx_psf
    ny_psf = dds[0].ny_psf
    for ds in dds:
        assert ds.nx == nx
        assert ds.ny == ny

    try:
        output_type = dds[0].get(opts.residual_name).dtype
    except:
        raise ValueError(f"{opts.residual_name} dot in dds")
    residual = np.zeros((nband, nx, ny), dtype=output_type)
    psf = np.zeros((nband, nx_psf, ny_psf), dtype=output_type)
    wsums = np.zeros(nband)
    for ds in dds:
        b = ds.bandid
        residual[b] += ds.get(opts.residual_name).values
        psf[b] += ds.PSF.values
        wsums[b] += ds.WSUM.values[0]
    wsum = wsums.sum()
    psf_mfs = np.sum(psf, axis=0)/wsum
    residual_mfs = np.sum(residual, axis=0)/wsum
    fmask = wsums > 0
    residual[fmask] /= wsums[fmask, None, None]
    psf[fmask] /= wsums[fmask, None, None]
    # sanity check
    assert (psf_mfs.max() - 1.0) < 2e-7
    assert ((np.amax(psf, axis=(1,2)) - 1.0) < 2e-7).all()

    # fit restoring psf
    GaussPar = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)
    GaussPars = fitcleanbeam(psf, level=0.5, pixsize=1.0)  # pixel units

    cpsf_mfs = np.zeros(residual_mfs.shape, dtype=output_type)
    cpsf = np.zeros(residual.shape, dtype=output_type)

    lpsf = -(nx//2) + np.arange(nx)
    mpsf = -(ny//2) + np.arange(ny)
    xx, yy = np.meshgrid(lpsf, mpsf, indexing='ij')

    cpsf_mfs = Gaussian2D(xx, yy, GaussPar[0], normalise=False)

    for v in range(opts.nband):
        cpsf[v] = Gaussian2D(xx, yy, GaussPars[v], normalise=False)

    model = mds.get(opts.model_name).values
    if not model.any():
        print("Warning - model is empty", file=log)
    model_mfs = np.mean(model, axis=0)

    image_mfs = convolve2gaussres(model_mfs[None], xx, yy,
                                  GaussPar[0], opts.nthreads,
                                  norm_kernel=False)[0]  # peak of kernel set to unity
    image_mfs += residual_mfs
    image = np.zeros_like(model)
    for b in range(nband):
        image[b:b+1] = convolve2gaussres(model[b:b+1], xx, yy,
                                         GaussPars[b], opts.nthreads,
                                         norm_kernel=False)  # peak of kernel set to unity
        image[b] += residual[b]

    # convert pixel units to deg
    GaussPar = list(GaussPar[0])
    GaussPar[0] *= cell_deg
    GaussPar[1] *= cell_deg
    GaussPar = tuple(GaussPar)

    GaussPars = list(GaussPars)
    for i, gp in enumerate(GaussPars):
        GaussPars[i] = (gp[0]*cell_deg, gp[1]*cell_deg, gp[2])
    GaussPars = tuple(GaussPars)

    # init fits headers
    radec = (mds.ra, mds.dec)
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, np.mean(freq))
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq)

    if 'm' in opts.outputs:
        save_fits(f'{basename}{opts.postfix}.model_mfs.fits', model_mfs, hdr_mfs,
                  overwrite=opts.overwrite)

    if 'M' in opts.outputs:
        save_fits(f'{basename}{opts.postfix}.model.fits', model, hdr,
                  overwrite=opts.overwrite)

    # model does not get resolution info
    hdr_mfs = add_beampars(hdr_mfs, GaussPar)
    hdr = add_beampars(hdr, GaussPar, GaussPars)

    if 'r' in opts.outputs:
        save_fits(f'{basename}{opts.postfix}.residual_mfs.fits', residual_mfs, hdr_mfs,
                  overwrite=opts.overwrite)

    if 'R' in opts.outputs:
        save_fits(f'{basename}{opts.postfix}.residual.fits', residual, hdr,
                  overwrite=opts.overwrite)

    if 'i' in opts.outputs:
        save_fits(f'{basename}{opts.postfix}.image_mfs.fits', image_mfs, hdr_mfs,
                  overwrite=opts.overwrite)

    if 'I' in opts.outputs:
        save_fits(f'{basename}{opts.postfix}.image.fits', image, hdr,
                  overwrite=opts.overwrite)

    if 'c' in opts.outputs:
        save_fits(f'{basename}{opts.postfix}.cpsf_mfs.fits', cpsf_mfs, hdr_mfs,
                  overwrite=opts.overwrite)

    if 'C' in opts.outputs:
        save_fits(f'{basename}{opts.postfix}.cpsf.fits', cpsf, hdr,
                  overwrite=opts.overwrite)

    if opts.scheduler=='distributed':
        from distributed import get_client
        client = get_client()
        client.close()

    print("All done here", file=log)
