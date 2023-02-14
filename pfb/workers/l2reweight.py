from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('L2REWEIGHT')

from scabha.schema_utils import clickify_parameters
from pfb.parser.schemas import schema

import numpy as np

# create default parameters from schema
defaults = {}
for key in schema.l2reweight["inputs"].keys():
    defaults[key] = schema.l2reweight["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.l2reweight)
def l2reweight(**kw):
    '''
    Initialise data products for imaging
    '''
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'l2reweight_{timestamp}.log')

    if opts.nworkers is None:
        if opts.scheduler=='distributed':
            opts.nworkers = opts.nband
        else:
            opts.nworkers = 1

    if opts.product.upper() not in ["I"]:
                                    # , "Q", "U", "V", "XX", "YX", "XY",
                                    # "YY", "RR", "RL", "LR", "LL"]:
        raise NotImplementedError(f"Product {opts.product} not yet supported")

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log, scheduler=opts.scheduler)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _l2reweight(**opts)

def _l2reweight(**kw):
    opts = OmegaConf.create(kw)

    basename = f'{opts.output_filename}_{opts.product.upper()}'

    xds_name = f'{basename}.xds.zarr'
    mds_name = f'{basename}{opts.postfix}.mds.zarr'

    from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
    from africanus.gridding.wgridder.dask import model as im2vis
    import dask.array as da
    from pfb.utils.beam import eval_beam
    import xarray as xr
    from ducc0.wgridder import dirty2ms
    import dask

    xds = xds_from_zarr(xds_name, chunks={'row': -1, 'chan': -1})
    mds = xds_from_zarr(mds_name, chunks={'band': 1, 'nx': 1, 'ny': 1})[0]

    model = mds.CLEAN_MODEL.values # dask array use .values for numpy
    nband, nx, ny = model.shape
    out_datasets = []


    for ds in xds:
        b = ds.bandid

        cell_rad = mds.cell_rad
        cell_deg = np.rad2deg(cell_rad)
        l = (-(nx//2) + da.arange(nx)) * cell_deg
        m = (-(ny//2) + da.arange(ny)) * cell_deg
        ll, mm = da.meshgrid(l, m, indexing='ij')
        bvals = eval_beam(ds.BEAM.data, ll, mm)


        modelb = model[b]
        uvw = ds.UVW.values
        freq = ds.FREQ.values

        print(f'freq shape : {freq.shape}', file=log)

        from pfb.operators.hessian import hessian
        hessopts = {
            'cell': cell_rad,
            'wstack': opts.wstack,
            'epsilon': opts.epsilon,
            'double_accum': opts.double_accum,
            'nthreads': opts.nvthreads
        }
        # we only want to apply the beam once here
        wgt = ds.WEIGHT.data
        mask = ds.MASK.data
        model = np.array(hessian(bvals * model[ds.bandid], uvw, wgt,
                                    mask, freq, None, hessopts)).squeeze()

        model_vis = dirty2ms(uvw=uvw,
                                freq=freq,
                                dirty=model,
                                pixsize_x=mds.cell_rad,
                                pixsize_y=mds.cell_rad,
                                epsilon=opts.epsilon,
                                nthreads=opts.nthreads)


        dof = opts.dof 
        res = ds.VIS.values - model_vis
        res = np.expand_dims(res, axis=-1)

        sigma2 = np.linalg.norm(res)**2/len(res.reshape(-1))

        if dof < 1:
            new_wgt = np.ones_like(ds.WEIGHT.values)
            # new_wgt[0,0] = 1e4
        else:
            new_wgt = (dof + 1) / (dof + (1/sigma2)*np.linalg.norm(res, axis=-1)**2)

        print(new_wgt.shape, file=log)

        ds_out = ds.assign(**{'WEIGHT': (('row', 'chan'), new_wgt)})
        out_datasets.append(ds_out)


    writes = xds_to_zarr(out_datasets, xds_name, columns='WEIGHT')

    dask.compute(writes)



    print('All done here')
