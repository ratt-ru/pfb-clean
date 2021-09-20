# flake8: noqa
from contextlib import ExitStack
from pfb.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('J2COL')


@cli.command()
@click.option('-ms', '--ms', required=True,
              help='Path to measurement set.')
@click.option('-mc', '--mueller-column', default='CORRECTED_DATA',
              help="Column to write Mueller term to.")
@click.option('-gt', '--gain-table', required=True,
              help="QuartiCal gain table at the same resolution as MS")
@click.option('-acol', '--acol',
              help='Will apply gains to this column if supplied')
@click.option('-c2', '--compareto',
              help="Will compare corrupted vis to this column if provided. "
              "Mainly useful for testing.")
@click.option('-ctol', '--ctol', type=float, default=1e-5)
@click.option('-o', '--output-filename', type=str, required=True,
              help="Basename of output.")
@click.option('-ha', '--host-address',
              help='Address where the distributed client lives. '
              'Will use a local cluster if no address is provided')
@click.option('-nw', '--nworkers', type=int, default=1,
              help='Number of workers for the client.')
@click.option('-ntpw', '--nthreads-per-worker', type=int, default=1,
              help='Number of dask threads per worker.')
@click.option('-nvt', '--nvthreads', type=int,
              help="Total number of threads to use for vertical scaling (eg. gridder, fft's etc.)")
@click.option('-mem', '--mem-limit', type=int,
              help="Memory limit in GB. Default uses all available memory")
@click.option('-nthreads', '--nthreads', type=int,
              help="Total available threads. Default uses all available threads")
def jones2col(**kw):
    '''
    Write product of diagonal Jones matrices to 'Mueller' column
    '''
    args = OmegaConf.create(kw)
    pyscilog.log_to_file(args.output_filename + '.log')
    from glob import glob
    ms = glob(args.ms)
    try:
        assert len(ms) == 1
        args.ms = ms
    except:
        raise ValueError(f"There must be exactly one MS at {args.ms}")

    OmegaConf.set_struct(args, True)

    with ExitStack() as stack:
        from pfb import set_client
        args = set_client(args, stack, log)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in args.keys():
            print('     %25s = %s' % (key, args[key]), file=log)

        return _jones2col(**args)

def _jones2col(**kw):
    args = OmegaConf.create(kw)
    from omegaconf import ListConfig
    if not isinstance(args.ms, list) and not isinstance(args.ms, ListConfig):
        args.ms = [args.ms]
    OmegaConf.set_struct(args, True)

    import numpy as np
    from daskms.experimental.zarr import xds_from_zarr
    from daskms import xds_from_ms, xds_to_table
    import dask.array as da
    import dask
    from africanus.calibration.utils import chunkify_rows
    from africanus.calibration.utils.dask import corrupt_vis

    # only allowing NET gain
    G = xds_from_zarr(args.gain_table + '::NET')
    xds = xds_from_ms(args.ms[0], columns=['TIME'],
                      group_cols=('FIELD_ID', 'DATA_DESC_ID',
                      'SCAN_NUMBER'))

    # chunks are computed per dataset to make sure
    # they match those in the gain table
    chunks = []
    tbin_idx = []
    tbin_counts = []
    for gain, ds in zip(G, xds):
        try:
            assert gain.gains.shape[3] == 1
        except Exception as e:
            raise ValueError("Only DI gains currently supported")

        tchunks, fchunks, _, _, _ = gain.gains.chunks
        time = ds.get('TIME').values
        row_chunks, tidx, tcounts = chunkify_rows(
                                            time,
                                            utimes_per_chunk=tchunks[0],
                                            daskify_idx=True)

        tbin_idx.append(tidx)
        tbin_counts.append(tcounts)

        chunks.append({'row': row_chunks, 'chan':fchunks[0]})

    columns = ('DATA', 'FLAG', 'FLAG_ROW', 'ANTENNA1', 'ANTENNA2')
    schema = {}
    schema['FLAG'] = {'dims': ('chan', 'corr')}
    if args.acol is not None:
        columns += (args.acol,)
        schema[args.acol] = {'dims': ('chan', 'corr')}

    if args.compareto is not None:
        columns += (args.compareto,)
        schema[args.compareto] = {'dims': ('chan', 'corr')}

    xds = xds_from_ms(args.ms[0],
                      chunks=chunks,
                      columns=columns,
                      group_cols=('FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'),
                      table_schema=schema)

    # print(xds)

    # quit()

    out_data = []
    flags = []
    for tidx, tcounts, g, ds in zip(tbin_idx, tbin_counts, G, xds):
        # TODO - we should probably compare field names to be sure
        try:
            assert g.FIELD_ID == ds.FIELD_ID
            assert g.DATA_DESC_ID == ds.DATA_DESC_ID
            assert g.SCAN_NUMBER == ds.SCAN_NUMBER
        except Exception as e:
            raise ValueError("Datasets not aligned")

        nrow = ds.dims['row']
        nchan = ds.dims['chan']
        ncorr = ds.dims['corr']

        # need to swap axes for africanus
        jones = da.swapaxes(g.gains.data, 1, 2)
        flag = ds.FLAG.data
        frow = ds.FLAG_ROW.data
        ant1 = ds.ANTENNA1.data
        ant2 = ds.ANTENNA2.data

        # frow = (frow | (ant1 == ant2))
        # flag = (flag[:, :, 0] | flag[:, :, -1])
        # flag = da.logical_or(flag, frow[:, None])

        row_chunks, chan_chunks, _ = flag.chunks

        if args.acol is not None:
            if ncorr > 2:
                assert ncorr == 4
                acol = ds.get(args.acol).data.reshape(nrow, nchan, 1, 2, 2)
            else:
                acol = ds.get(args.acol).data.reshape(nrow, nchan, 1, ncorr)
        else:
            if ncorr > 2:
                assert ncorr == 4
                acol = da.ones((nrow, nchan, 1, 2, 2),
                            chunks=(row_chunks, chan_chunks, 1, -1, -1),
                            dtype=jones.dtype)
            else:
                acol = da.ones((nrow, nchan, 1, ncorr),
                            chunks=(row_chunks, chan_chunks, 1, -1),
                            dtype=jones.dtype)

        cvis = corrupt_vis(tidx, tcounts, ant1, ant2, jones, acol)

        if ncorr == 4:
            cvis = cvis.reshape(nrow, nchan, ncorr)

        # compare where unflagged
        if args.compareto is not None:
            if len(jones.shape)  == 5 and ncorr > 2:
                mode = 'diag'
            else:
                mode = 'full'
            vis = ds.get(args.compareto)
            flag = compare_vals(cvis, vis, flag, mode, args.ctol)

            flags.append(flag)

        out_ds = ds.assign(**{args.mueller_column: (("row", "chan", "corr"), cvis)})
        out_data.append(out_ds)

    writes = xds_to_table(out_data, args.ms[0], columns=[args.mueller_column])

    import pdb; pdb.set_trace()
    dask.compute(writes, flags)

    print("All done here", file=log)

def compare_vals(vis1, vis2, flag, mode, ctol):
    import dask.array as da

    flag = da.blockwise(_compare_vals, 'rfc',
                        vis1, 'rfc',
                        vis2, 'rfc',
                        flag, 'rfc',
                        mode, None,
                        ctol, None,
                        dtype=bool)
    return flag

def _compare_vals(vis1, vis2, flag, mode, ctol):
    import numpy as np
    # flag off diagonals if only jones is diag
    if mode == 'diag':
        flag[:, :, 1] = True
        flag[:, :, 2] = True
    tmp = vis1[~flag] - vis2[~flag]

    print(np.mean(tmp))
    try:
        assert np.allclose(tmp, 0.0, rtol=1, atol=ctol)
        print(f"Matches. ", file=log)
    except:
        print(f"Does not match at required tolerance. "
              f"Max difference = {np.abs(tmp).max()}", file=log)
    return flag

