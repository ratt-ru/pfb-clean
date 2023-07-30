"""
Pre-conditioned Forward Backward Clean algorithm

author - Landman Bester
email  - lbester@ska.ac.za
date   - 31/03/2020
"""
__version__ = '0.0.1'

import os

def set_client(opts, stack, log, scheduler='distributed'):

    from omegaconf import open_dict
    # attempt somewhat intelligent default setup
    import multiprocessing
    nthreads_max = max(multiprocessing.cpu_count(), 1)
    if opts.nvthreads is None:
        # we allocate half by default
        nthreads_tot = max(nthreads_max//2, 1)
        if opts.scheduler in ['single-threaded', 'sync']:
            with open_dict(opts):
                opts.nvthreads = nthreads_tot
        else:
            ndask_chunks = opts.nthreads_dask*opts.nworkers
            nvthreads = max(nthreads_tot//ndask_chunks, 1)
            with open_dict(opts):
                opts.nvthreads = nvthreads

    os.environ["OMP_NUM_THREADS"] = str(opts.nvthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(opts.nvthreads)
    os.environ["MKL_NUM_THREADS"] = str(opts.nvthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(opts.nvthreads)
    import numexpr as ne
    max_cores = ne.detect_number_of_cores()
    # ne_threads = min(max_cores, opts.nvthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_cores)
    os.environ["NUMBA_NUM_THREADS"] = str(max_cores)

    import dask
    if scheduler=='distributed':
        # TODO - investigate what difference this makes
        # with dask.config.set({"distributed.scheduler.worker-saturation":  1.1}):
        #     client = distributed.Client()
        # set up client
        if opts.host_address is not None:
            from distributed import Client
            print("Initialising distributed client.", file=log)
            client = stack.enter_context(Client(opts.host_address))
        else:
            if opts.nthreads_dask * opts.nvthreads > nthreads_max:
                print("Warning - you are attempting to use more threads than "
                      "available. This may lead to suboptimal performance.",
                      file=log)
            from dask.distributed import Client, LocalCluster
            print("Initialising client with LocalCluster.", file=log)
            with dask.config.set({"distributed.scheduler.worker-saturation":  1.1}):
                cluster = LocalCluster(processes=True, n_workers=opts.nworkers,
                                    threads_per_worker=opts.nthreads_dask,
                                    memory_limit=0)  # str(mem_limit/nworkers)+'GB'
                cluster = stack.enter_context(cluster)
                client = stack.enter_context(Client(cluster))

        from quartical.scheduling import install_plugin
        client.run_on_scheduler(install_plugin)
        client.wait_for_workers(opts.nworkers)
    elif scheduler in ['sync', 'single-threaded']:
        dask.config.set(scheduler=scheduler)
        print(f"Initialising with synchronous scheduler",
              file=log)
    elif scheduler=='threads':
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(opts.nthreads_dask))
        print(f"Initialising ThreadPool with {opts.nthreads_dask} threads",
              file=log)
    else:
        raise ValueError(f"Unknown scheduler option {opts.scheduler}")

    # return updated opts
    return opts


def logo():
    print("""
    ███████████  ███████████ ███████████
   ░░███░░░░░███░░███░░░░░░█░░███░░░░░███
    ░███    ░███ ░███   █ ░  ░███    ░███
    ░██████████  ░███████    ░██████████
    ░███░░░░░░   ░███░░░█    ░███░░░░░███
    ░███         ░███  ░     ░███    ░███
    █████        █████       ███████████
   ░░░░░        ░░░░░       ░░░░░░░░░░░
    """)
