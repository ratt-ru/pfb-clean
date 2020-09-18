import numpy as np
import dask.array as da
from scipy.linalg import norm
from pfb.opt import pcg, power_method, simple_pd
from pfb.utils import load_fits, save_fits, data_from_header, prox_21, str2bool, compare_headers
from pfb.operators import Prior, PSF, PSI, DaskPSI
from astropy.io import fits
import argparse

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dirty", type=str,
                   help="Fits file with dirty cube")
    p.add_argument("--psf", type=str,
                   help="Fits file with psf cube")
    p.add_argument("--outfile", default='image', type=str,
                   help='base name of output.')
    p.add_argument("--ncpu", default=0, type=int,
                   help='Number of threads to use.')
    p.add_argument("--gamma", type=float, default=0.9,
                   help="Step size of 'primal' update.")
    p.add_argument("--maxit", type=int, default=20,
                   help="Number of iterations")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Tolerance")
    p.add_argument("--report_freq", type=int, default=2,
                   help="How often to save output images during deconvolution")
    p.add_argument("--beta", type=float, default=None,
                   help="Lipschitz constant of F")
    p.add_argument("--sig_l2", default=1.0, type=float,
                   help="The strength of the l2 norm regulariser")
    p.add_argument("--lfrac", default=0.2, type=float,
                   help="The length scale of the frequency prior will be lfrac * fractional bandwidth")
    p.add_argument("--sig_21", type=float, default=1e-3,
                   help="Strength of l21 regulariser")
    p.add_argument("--use_psi", type=str2bool, nargs='?', const=True, default=False,
                   help="Use SARA basis")
    p.add_argument("--psi_levels", type=int, default=2,
                   help="Wavelet decomposition level")
    p.add_argument("--x0", type=str, default=None,
                   help="Initial guess in form of fits file")
    p.add_argument("--reweight_iters", type=int, default=None, nargs='+',
                   help="Set reweighting iters explicitly")
    p.add_argument("--reweight_start", type=int, default=10,
                   help="When to start l1 reweighting scheme")
    p.add_argument("--reweight_freq", type=int, default=2,
                   help="How often to do l1 reweighting")
    p.add_argument("--reweight_end", type=int, default=90,
                   help="When to end the l1 reweighting scheme")
    p.add_argument("--reweight_alpha_min", type=float, default=1.0e-7,
                   help="Determines how aggressively the reweighting is applied."
                   " >= 1 is very mild whereas << 1 is aggressive.")
    p.add_argument("--reweight_alpha_percent", type=float, default=30)
    p.add_argument("--reweight_alpha_ff", type=float, default=0.5,
                   help="Determines how quickly the reweighting progresses.")
    p.add_argument("--cgtol", type=float, default=1e-2,
                   help="Tolerance for cg updates")
    p.add_argument("--cgmaxit", type=int, default=10,
                   help="Maximum number of iterations for the cg updates")
    p.add_argument("--cgverbose", type=int, default=0,
                   help="Verbosity of cg method used to invert Hess. Set to 1 or 2 for debugging.")
    p.add_argument("--pmtol", type=float, default=1e-14,
                   help="Tolerance for power method used to compute spectral norms")
    p.add_argument("--pmmaxit", type=int, default=25,
                   help="Maximum number of iterations for power method")
    p.add_argument("--pdtol", type=float, default=1e-4,
                   help="Tolerance for primal dual")
    p.add_argument("--pdmaxit", type=int, default=50,
                   help="Maximum number of iterations for primal dual")
    return p

def main(args):
    # load dirty and psf
    dirty = load_fits(args.dirty)
    real_type = dirty.dtype
    hdr = fits.getheader(args.dirty)
    freq = data_from_header(hdr, axis=3)
    l_coord = data_from_header(hdr, axis=1)
    m_coord = data_from_header(hdr, axis=2)
    
    nband, nx, ny = dirty.shape
    psf_array = load_fits(args.psf)
    hdr_psf = fits.getheader(args.psf)
    try:
        assert np.array_equal(freq, data_from_header(hdr_psf, axis=3))
    except:
        raise ValueError("Fits frequency axes dont match")

    print("Image size is (%i, %i, %i)"%(nband, nx, ny))
    
    psf_max = np.amax(psf_array.reshape(nband, 4*nx*ny), axis=1)
    wsum = np.sum(psf_max)
    psf_max[psf_max < 1e-15] = 1e-15

    dirty_mfs = np.sum(dirty, axis=0)/wsum 
    rmax = np.abs(dirty_mfs).max()
    rms = np.std(dirty_mfs)
    print("Peak of dirty is %f and rms is %f"%(rmax, rms))

    psf_mfs = np.sum(psf_array, axis=0)/wsum 
   
    # set operators
    psf = PSF(psf_array, args.ncpu, sigma0=args.sig_l2)
    K = Prior(args.sig_l2, nband, nx, ny, nthreads=args.ncpu)
    
    # def hess(x):
    #     return psf.convolve(x) + K.iconvolve(x)

    # get Lipschitz constant
    if args.beta is None:
        from pfb.opt import power_method
        beta = power_method(psf.hess, dirty.shape, tol=args.pmtol, maxit=args.pmmaxit)
    else:
        beta = args.beta
    print("beta = ", beta)

    # Reweighting
    if args.reweight_iters is not None:
        reweight_iters = args.reweight_iters
    else:  
        reweight_iters = list(np.arange(args.reweight_start, args.reweight_end, args.reweight_freq))
        reweight_iters.append(args.reweight_end)

    # Reporting    
    report_iters = list(np.arange(0, args.maxit, args.report_freq))
    if report_iters[-1] != args.maxit-1:
        report_iters.append(args.maxit-1)

    # set up wavelet basis
    if args.use_psi:
        nband, nx, ny = dirty.shape
        psi = PSI(nband, nx, ny, nlevels=args.psi_levels)
        nbasis = psi.nbasis
        weights_21 = np.ones((psi.nbasis, psi.nmax), dtype=real_type)
    else:
        psi = None
        weights_21 = np.ones(nx*ny, dtype=real_type)

    # initalise model
    if args.x0 is None:
        model = np.zeros(dirty.shape, dtype=real_type)
        dual = np.zeros((psi.nbasis, nband, psi.nmax), dtype=real_type)
        residual = dirty
    else:
        compare_headers(hdr, fits.getheader(args.x0))
        model = load_fits(args.x0).astype(real_type)
        dual = np.zeros((psi.nbasis, nband, psi.nmax), dtype=real_type)
        residual = dirty - psf.convolve(model)
    
    residual_mfs = np.sum(residual, axis=0)/wsum 
    rmax = np.abs(residual_mfs).max()
    rms = np.std(residual_mfs)
    print("At iteration 0 peak of residual is %f and rms is %f" % (rmax, rms))

    # deconvolve
    for k in range(args.maxit):
        x = pcg(psf.hess, residual, np.zeros(dirty.shape, dtype=real_type), M=K.dot, 
                tol=args.cgtol, maxit=args.cgmaxit, verbosity=args.cgverbose)
        
        modelp = model.copy()
        model = modelp + args.gamma * x

        if args.use_psi:
            model, dual = simple_pd(psf.hess, model, modelp, dual, args.sig_21, psi, weights_21, beta, tol=args.pdtol, maxit=args.pdmaxit, report_freq=10)
        else:
            model = prox_21(model, args.sig_21, weights_21, psi=psi, positivity=True)

        # convergence check
        normx = norm(model)
        if np.isnan(normx) or normx == 0.0:
            normx = 1.0
        
        eps = norm(model-modelp)/normx
        if eps < args.tol:
            break

        # reweighting
        if k  in reweight_iters:
            if psi is None:
                l2norm = norm(model.reshape(nband, npix), axis=0)
                weights_21 = 1.0/(l2norm + alpha)
            else:
                v = psi.hdot(model)
                l2_norm = norm(v, axis=1)
                for m in range(psi.nbasis):
                    indnz = l2_norm[m].nonzero()
                    alpha = np.percentile(l2_norm[m, indnz].flatten(), args.reweight_alpha_percent)
                    alpha = np.maximum(alpha, args.reweight_alpha_min)
                    # alpha = args.reweight_alpha_min
                    print("Reweighting - ", m, alpha)
                    weights_21[m] = 1.0/(l2_norm[m] + alpha)
                args.reweight_alpha_percent *= args.reweight_alpha_ff
                print(" reweight alpha percent = ", args.reweight_alpha_percent)

        # get residual
        residual = dirty - psf.convolve(model)
       
        # check stopping criteria
        residual_mfs = np.sum(residual, axis=0)/wsum 
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)

        # reporting
        if k in report_iters:
            save_fits(args.outfile + str(k+1) + '_model.fits', model, hdr, dtype=real_type)
            
            model_mfs = np.mean(model, axis=0)
            save_fits(args.outfile + str(k+1) + '_model_mfs.fits', model_mfs, hdr)

            save_fits(args.outfile + str(k+1) + '_update.fits', x, hdr)

            save_fits(args.outfile + str(k+1) + '_residual.fits', residual, hdr, dtype=real_type)

            save_fits(args.outfile + str(k+1) + '_residual_mfs.fits', residual_mfs, hdr)

        print("At iteration %i peak of residual is %f, rms is %f, current eps is %f" % (k+1, rmax, rms, eps))

    
    # save final results
    save_fits(args.outfile + '_model.fits', model, hdr, dtype=real_type)

    residual = dirty - psf.convolve(model)

    save_fits(args.outfile + '_residual.fits', residual/psf_max[:, None, None], hdr)


if __name__=="__main__":
    args = create_parser().parse_args()

    if args.ncpu:
        from multiprocessing.pool import ThreadPool
        import dask
        dask.config.set(pool=ThreadPool(args.ncpu))
    else:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()

    GD = vars(args)
    print('Input Options:')
    for key in GD.keys():
        print(key, ' = ', GD[key])

    main(args)


