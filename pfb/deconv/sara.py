import numpy as np
from pfb.opt import power_method, pcg, primal_dual
from pfb.operators import PSF, DaskPSI

def grad_func(x, dirty, psfo):
    return psfo.convolve(x) - dirty

def sara(dirty, psf, model, residual, mask, sig_21, dual=None, weights21=None, 
         nthreads=0, maxit=10, gamma=0.99,  tol=1e-3,  # options for outer optimisation
         psi_levels=3, psi_basis=None,  # sara dict options
         reweight_iters=None, reweight_alpha_ff=0.5, reweight_alpha_percent=10,  # reweighting options
         pdtol=1e-4, pdmaxit=250, pdverbose=1, positivity=True,  # primal dual options
         cgtol=1e-4, cgminit=50, cgmaxit=150, cgverbose=1,  # conjugate gradient options
         pmtol=1e-5, pmmaxit=50):  # power method options
    
    nband, nx, ny = dirty.shape
    
    # PSF operator
    psfo = PSF(psf, nthreads)

    # wavelet dictionary
    if psi_basis is None:
        psi = DaskPSI(nband, nx, ny, nlevels=psi_levels, nthreads=nthreads)
    else:
        if not isinstance(psi_basis, list):
            psi_basis = [args.psi_basis]
        psi = DaskPSI(nband, nx, ny, nlevels=psi_levels, nthreads=nthreads, bases=list(psi_basis))
    
    # l21 weights and dual 
    if weights21 is None:
        weights21 = np.ones((psi.nbasis, psi.nmax), dtype=np.float64)
    if dual is None:
        dual = np.zeros((psi.nbasis, nband, psi.nmax), dtype=np.float64)

    # l21 reweighting
    if reweight_iters is not None:
        reweight_iters = list(reweight_iters)
    else:
        reweight_iters = []
    
    # residual
    residual_mfs = np.sum(residual, axis=0)
    rms = np.std(residual_mfs)
    rmax = np.abs(residual_mfs).max()
    
    #  preconditioning operator
    def hess(x):  
        return mask*psfo.convolve(mask*x) + x / (0.2*rmax) 

    # spectral norm
    beta, betavec = power_method(hess, dirty.shape, tol=pmtol, maxit=pmmaxit)

    # deconvolve
    for i in range(0, maxit):
        M = lambda x: x * (0.2*rmax)  # preconditioner
        x = pcg(hess, mask*residual, np.zeros(dirty.shape, dtype=np.float64), M=M, tol=cgtol,
                maxit=cgmaxit, minit=cgminit, verbosity=cgverbose)
        
        # update model
        modelp = model
        model = modelp + gamma * x
        model, dual = primal_dual(hess, model, modelp, dual, sig_21, psi, weights21, beta,
                                  tol=pdtol, maxit=pdmaxit, report_freq=25, mask=mask, verbosity=pdverbose,
                                  positivity=positivity)

        # reweighting
        if i in reweight_iters:
            l2_norm = np.linalg.norm(dual, axis=1)
            for m in range(psi.nbasis):
                indnz = l2_norm[m].nonzero()
                alpha = np.percentile(l2_norm[m, indnz].flatten(), reweight_alpha_percent)
                alpha = np.maximum(alpha, 1e-8)  # hardcode minimum
                weights21[m] = alpha/(l2_norm[m] + alpha)
            reweight_alpha_percent *= reweight_alpha_ff

        # get residual
        residual = -grad_func(model, dirty, psfo)
       
        # check stopping criteria
        residual_mfs = np.sum(residual, axis=0)
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        print("     SARA - At iteration %i peak of residual is %f, rms is %f, current eps is %f" % (i+1, rmax, rms, eps))

        if eps < tol:
            print("     SARA - Success, convergence after %i iterations" %(i+1))
            break

        beta, betavec = power_method(hess, dirty.shape, b0=betavec, tol=pmtol, maxit=pmmaxit)

    return model, dual, residual_mfs, weights21