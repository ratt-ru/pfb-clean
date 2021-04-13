import numpy as np
from pfb.opt import power_method, pcg, primal_dual
from pfb.operators import PSF, DaskPSI
from pfb.prox import prox_21
import pyscilog
log = pyscilog.get_logger('SARA')


def resid_func(x, dirty, psfo, mask, beam):
    """
    Returns the unattenuated but masked residual_mfs useful for peak finding
    and the masked residual cube attenuated by the beam pattern which is the
    data source in the pcg forward step. Masked regions are set to zero.
    """
    residual = dirty - psfo.convolve(mask(beam(x)))
    residual_mfs = np.sum(residual, axis=0)
    residual = mask(beam(residual))
    return residual, mask(residual_mfs)


def sara(psf, model, residual, sig_21=1e-6, sigma_frac=0.5, mask=None,
         beam=None, dual=None, weights21=None, nthreads=1, maxit=10,
         gamma=0.99,  tol=1e-3, psi_levels=3, psi_basis=None,
         reweight_iters=None, reweight_alpha_ff=0.5,
         reweight_alpha_percent=10,
         pdtol=1e-6, pdmaxit=250, pdverbose=1, positivity=True, tidy=True,
         cgtol=1e-6, cgminit=25, cgmaxit=150, cgverbose=1,
         pmtol=1e-5, pmmaxit=50, pmverbose=1):

    if len(residual.shape) > 3:
        raise ValueError("Residual must have shape (nband, nx, ny)")

    nband, nx, ny = residual.shape

    if beam is None:
        def beam(x): return x
    else:
        try:
            assert beam.shape == (nband, nx, ny)
            def beam(x): return beam * x
        except Exception as e:
            raise e

    if mask is None:
        def mask(x): return x
    else:
        try:
            if mask.ndim == 2:
                assert mask.shape == (nx, ny)
                def mask(x): return mask[None] * x
            elif mask.ndim == 3:
                assert mask.shape == (1, nx, ny)
                def mask(x): return mask * x
            else:
                raise BaseException
        except BaseException:
            raise ValueError("Mask has incorrect shape")

    # PSF operator
    psfo = PSF(psf, nthreads=nthreads, imsize=residual.shape)
    if model.any():
        dirty = residual + psfo.convolve(mask(beam(model)))
    else:
        dirty = residual
    residual_mfs = np.sum(residual, axis=0)
    residual = mask(beam(residual))
    rmax = np.abs(residual_mfs).max()
    rms = np.std(residual_mfs)

    # wavelet dictionary
    if psi_basis is None:
        psi = DaskPSI(imsize=residual.shape,
                      nlevels=psi_levels, nthreads=nthreads)
    else:
        if not isinstance(psi_basis, list):
            psi_basis = [psi_basis]
        psi = DaskPSI(imsize=residual.shape, nlevels=psi_levels,
                      nthreads=nthreads, bases=psi_basis)

    # l21 weights and dual
    if weights21 is None:
        print("Initialising all l21 weights to unity.", file=log)
        weights21 = np.ones((psi.nbasis, psi.nmax), dtype=residual.dtype)
    if dual is None:
        dual = np.zeros((psi.nbasis, nband, psi.nmax), dtype=residual.dtype)

    # l21 reweighting
    if reweight_iters is not None:
        reweight_iters = list(reweight_iters)
    else:
        reweight_iters = []

    #  preconditioning operator
    def hess(x):
        return psfo.convolve(x) + x / (sigma_frac*rmax)

    if tidy:
        # spectral norm
        posthess = hess
        beta, betavec = power_method(hess, residual.shape, tol=pmtol,
                                     maxit=pmmaxit, verbosity=pmverbose)
    else:
        def posthess(x): return x
        beta = 1.0
        betavec = 1.0

    # deconvolve
    for i in range(0, maxit):
        def M(x): return x * (sigma_frac*rmax)  # preconditioner
        x = pcg(hess, residual, np.zeros_like(residual), M=M, tol=cgtol,
                maxit=cgmaxit, minit=cgminit, verbosity=cgverbose)

        # update model
        modelp = model
        model = modelp + gamma * x
        model, dual = primal_dual(posthess, model, modelp, dual, sig_21, psi,
                                  weights21, beta, prox_21, tol=pdtol,
                                  maxit=pdmaxit, report_freq=25, mask=mask,
                                  verbosity=pdverbose, positivity=positivity)

        # reweighting
        if i in reweight_iters:
            l2_norm = np.linalg.norm(dual, axis=1)
            for m in range(psi.nbasis):
                indnz = l2_norm[m].nonzero()
                alpha = np.percentile(
                    l2_norm[m, indnz].flatten(), reweight_alpha_percent)
                alpha = np.maximum(alpha, 1e-8)  # hardcode minimum
                weights21[m] = alpha/(l2_norm[m] + alpha)
            reweight_alpha_percent *= reweight_alpha_ff

        # get residual
        residual, residual_mfs = resid_func(model, dirty, psfo, mask, beam)

        # check stopping criteria
        rmax = np.abs(residual_mfs).max()
        rms = np.std(residual_mfs)
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        print("Iter %i: peak residual = %f, rms = %f, eps = %f" % (
            i+1, rmax, rms, eps), file=log)

        if eps < tol:
            print("Success, convergence after %i iterations" % (i+1),
                  file=log)
            break

        if tidy and i < maxit-1:
            beta, betavec = power_method(
                hess, residual.shape, b0=betavec, tol=pmtol, maxit=pmmaxit)

    return model, dual, residual_mfs, weights21
