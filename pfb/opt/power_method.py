import numpy as np
from scipy.linalg import norm


def power_method(A, imsize, b0=None, tol=1e-5, maxit=250, verbosity=1, report_freq=25):
    if b0 is None:
        b = np.random.randn(*imsize)
        b /= norm(b)
    else:
        b = b0/norm(b0)
    beta = 1.0
    eps = 1.0
    k = 0
    while eps > tol and k < maxit:
        bp = b
        b = A(bp)
        bnorm = np.linalg.norm(b)
        betap = beta
        beta = np.vdot(bp, b)/np.vdot(bp, bp)
        b /= bnorm
        eps = np.linalg.norm(beta - betap)/betap
        k += 1

        if not k%report_freq and verbosity > 1:
            print("         At iteration %i eps = %f"%(k, eps))

    if k == maxit:
        if verbosity:
            print("         PM - Maximum iterations reached. eps = %f, current beta = %f"%(eps, beta))
    else:
        if verbosity:
            print("         PM - Success, converged after %i iterations. beta = %f"%(k, beta))
    return beta, bp
