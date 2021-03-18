import numpy as np
from numba import njit
import numexpr as ne
import pyscilog
log = pyscilog.get_logger('HOGBOM')

def hogbom(
        ID,
        PSF,
        gamma=0.1,
        pf=0.1,
        maxit=10000,
        report_freq=1000,
        verbosity=1):
    nband, nx, ny = ID.shape
    x = np.zeros((nband, nx, ny), dtype=ID.dtype)
    IR = ID.copy()
    IRsearch = np.sum(IR, axis=0)**2
    pq = IRsearch.argmax()
    p = pq//ny
    q = pq - p*ny
    IRmax = np.sqrt(IRsearch[p, q])

    tol = pf * IRmax
    k = 0
    while IRmax > tol and k < maxit:
        xhat = IR[:, p, q]
        x[:, p, q] += gamma * xhat
        ne.evaluate('IR - gamma * xhat * psf', local_dict={
                    'IR': IR,
                    'gamma': gamma,
                    'xhat': xhat[:, None, None],
                    'psf':PSF[:, nx - p:2 * nx - p, ny - q:2 * ny - q]},
                    out=IR, casting='same_kind')
        IRsearch = np.sum(IR, axis=0)**2
        pq = IRsearch.argmax()
        p = pq//ny
        q = pq - p*ny
        IRmax = np.sqrt(IRsearch[p, q])
        k += 1

        if not k % report_freq and verbosity > 1:
            print("At iteration %i max residual = %f" % (k, IRmax), file=log)

    if k >= maxit:
        if verbosity:
            print(
                "Maximum iterations reached. Max of residual = %f.  " %
                (IRmax), file=log)
    else:
        if verbosity:
            print("Success, converged after %i iterations" % k, file=log)
    return x