import numpy as np
import numexpr as ne
from pfb.utils.misc import give_edges
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
    _, nx_psf, ny_psf = PSF.shape
    nx0 = nx_psf//2
    ny0 = ny_psf//2
    x = np.zeros((nband, nx, ny), dtype=ID.dtype)
    IR = ID.copy()
    IRsearch = np.sum(IR, axis=0)**2
    pq = IRsearch.argmax()
    p = pq//ny
    q = pq - p*ny
    IRmax = np.sqrt(IRsearch[p, q])
    wsums = np.amax(PSF, axis=(1,2))
    tol = pf * IRmax
    k = 0
    stall_count = 0
    while IRmax > tol and k < maxit and stall_count < 5:
        xhat = IR[:, p, q] / wsums
        x[:, p, q] += gamma * xhat
        ne.evaluate('IR - gamma * xhat * psf', local_dict={
                    'IR': IR,
                    'gamma': gamma,
                    'xhat': xhat[:, None, None],
                    'psf': PSF[:, nx0 - p:nx0 + nx - p,
                                  ny0 - q:ny0 + ny - q]},
                    out=IR, casting='same_kind')
        IRsearch = np.sum(IR, axis=0)**2
        pq = IRsearch.argmax()
        p = pq//ny
        q = pq - p*ny
        IRmaxp = IRmax
        IRmax = np.sqrt(IRsearch[p, q])
        k += 1

        if np.abs(IRmaxp - IRmax) / np.abs(IRmaxp) < 5e-3:
            stall_count += stall_count

        if not k % report_freq and verbosity > 1:
            print("At iteration %i max residual = %f" % (k, IRmax), file=log)

    if k >= maxit:
        if verbosity:
            print("Maximum iterations reached. Max of residual = %f." %
                  (IRmax), file=log)
    elif stall_count >= 5:
        if verbosity:
            print("Stalled. Max of residual = %f." %
                  (IRmax), file=log)
    else:
        if verbosity:
            print("Success, converged after %i iterations" % k, file=log)
    return x


import jax.numpy as jnp
from jax import jit
from jax.ops import index_add
import jax.lax as lax
@jit
def hogbom_jax(ID, PSF, x, gamma=0.1, pf=0.1, maxit=5000):
    nx, ny = ID.shape
    IR = jnp.array(ID, copy=True)
    IRsearch = jnp.square(IR)
    pq = jnp.argmax(IRsearch)
    p = pq//ny
    q = pq - p*ny
    IRmax = jnp.sqrt(IRsearch[p, q])
    tol = pf*IRmax
    k = 0

    def cond_func(inputs):

        IRmax, IR, IRsearch, PSF, x, loc, tol, gamma, k = inputs

        return (k < maxit) & (IRmax > tol)

    def body_func(inputs):
        IRmax, IR, IRsearch, PSF, x, loc, tol, gamma, k = inputs
        nx, ny = IR.shape
        p, q = loc
        xhat = IR[p, q]
        x = index_add(x, (p, q), gamma * xhat)
        modconv = lax.dynamic_slice(PSF, [nx-p, ny-q], [nx, ny])
        IR = IR - gamma * xhat * modconv
        IRsearch = jnp.square(IR)
        pq = IRsearch.argmax()
        p = pq//ny
        q = pq - p*ny
        IRmax = jnp.sqrt(IRsearch[p, q])
        return (IRmax, IR, IRsearch, PSF, x, (p, q), tol, gamma, k+1)

    init_val = (IRmax, IR, IRsearch, PSF, x, (p, q), tol, gamma, k)
    out = lax.while_loop(cond_func, body_func, init_val)

    return out[4], out[1]
