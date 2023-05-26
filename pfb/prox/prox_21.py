import numpy as np


def prox_21(v, sigma, weight=None, axis=0):
    """
    Computes weighted version of

    prox_{sigma * || . ||_{21}}(v)

    Assumed that v has shape (nband, nbasis, ntot) where

    nband   - number of imaging bands
    nbasis  - number of orthogonal bases
    ntot    - total number of coefficients for each basis (must be equal)
    """
    l2_norm = np.linalg.norm(v, axis=axis)  # drops axis
    l2_soft = np.maximum(l2_norm - sigma * weight, 0.0)  # norm positive
    mask = l2_norm != 0
    ratio = np.zeros(mask.shape, dtype=v.dtype)
    ratio[mask] = l2_soft[mask] / l2_norm[mask]
    return v * np.expand_dims(ratio, axis=axis)  # restores axis
