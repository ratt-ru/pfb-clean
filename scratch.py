import numpy as np
np.random.seed(420)
from pfb.operators import Prior
from africanus.linalg import kronecker_tools as kt
from ducc0.fft import r2c, c2r, c2c
from time import time


def expsq(x, sigmaf, l):
    N = x.size
    xx = np.abs(np.tile(x, (N, 1)).T - np.tile(x, (N, 1)))
    return sigmaf**2*np.exp(-xx**2/(2*l**2))


def expsq_hat(s, sigmaf, l):
    # print(np.sqrt(2*np.pi*l**2))
    # print(sigmaf**2)
    # print(np.exp(-2 * np.pi**2 * l**2 * s**2) )
    return np.sqrt(2*np.pi*l**2) * sigmaf**2.0 * np.exp(-2 * np.pi**2 * l**2 * s**2) 

from scipy.special import comb
def smoothstep(x, x_min, x_max, N=5):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result

if __name__=="__main__":
    nx = 256
    ny = 256
    nchan = 8

    x = np.zeros((nchan, 2*nx, 2*ny), dtype=np.float64)
    x[:, nx//2:3*nx//2, ny//2:3*ny//2] = 1.0

    

    
    # Kop = Prior(1.0, nchan, nx, ny, 8)

    # xi = np.random.randn(nchan, nx, ny)
    # xi = np.zeros((nchan, nx, ny))
    # xi[10:22, 10:22, 10:22] = np.random.randn(12, 12, 12)
    # xi[:, nx//4, ny//4] = 1.0
    # xi[:, 3*nx//4, ny//2] = 1.0
    
    

    # ti = time()
    # res2 = Kop.dothat(xi).real
    # print("dot = ", time()-ti)


    # ti = time()
    # res1 = Kop.dot(xi)
    # print(" dot Vanilla = ", time()-ti)

    # # print("dot diff = ", np.abs(res1 - res2).max())

    # xirec1 = Kop.idot(res1)

    # print(" idot Vanilla = ", np.abs(xi - xirec1).max())

    # xirec2 = Kop.idothat(res2)

    # print(" idot = ", np.abs(xi - xirec2).max())

    # ti = time()
    # xipad = np.zeros((2*nchan-2, 2*nx-2, 2*ny-2), dtype=np.complex128)
    # xipad[0:nchan, 0:nx, 0:ny] = res2
    # xihat = c2c(xipad, axes=(0, 1, 2), forward=True, nthreads=8)
    # res3 = c2c(xihat/S, axes=(0,1,2), forward=False, inorm=2, nthreads=8)[0:nchan, 0:nx, 0:ny]
    # print("idot = ", time()-ti)
    
    # print(np.abs(xi - res3).max())

    # from time import time
    # ti = time()
    # tmp = Kop.dot(xi)
    # print(time() - ti)
    # ti = time()
    # rec = Kop.idot(tmp)
    # print(time() - ti)

    # print(np.abs(xi - rec).max())

    # test spectral
    # xihat = np.zeros(Kop.)
    # xihat = c2c(xi, xi)

    # res2 = Kop.dot(xi)

    # res = Kop.convolve(xi)

    # print(np.abs(res-res2).max())
    # rec = Kop.iconvolve(res)

    # print(np.abs(rec-xi).max())

    

    import matplotlib.pyplot as plt
    
    plt.imshow(x[0])
    plt.show()

    # plt.figure('fft')
    # plt.plot(sv, chatv.real, '.')

    # plt.figure('S')
    # plt.plot(sv, Sv, '.')

    # plt.figure('diff')
    # plt.plot(sv, chatv.real - Sv, '.')    

    # plt.figure('hist')
    # plt.hist((xi - res3).real)
    
    # plt.show()

    



