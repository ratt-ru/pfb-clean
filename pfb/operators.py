import numpy as np
import dask.array as da
from daskms import xds_from_table
import pywt
import nifty_gridder as ng
from pypocketfft import r2c, c2r
from pfb.utils import freqmul
from africanus.gps.kernels import exponential_squared as expsq
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

class Gridder(object):
    def __init__(self, uvw, freq, sqrtW, nx, ny, cell_size, nband=None, precision=1e-7, ncpu=8, do_wstacking=1):
        self.wgt = sqrtW
        self.uvw = uvw
        self.nrow = uvw.shape[0]
        self.nx = nx
        self.ny = ny
        self.cell = cell_size * np.pi/60/60/180
        self.freq = freq
        self.precision = precision
        self.nthreads = ncpu
        self.do_wstacking = do_wstacking
        self.flags = np.where(self.wgt==0, 1, 0)

        # freq mapping
        self.nchan = freq.size
        if nband is None or nband == 0:
            self.nband = self.nchan
        else:
            self.nband = nband
        step = self.nchan//self.nband
        freq_mapping = np.arange(0, self.nchan, step)
        self.freq_mapping = np.append(freq_mapping, self.nchan)
        self.freq_out = np.zeros(self.nband)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            self.freq_out[i] = np.mean(self.freq[Ilow:Ihigh])  # weighted mean?

    def dot(self, x):
        model_data = np.zeros((self.nrow, self.nchan), dtype=np.complex128)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            model_data[:, Ilow:Ihigh] = ng.dirty2ms(uvw=self.uvw, freq=self.freq[Ilow:Ihigh], dirty=x[i], wgt=self.wgt[:, Ilow:Ihigh],
                                                    pixsize_x=self.cell, pixsize_y=self.cell, epsilon=self.precision,
                                                    nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return model_data

    def hdot(self, x):
        image = np.zeros((self.nband, self.nx, self.ny), dtype=np.float64)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            image[i] = ng.ms2dirty(uvw=self.uvw, freq=self.freq[Ilow:Ihigh], ms=x[:, Ilow:Ihigh], wgt=self.wgt[:, Ilow:Ihigh],
                                   npix_x=self.nx, npix_y=self.ny, pixsize_x=self.cell, pixsize_y=self.cell, epsilon=self.precision,
                                   nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return image

    def make_psf(self):
        psf_array = np.zeros((self.nband, 2*self.nx, 2*self.ny))
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            psf_array[i] = ng.ms2dirty(uvw=self.uvw, freq=self.freq[Ilow:Ihigh],
                                       ms=self.wgt[:, Ilow:Ihigh].astype(np.complex128), wgt=self.wgt[:, Ilow:Ihigh],
                                       npix_x=2*self.nx, npix_y=2*self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                       epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking)
        return psf_array

    def convolve(self, x):
        return self.hdot(self.dot(x))


class OutMemGridder(object):
    def __init__(self, table_name, nx, ny, cell_size, freq, nband=None, field=0, precision=1e-7, ncpu=8, do_wstacking=1):
        if precision > 1e-6:
            self.real_type = np.float32
            self.complex_type = np.complex64
        else:
            self.real_type = np.float64
            self.complex_type=np.complex128

        self.nx = nx
        self.ny = ny
        self.cell = cell_size * np.pi/60/60/180
        if isinstance(field, list):
            self.field = field
        else:
            self.field = [field]
        self.precision = precision
        self.nthreads = ncpu
        self.do_wstacking = do_wstacking

        # freq mapping
        self.freq = freq
        self.nchan = freq.size
        if nband is None:
            self.nband = self.nchan
        else:
            self.nband = nband
        step = self.nchan//self.nband
        freq_mapping = np.arange(0, self.nchan, step)
        self.freq_mapping = np.append(freq_mapping, self.nchan)
        self.freq_out = np.zeros(self.nband)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            self.freq_out[i] = np.mean(self.freq[Ilow:Ihigh])

        self.chan_chunks = self.freq_mapping[1] - self.freq_mapping[0]

        # meta info for xds_from_table
        self.table_name = table_name
        self.schema = {
            "DATA": {'dims': ('chan',)},
            "IMAGING_WEIGHT": {'dims': ('chan', )},
            "UVW": {'dims': ('uvw',)},
        }

    def make_residual(self, x, v_dof=None):
        print("Making residual")
        residual = np.zeros(x.shape, dtype=x.dtype)
        xds = xds_from_table(self.table_name, group_cols=('FIELD_ID'), chunks={"row":-1, "chan": self.chan_chunks}, table_schema=self.schema)
        for ds in xds:
            if ds.FIELD_ID not in list(self.field):
                continue
            print("Processing field %i"%ds.FIELD_ID)
            data = ds.DATA.data
            weights = ds.IMAGING_WEIGHT.data
            uvw = ds.UVW.data.compute().astype(self.real_type)

            for i in range(self.nband):
                Ilow = self.freq_mapping[i]
                Ihigh = self.freq_mapping[i+1]
                weighti = weights.blocks[:, i].compute().astype(self.real_type)
                datai = data.blocks[:, i].compute().astype(self.complex_type)

                # TODO - load and apply interpolated fits beam patterns for field

                # get residual vis
                if weighti.any():
                    residual_vis = weighti * datai - ng.dirty2ms(uvw=uvw, freq=self.freq[Ilow:Ihigh], dirty=x[i], wgt=weighti,
                                                                pixsize_x=self.cell, pixsize_y=self.cell, epsilon=self.precision,
                                                                nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)

                    # make residual image
                    residual[i] += ng.ms2dirty(uvw=uvw, freq=self.freq[Ilow:Ihigh], ms=residual_vis, wgt=weighti,
                                            npix_x=self.nx, npix_y=self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                            epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return residual

    def make_dirty(self):
        print("Making dirty")
        dirty = np.zeros((self.nband, self.nx, self.ny), dtype=self.real_type)
        xds = xds_from_table(self.table_name, group_cols=('FIELD_ID'), chunks={"row":-1, "chan": self.chan_chunks}, table_schema=self.schema)
        for ds in xds:
            if ds.FIELD_ID not in list(self.field):
                continue
            print("Processing field %i"%ds.FIELD_ID)
            data = ds.DATA.data
            weights = ds.IMAGING_WEIGHT.data
            uvw = ds.UVW.data.compute().astype(self.real_type)

            for i in range(self.nband):
                Ilow = self.freq_mapping[i]
                Ihigh = self.freq_mapping[i+1]
                weighti = weights.blocks[:, i].compute().astype(self.real_type)
                datai = data.blocks[:, i].compute().astype(self.complex_type)

                # TODO - load and apply interpolated fits beam patterns for field
                if weighti.any():
                    dirty[i] += ng.ms2dirty(uvw=uvw, freq=self.freq[Ilow:Ihigh], ms=weighti*datai, wgt=weighti,
                                            npix_x=self.nx, npix_y=self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                            epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return dirty

    def make_psf(self):
        print("Making PSF")
        psf_array = np.zeros((self.nband, 2*self.nx, 2*self.ny))
        xds = xds_from_table(self.table_name, group_cols=('FIELD_ID'), chunks={"row":-1, "chan": self.chan_chunks}, table_schema=self.schema)
        for ds in xds:
            if ds.FIELD_ID not in list(self.field):
                continue
            print("Processing field %i"%ds.FIELD_ID)
            weights = ds.IMAGING_WEIGHT.data
            uvw = ds.UVW.data.compute().astype(self.real_type)

            for i in range(self.nband):
                Ilow = self.freq_mapping[i]
                Ihigh = self.freq_mapping[i+1]
                weighti = weights.blocks[:, i].compute().astype(self.real_type)

                if weighti.any():
                    psf_array[i] += ng.ms2dirty(uvw=uvw, freq=self.freq[Ilow:Ihigh], ms=weighti.astype(self.complex_type), wgt=weighti,
                                                npix_x=2*self.nx, npix_y=2*self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                                epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking)
        return psf_array


class PSF(object):
    def __init__(self, psf, nthreads):
        self.nthreads = nthreads
        self.nband, nx_psf, ny_psf = psf.shape
        nx = nx_psf//2
        ny = ny_psf//2
        npad_x = (nx_psf - nx)//2
        npad_y = (ny_psf - ny)//2
        self.padding = ((0,0), (npad_x, npad_x), (npad_y, npad_y))
        self.ax = (1,2)
        self.unpad_x = slice(npad_x, -npad_x)
        self.unpad_y = slice(npad_y, -npad_y)
        self.lastsize = ny + np.sum(self.padding[-1])
        self.psf = psf
        psf_pad = iFs(psf, axes=self.ax)
        self.psfhat = r2c(psf_pad, axes=self.ax, forward=True, nthreads=nthreads, inorm=0)

    def convolve(self, x):
        xhat = iFs(np.pad(x, self.padding, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
        xhat = c2r(xhat * self.psfhat, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
        return Fs(xhat, axes=self.ax)[:, self.unpad_x, self.unpad_y]

class Prior(object):
    def __init__(self, freq, sigma0, l, nx, ny, nthreads=8):
        self.nthreads = nthreads
        self.nx = nx
        self.ny = ny
        self.nband = freq.size
        self.freq = freq/np.mean(freq)
        self.Kv = expsq(self.freq, self.freq, sigma0, l)
        self.x0 = np.zeros((self.nband, self.nx, self.ny), dtype=freq.dtype)

        self.Kvinv = np.linalg.inv(self.Kv + 1e-12*np.eye(self.nband))

        self.L = np.linalg.cholesky(self.Kv + 1e-12*np.eye(self.nband))
        self.LH = self.L.T

    def idot(self, x):
        return freqmul(self.Kvinv, x.reshape(self.nband, self.nx*self.ny)).reshape(self.nband, self.nx, self.ny)

    def dot(self, x):
        return freqmul(self.Kv, x.reshape(self.nband, self.nx*self.ny)).reshape(self.nband, self.nx, self.ny)

    def sqrtdot(self, x):
        return freqmul(self.L, x.reshape(self.nband, self.nx*self.ny)).reshape(self.nband, self.nx, self.ny)

    def sqrthdot(self, x):
        return freqmul(self.LH, x.reshape(self.nband, self.nx*self.ny)).reshape(self.nband, self.nx, self.ny)

class PSI(object):
    def __init__(self, nband, nx, ny,
                 nlevels=2,
                 basis=['self', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8']):
        """
        Sets up operators to move between wavelet coefficients
        in each basis and the image x.

        Parameters
        ----------
        nband - number of bands
        nx - number of pixels in x-dimension
        ny - number of pixels in y-dimension
        nlevels - The level of the decomposition. Default=2
        basis - List holding basis names.
                Default is delta + first 8 DB wavelets

        Returns
        =======
        Psi - list of operators performing coeff to image where
            each entry corresponds to one of the basis elements.
        Psi_t - list of operators performing image to coeff where
                each entry corresponds to one of the basis elements.
        """
        self.real_type = np.float64
        self.nband = nband
        self.nx = nx
        self.ny = ny
        self.nlevels = nlevels
        self.P = len(basis)
        self.sqrtP = np.sqrt(self.P)
        self.basis = basis
        self.nbasis = len(basis)

        tmpx = np.zeros(self.nlevels)
        tmpx[-1] = self.nx//2 + self.nx%2
        tmpy = np.zeros(self.nlevels)
        tmpy[-1] = self.ny//2 + self.ny%2
        for i in range(self.nlevels-2, -1, -1):
            tmpx[i] = tmpx[i+1]//2
            tmpx[i] += tmpx[i+1]%2
            tmpy[i] = tmpy[i+1]//2
            tmpy[i] += tmpy[i+1]%2

        self.indx = np.append(np.array(tmpx[0]), tmpx)
        self.indy = np.append(np.array(tmpy[0]), tmpy)
        self.n = self.indx * self.indy

        self.ntot = 4 * self.indx[0] * self.indy[0]
        for i in range(2, self.nlevels+1):
            self.ntot += 3 * self.indx[i] * self.indy[i]

        self.ntot = int(self.ntot)

    def dot(self, alpha, basis_k):
        """
        Takes array of coefficients to image.
        The input does not have the form expected by pywt
        so we have to reshape it. Comes in as a flat vector
        arranged as

        [cAn, cHn, cVn, cDn, ..., cH1, cV1, cD1]

        where each entry is a flattened array and n denotes the
        level of the decomposition. This has to be restructured as

        [cAn, (cHm, cVn, cDn), ..., (cH1, cV1, cD1)]

        where entries are arrays with size defined by set_index_scheme.
        """
        base = self.basis[basis_k]
        if base == 'self':
            return alpha.reshape(self.nband, self.nx, self.ny)/self.sqrtP
        else:
            x = np.zeros((self.nband, self.nx, self.ny), dtype=self.real_type)
            for l in range(self.nband):
                # stack array back into expected shape
                n = int(self.n[0])
                idx = int(self.indx[0])
                idy = int(self.indy[0])

                alpha_rec = [alpha[l, 0:n].reshape(idx, idy)]
                ind = int(self.n[0])
                for i in range(1, self.nlevels+1):
                    n = int(self.n[i])
                    idx = int(self.indx[i])
                    idy = int(self.indy[i])
                    tpl = ()

                    for j in range(3):
                        tpl += (alpha[l, ind:ind+n].reshape(idx, idy),)
                        ind += n

                    alpha_rec.append(tpl)

                # return reconstructed image from coeff
                wave = pywt.waverec2(alpha_rec, base, mode='periodization')
                x[l, :, :] = wave / self.sqrtP

            return x

    def hdot(self, x, basis_k):
        """
        This implements the adjoint of Psi_func i.e. image to coeffs
        """
        base = self.basis[basis_k]
        if base == 'self':
            # just flatten image, no need to stack in this case
            return x.reshape(self.nband, self.nx*self.ny)/self.sqrtP
        else:
            alpha = np.zeros((self.nband, self.ntot), dtype=self.real_type)
            for l in range(self.nband):
                # decompose
                alphal = pywt.wavedec2(x[l], base, mode='periodization', level=self.nlevels)
                # stack decomp into vector
                tmp = [alphal[0].ravel()]
                for item in alphal[1::]:
                    for j in range(len(item)):
                        tmp.append(item[j].ravel())
                alpha[l] = np.concatenate(tmp)/self.sqrtP
            return alpha


import dask.array as da

def _dot_internal(alpha, base, nd, indx, indy, sqrtp, real_type):
    assert type(base) == np.ndarray and base.shape[0] == 1
    base = base[0]

    if base == "self":
        return alpha[None, ...] / sqrtp

    nband, nx, ny = alpha.shape
    nlevels = nd.shape[0] - 1
    # Note extra dimension for the single basis chunk
    x = np.zeros((1,) + alpha.shape, dtype=real_type)

    alpha = alpha.reshape(nband, nx*ny)

    for l in range(nband):
        # stack array back into expected shape
        n = ind = int(nd[0])
        idx = int(indx[0])
        idy = int(indy[0])

        alpha_rec = [alpha[l, 0:ind].reshape(idx, idy)]

        for i in range(1, nlevels + 1):
            n = int(nd[i])
            idx = int(indx[i])
            idy = int(indy[i])
            tpl = ()

            for j in range(3):
                tpl += (alpha[l, ind:ind+n].reshape(idx, idy),)
                ind += n

            alpha_rec.append(tpl)

        wave = pywt.waverec2(alpha_rec, base, mode='periodization')

        # return reconstructed image from coeff
        x[0, l, :, :] = wave / sqrtp

    return x

def _hdot_internal(x, base, ntot, nlevels, sqrtp, real_type):
    assert type(base) == np.ndarray and base.shape[0] == 1
    base = base[0]

    nband, nx, ny = x.shape

    if base == 'self':
        # just flatten image, no need to stack in this case
        return x.reshape(1, nband, nx, ny) / sqrtp

    alpha = np.zeros((1, nband, ntot), dtype=real_type)

    for l in range(nband):
        # decompose
        alphal = pywt.wavedec2(x[l], base, mode='periodization', level=nlevels)
        # stack decomp into vector
        tmp = [alphal[0].ravel()]

        for item in alphal[1::]:
            for j in range(len(item)):
                tmp.append(item[j].ravel())

        alpha[0, l] = np.concatenate(tmp) / sqrtp

    return alpha.reshape(1, nband, nx, ny)


class DaskPSI(PSI):
    def dot(self, alpha):
        # Chunk per basis
        bases = da.from_array(self.basis, chunks=1)
        # Chunk per band
        alpha = alpha.reshape(self.nband, self.nx, self.ny)
        alpha = alpha.rechunk((1, self.nx, self.ny))

        x = da.blockwise(_dot_internal, ("basis", "nband", "nx", "ny"),
                         alpha, ("nband", "nx", "ny"),
                         bases, ("basis", ),
                         self.n, None,
                         self.indx, None,
                         self.indy, None,
                         self.sqrtP, None,
                         self.real_type, None,
                         dtype=self.real_type)

        return x

    def hdot(self, x):
        # Chunk per basis
        bases = da.from_array(self.basis, chunks=1)
        # Chunk per band
        x = x.reshape(self.nband, self.nx, self.ny)
        x = x.rechunk((1, self.nx, self.ny))

        alpha = da.blockwise(_hdot_internal, ("basis", "nband", "nx", "ny"),
                             x, ("nband", "nx", "ny"),
                             bases, ("basis", ),
                             self.ntot, None,
                             self.nlevels, None,
                             self.sqrtP, None,
                             self.real_type, None,
                             dtype=self.real_type)

        return alpha
