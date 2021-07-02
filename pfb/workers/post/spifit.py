# flake8: noqa
import click
from omegaconf import OmegaConf
from pfb.workers.main import cli
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('SPIFIT')

@cli.command()
@click.option('-image', '--image', required=True,
              help="Path to model or restored image cube.")
@click.option('-resid', "--residual", required=False,
              help="Path to residual image cube.")
@click.option('-o', '--output-filename', required=True,
              help="Path to output directory + prefix. "
              "Placed next to input model if outfile not provided.")
@click.option('-pp', '--psf-pars', nargs=3, type=float,
              help="Beam parameters matching FWHM of restoring beam "
                   "specified as emaj emin pa."
                   "By default these are taken from the fits header "
                   "of the residual image.")
@click.option('--circ-psf/--no-circ-psf', default=False)
@click.option('-th', '--threshold', default=10, type=float, show_default=True,
              help="Multiple of the rms in the residual to threshold on."
                   "Only components above threshold*rms will be fit.")
@click.option('-maxdr', '--maxDR', default=100, type=float, show_default=True,
              help="Maximum dynamic range used to determine the "
                   "threshold above which components need to be fit. "
                   "Only used if residual is not passed in.")
@click.option('-nthreads', '--nthreads', default=1, type=int, show_default=True,
              help="Number of threads to use.")
@click.option('-pb-min', '--pb-min', type=float, default=0.15,
              help="Set image to zero where pb falls below this value")
@click.option('-products', '--products', default='aeikIcmrb', type=str,
              help="Outputs to write. Letter correspond to: \n"
              "a - alpha map \n"
              "e - alpha error map \n"
              "i - I0 map \n"
              "k - I0 error map \n"
              "I - reconstructed cube form alpha and I0 \n"
              "c - restoring beam used for convolution \n"
              "m - convolved model \n"
              "r - convolved residual \n"
              "b - average power beam \n"
              "Default is to write all of them")
@click.option('-pf', "--padding-frac", default=0.5, type=float,
              show_default=True, help="Padding factor for FFT's.")
@click.option('-dc', "--dont-convolve", is_flag=True,
              help="Do not convolve by the clean beam before fitting")
@click.option('-rf', '--ref-freq', type=float,
              help='Reference frequency where the I0 map is sought. '
              "Will overwrite in fits headers of output.")
@click.option('-otype', '--out-dtype', default='f4', type=str,
              help="Data type of output. Default is single precision")
@click.option('-acr', '--add-convolved-residuals', is_flag=True,
              help='Flag to add in the convolved residuals before '
              'fitting components')
@click.option('-ms', "--ms", nargs="+", type=str,
                help="Mesurement sets used to make the image. \n"
                "Used to get paralactic angles if doing primary beam correction")
@click.option('-f', "--field", type=int, default=0,
                help="Field ID")
@click.option('-bm', '--beam-model', default=None, type=str,
                help="Fits beam model to use. \n"
                    "It is assumed that the pattern is path_to_beam/"
                    "name_corr_re/im.fits. \n"
                    "Provide only the path up to name "
                    "e.g. /home/user/beams/meerkat_lband. \n"
                    "Patterns mathing corr are determined "
                    "automatically. \n"
                    "Only real and imaginary beam models currently "
                    "supported.")
@click.option('-st', "--sparsify-time", type=int, default=10,
                help="Used to select a subset of time ")
@click.option('-ct', '--corr-type', type=str, default='linear',
                help="Correlation typ i.e. linear or circular. ")
@click.option('-band', "--band", type=str, default='l',
                help="Band to use with JimBeam. L or UHF")
def spifit(**kw):
    args = OmegaConf.create(kw)

    from glob import glob
    args.image = glob(args.image)
    if args.residual is not None:
        args.residual = glob(args.residual)
    OmegaConf.set_struct(args, True)
    pyscilog.log_to_file(args.output_filename + '.log')
    pyscilog.enable_memory_logging(level=3)

    import dask

    if args.nthreads:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.nthreads))
    else:
        import multiprocessing
        args.nthreads = multiprocessing.cpu_count()

    print('Input Options:', file=log)
    for key in kw.keys():
        print('     %25s = %s' % (key, args[key]), file=log)

    import dask.array as da
    import numpy as np
    from astropy.io import fits
    from africanus.model.spi.dask import fit_spi_components
    from pfb.utils.fits import load_fits, save_fits, data_from_header, set_wcs
    from pfb.utils.misc import convolve2gaussres
    # from scripts.power_beam_maker import interpolate_beam

    # check consistent number of inputs and bands
    if not isinstance(args.image, list):
        args.image = list(args.image)

    if len(args.image) > 1:
        if args.residual is not None:
            assert len(args.image) == len(args.residual)

        if isinstance(args.beam_model, list):
            try:
                assert len(args.beam_model) == len(args.image)
            except Exception as e:
                raise ValueError("You have to pass in a beam per image if you specify them manually")
        elif args.beam_model.lower() == 'jimbeam':
            args.beam_model = [args.beam_model.lower()]*len(args.image)
            if len(args.band) > 1:
                try:
                    assert len(args.band) == len(args.image)
                except Exception as e:
                    raise ValueError("Inconsistent input for --band. "
                                    "You either need to use teh same band for "
                                    "all images or provide a band per image.")
            else:
                print("Using %s band beam for all images"%args.beam_model[0], file=log)
                args.band = list(args.band)*len(args.image)

    # get max gausspars
    gaussparf = None
    for i in range(len(args.image)):
        if args.psf_pars is None:
            try:
                rhdr = fits.getheader(args.residual[i])
            except Exception as e:
                raise e

            if 'BMAJ0' in rhdr.keys():
                emaj = rhdr['BMAJ0']
                emin = rhdr['BMIN0']
                pa = rhdr['BPA0']
                gausspars = [emaj, emin, pa]
                freq_idx0 = 0
            elif 'BMAJ1' in rhdr.keys():
                emaj = rhdr['BMAJ1']
                emin = rhdr['BMIN1']
                pa = rhdr['BPA1']
                gausspars = [emaj, emin, pa]
                freq_idx0 = 1
            elif 'BMAJ' in rhdr.keys():
                emaj = rhdr['BMAJ']
                emin = rhdr['BMIN']
                pa = rhdr['BPA']
                gausspars = [emaj, emin, pa]
                freq_idx0 = 0
            else:
                raise ValueError("No beam parameters found in residual."
                                "You will have to provide them manually.")

            if gaussparf is None:
                gaussparf = gausspars
            else:
                # we need to take the max in both directions
                gaussparf[0] = np.maximum(gaussparf[0], gausspars[0])
                gaussparf[1] = np.maximum(gaussparf[1], gausspars[1])
        else:
            gaussparf = list(args.psf_pars)

        if args.circ_psf:
            e = np.maximum(gaussparf[0], gaussparf[1])
            gaussparf[0] = e
            gaussparf[1] = e
            gaussparf[2] = 0.0

    gaussparf = tuple(gaussparf)
    print("Using emaj = %3.2e, emin = %3.2e, PA = %3.2e \n" % gaussparf, file=log)

    # get required data products
    image_dict = {}
    for i in range(len(args.image)):
        image_dict[i] = {}

        # load model image
        model = load_fits(args.image[i], dtype=args.out_dtype).squeeze()
        mhdr = fits.getheader(args.image[i])

        if model.ndim < 3:
            model = model[None, :, :]

        l_coord, ref_l = data_from_header(mhdr, axis=1)
        l_coord -= ref_l
        m_coord, ref_m = data_from_header(mhdr, axis=2)
        m_coord -= ref_m
        if mhdr["CTYPE4"].lower() == 'freq':
            freq_axis = 4
            stokes_axis = 3
        elif mhdr["CTYPE3"].lower() == 'freq':
            freq_axis = 3
            stokes_axis = 4
        else:
            raise ValueError("Freq axis must be 3rd or 4th")

        freqs, ref_freq = data_from_header(mhdr, axis=freq_axis)

        image_dict[i]['freqs'] = freqs

        nband = freqs.size
        npix_l = l_coord.size
        npix_m = m_coord.size

        xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')

        # load beam
        if args.beam_model is not None:
            if args.beam_model[i] == "jimbeam":
                from katbeam import JimBeam
                if args.band[i].lower() == 'l':
                    beam = JimBeam('MKAT-AA-L-JIM-2020')
                elif args.band[i].lower() == 'uhf':
                    beam = JimBeam('MKAT-AA-UHF-JIM-2020')
                else:
                    raise ValueError("Unkown band %s"%args.band[i])

                beam_image = np.zeros(model.shape, dtype=args.out_dtype)
                for v in range(freqs.size):
                    beam_image[v] = beam.I(xx, yy, freqs[v])
            elif args.beam_model[i].endswith('.fits'):  # beam already interpolated
                bhdr = fits.getheader(args.beam_model)
                l_coord_beam, ref_lb = data_from_header(bhdr, axis=1)
                l_coord_beam -= ref_lb
                if not np.array_equal(l_coord_beam, l_coord):
                    raise ValueError("l coordinates of beam model do not match those of image. Use power_beam_maker to interpolate to fits header.")

                m_coord_beam, ref_mb = data_from_header(bhdr, axis=2)
                m_coord_beam -= ref_mb
                if not np.array_equal(m_coord_beam, m_coord):
                    raise ValueError("m coordinates of beam model do not match those of image. Use power_beam_maker to interpolate to fits header.")

                freqs_beam, _ = data_from_header(bhdr, axis=freq_axis)
                if not np.array_equal(freqs, freqs_beam):
                    raise ValueError("Freqs of beam model do not match those of image. Use power_beam_maker to interpolate to fits header.")

                beam_image = load_fits(args.beam_model, dtype=args.out_dtype).squeeze()

            else:  # interpolate from scratch
                raise NotImplementedError("Not yet there, sorry")
                # beam_image = interpolate_beam(xx, yy, freqs, args)

        else:
            beam_image = np.ones(model.shape, dtype=args.out_dtype)

        image_dict[i]['beam'] = beam_image

        if not args.dont_convolve:
            print("Convolving model %i"%i, file=log)
            # convolve model to desired resolution
            model, gausskern = convolve2gaussres(model, xx, yy, gaussparf, args.nthreads, None, args.padding_frac)

        image_dict[i]['model'] = model

        # add in residuals and set threshold
        if args.residual is not None:
            resid = load_fits(args.residual[i], dtype=args.out_dtype).squeeze()
            if resid.ndim < 3:
                resid = resid[None, :, :]
            rhdr = fits.getheader(args.residual[i])
            l_res, ref_lb = data_from_header(rhdr, axis=1)
            l_res -= ref_lb
            if not np.array_equal(l_res, l_coord):
                raise ValueError("l coordinates of residual do not match those of model")

            m_res, ref_mb = data_from_header(rhdr, axis=2)
            m_res -= ref_mb
            if not np.array_equal(m_res, m_coord):
                raise ValueError("m coordinates of residual do not match those of model")

            # freqs_res, _ = data_from_header(rhdr, axis=freq_axis)
            # if not np.array_equal(freqs, freqs_res):
            #     raise ValueError("Freqs of residual do not match those of model")

            # convolve residual to same resolution as model
            gausspari = ()
            for b in range(nband):
                key = 'BMAJ' + str(b)
                if key in rhdr.keys():
                    emaj = rhdr[key]
                    emin = rhdr['BMIN' + str(b)]
                    pa = rhdr['BPA' + str(b)]
                    gausspari += ((emaj, emin, pa),)
                elif 'BMAJ' in rhdr.keys():
                    emaj = rhdr['BMAJ']
                    emin = rhdr['BMIN']
                    pa = rhdr['BPA']
                    gausspari += ((emaj, emin, pa),)
                else:
                    print("Can't find Gausspars in residual header, unable to add residuals back in", file=log)
                    gausspari = None
                    break

            if gausspari is not None and args.add_convolved_residuals:
                print("Convolving residuals %i"%i, file=log)
                resid, _ = convolve2gaussres(resid, xx, yy, gaussparf, args.nthreads, gausspari, args.padding_frac, norm_kernel=False)
                model += resid
                print("Convolved residuals added to convolved model %i"%i, file=log)


            image_dict[i]['resid'] = resid

        else:
            image_dict[i]['resid'] = None

    # concatenate images along frequency here
    freqs = []
    model = []
    beam_image = []
    resid = []
    for i in image_dict.keys():
        freqs.append(image_dict[i]['freqs'])
        model.append(image_dict[i]['model'])
        beam_image.append(image_dict[i]['beam'])
        resid.append(image_dict[i]['resid'])
    freqs = np.concatenate(freqs, axis=0)
    Isort = np.argsort(freqs)
    freqs = freqs[Isort]

    model = np.concatenate(model, axis=0)
    model = model[Isort]

    # create header
    cell_deg = mhdr['CDELT1']
    ra = np.deg2rad(mhdr['CRVAL1'])
    dec = np.deg2rad(mhdr['CRVAL2'])
    radec = [ra, dec]
    nband, nx, ny = model.shape
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freqs)
    for i in range(1, nband+1):
        hdr['BMAJ' + str(i)] = gaussparf[0]
        hdr['BMIN' + str(i)] = gaussparf[1]
        hdr['BPA' + str(i)] = gaussparf[2]
    if args.ref_freq is None:
        ref_freq = np.mean(freqs)
    else:
        ref_freq = args.ref_freq
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)
    hdr_mfs['BMAJ'] = gaussparf[0]
    hdr_mfs['BMIN'] = gaussparf[1]
    hdr_mfs['BPA'] = gaussparf[2]

    # save convolved model
    if 'm' in args.products:
        name = args.output_filename + '.convolved_model.fits'
        save_fits(name, model, hdr, dtype=args.out_dtype)
        print("Wrote convolved model to %s" % name, file=log)

    beam_image = np.concatenate(beam_image, axis=0)
    beam_image = beam_image[Isort]

    if 'b' in args.products:
        name = args.output_filename + '.power_beam.fits'
        save_fits(name, beam_image, hdr, dtype=args.out_dtype)
        print("Wrote average power beam to %s" % name, file=log)

    if resid[0] is not None:
        resid = np.concatenate(resid, axis=0)
        resid = resid[Isort]

        if 'r' in args.products:
            name = args.output_filename + '.convolved_residual.fits'
            save_fits(name, resid, hdr, dtype=args.out_dtype)
            print("Wrote convolved residuals to %s" % name, file=log)

        # get threshold
        counts = np.sum(resid != 0)
        rms = np.sqrt(np.sum(resid**2)/counts)
        rms_cube = np.std(resid.reshape(nband, npix_l*npix_m), axis=1).ravel()
        threshold = args.threshold * rms
    else:
        print("No residual provided. Setting  threshold i.t.o dynamic range. "
              "Max dynamic range is %i " % args.maxDR, file=log)
        threshold = model.max()/args.maxDR
        rms_cube = None

    print("Threshold set to %f Jy. \n" % threshold, file=log)

    # beam cut off
    beam_min = np.amin(beam_image, axis=0)
    model = np.where(beam_min[None] > args.pb_min, model, 0.0)

    # get pixels above threshold
    minimage = np.amin(model, axis=0)
    maskindices = np.argwhere(minimage > threshold)
    nanindices = np.argwhere(minimage <= threshold)
    if not maskindices.size:
        raise ValueError("No components found above threshold. "
                        "Try lowering your threshold."
                        "Max of convolved model is %3.2e" % model.max())
    fitcube = model[:, maskindices[:, 0], maskindices[:, 1]].T
    beam_comps = beam_image[:, maskindices[:, 0], maskindices[:, 1]].T

    # set weights for fit
    if rms_cube is not None:
        print("Using RMS in each imaging band to determine weights.", file=log)
        weights = np.where(rms_cube > 0, 1.0/rms_cube**2, 0.0)
        # normalise
        weights /= weights.max()
    else:
        if args.channel_weights is not None:
            weights = np.array(args.channel_weights)
            try:
                assert weights.size == nband
            except Exception as e:
                raise ValueError("Inconsistent weighst provided.")
            print("Using provided channel weights.", file=log)
        else:
            print("No residual or channel weights provided. Using equal weights.", file=log)
            weights = np.ones(nband, dtype=np.float64)

    ncomps, _ = fitcube.shape
    fitcube = da.from_array(fitcube.astype(np.float64),
                            chunks=(ncomps//args.nthreads, nband))
    beam_comps = da.from_array(beam_comps.astype(np.float64),
                               chunks=(ncomps//args.nthreads, nband))
    weights = da.from_array(weights.astype(np.float64), chunks=(nband))
    freqsdask = da.from_array(freqs.astype(np.float64), chunks=(nband))

    print("Fitting %i components" % ncomps, file=log)
    alpha, alpha_err, Iref, i0_err = fit_spi_components(fitcube, weights, freqsdask,
                                        np.float64(ref_freq), beam=beam_comps).compute()
    print("Done. Writing output.", file=log)

    alphamap = np.zeros(model[0].shape, dtype=model.dtype)
    alphamap[...] = np.nan
    alpha_err_map = np.zeros(model[0].shape, dtype=model.dtype)
    alpha_err_map[...] = np.nan
    i0map = np.zeros(model[0].shape, dtype=model.dtype)
    i0map[...] = np.nan
    i0_err_map = np.zeros(model[0].shape, dtype=model.dtype)
    i0_err_map[...] = np.nan
    alphamap[maskindices[:, 0], maskindices[:, 1]] = alpha
    alpha_err_map[maskindices[:, 0], maskindices[:, 1]] = alpha_err
    i0map[maskindices[:, 0], maskindices[:, 1]] = Iref
    i0_err_map[maskindices[:, 0], maskindices[:, 1]] = i0_err

    if 'I' in args.products:
        # get the reconstructed cube
        Irec_cube = i0map[None, :, :] * \
            (freqs[:, None, None]/ref_freq)**alphamap[None, :, :]
        name = args.output_filename + '.Irec_cube.fits'
        save_fits(name, Irec_cube, hdr, dtype=args.out_dtype)
        print("Wrote reconstructed cube to %s" % name, file=log)

    # save alpha map
    if 'a' in args.products:
        name = args.output_filename + '.alpha.fits'
        save_fits(name, alphamap, hdr_mfs, dtype=args.out_dtype)
        print("Wrote alpha map to %s" % name, file=log)

    # save alpha error map
    if 'e' in args.products:
        name = args.output_filename + '.alpha_err.fits'
        save_fits(name, alpha_err_map, mhdr, dtype=args.out_dtype)
        print("Wrote alpha error map to %s" % name, file=log)

    # save I0 map
    if 'i' in args.products:
        name = args.output_filename + '.I0.fits'
        save_fits(name, i0map, mhdr, dtype=args.out_dtype)
        print("Wrote I0 map to %s" % name, file=log)

    # save I0 error map
    if 'k' in args.products:
        name = args.output_filename + '.I0_err.fits'
        save_fits(name, i0_err_map, mhdr, dtype=args.out_dtype)
        print("Wrote I0 error map to %s" % name, file=log)

    print("All done here", file=log)