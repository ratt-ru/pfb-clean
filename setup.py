import pfb
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='pfb',  
     version='0.0.1',
    #  scripts=['scripts/simple_spi_fitter.py',
    #           'scripts/power_beam_maker.py',
    #           'scripts/image_convolver.py',
    #           'scripts/pfb.py',
    #           'scripts/solve_x0.py',
    #           'scripts/make_dirty.py',
    #           'scripts/prep_data.py'] ,
     author="Landman Bester",
     author_email="lbester@ska.ac.za",
     description="Pre-conditioned forward-backward CLEAN algorithm",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/ratt-ru/pfb-clean",
     packages=setuptools.find_packages(),
     install_requires=[
          'matplotlib',
          'codex-africanus[complete] >= 0.2.6',
          'dask-ms[xarray]',
          'PyWavelets',
      ],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )