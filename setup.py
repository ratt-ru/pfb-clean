from setuptools import setup, find_packages
import pfb

with open("README.rst", "r") as fh:
    long_description = fh.read()


setup(
     name='pfb',
     version=pfb.__version__,
     scripts=['scripts/spi_fitter.py',
              'scripts/power_beam_maker.py',
              'scripts/image_convolver.py',
              'scripts/pfbclean.py',
              'scripts/make_mask.py',
              'scripts/flag_outliers.py',
              'scripts/kgb2col.py',] ,
     author="Landman Bester",
     author_email="lbester@ska.ac.za",
     description="Pre-conditioned forward-backward CLEAN algorithm",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/ratt-ru/pfb-clean",
     packages=find_packages(),
     install_requires=[
          'matplotlib',
          'scikit-image',
          'codex-africanus[complete] >= 0.2.10',
          'dask-ms[xarray] >= 0.2.6',
          'PyWavelets',
          'zarr',
          'katbeam',
      ],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     extras_require={
         'testing' : ['packratt >= 0.1.3', 'pytest']
     }
 )