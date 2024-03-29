from setuptools import setup, find_packages
import pfb

with open("README.rst", "r") as fh:
    long_description = fh.read()

requirements = [
                'numpy',
                # 'matplotlib',
                # 'ipython',
                'scikit-image',
                'PyWavelets',
                'katbeam',
                # 'pytest<=7.4.4, >=7.3.1',
                'numexpr',
                'pyscilog >= 0.1.2',
                'Click',
                # "ipdb",
                # "numba < 0.59",
                "ducc0",
                "QuartiCal",
                # "@git+https://github.com/ratt-ru/QuartiCal.git"
                # "@bandpass_smoothing"
                "sympy",
                "stimela >= 2.0rc14"
            ]


setup(
     name='pfb-clean',
     version=pfb.__version__,
     author="Landman Bester",
     author_email="lbester@sarao.ac.za",
     description="Pre-conditioned forward-backward CLEAN algorithm",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/ratt-ru/pfb-clean",
     packages=find_packages(),
     include_package_data=True,
     zip_safe=False,
     python_requires='>=3.9',
     install_requires=requirements,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: POSIX :: Linux",
         "Topic :: Scientific/Engineering :: Astronomy",
     ],
     entry_points={'console_scripts':[
        'pfb = pfb.workers.main:cli'
        ]
     }


     ,
 )
