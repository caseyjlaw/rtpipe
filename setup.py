from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy, glob

ext_modules = [Extension("rtlib_cython", ["rtlib/rtlib_cython.pyx"], include_dirs=[numpy.get_include()])]

setup(
    name = 'rtpipe',
    description = 'Python scripts for fast transient searches with radio interferometer data',
    author = 'Casey Law',
    author_email = 'caseyjlaw@gmail.com',
    version = '1.41',
    url = 'http://github.com/caseyjlaw/rtpipe',
    data_files = [ ('notebooks', glob.glob('notebooks/*')) ],
    packages = find_packages(),        # get all python scripts in realtime
    install_requires=['cython', 'sdmpy', 'sdmreader', 'scipy', 'pwkit', 'pyfftw', 'numpy', 'click', 'matplotlib', 'requests'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    entry_points='''
        [console_scripts]
        rtpipe=rtpipe.cli:cli
'''
)
