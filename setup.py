from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("rtlib_cython", ["rtlib/rtlib_cython.pyx"], include_dirs=[numpy.get_include()])]

setup(
    name = 'rtpipe',
    description = 'Python scripts for fast transient searches with radio interferometer data',
    author = 'Casey Law',
    author_email = 'caseyjlaw@gmail.com',
    version = '1.26',
    url = 'http://github.com/caseyjlaw/rtpipe',
    packages = find_packages(),        # get all python scripts in realtime
    dependency_links = ['http://github.com/caseyjlaw/sdmpy', 'http://github.com/caseyjlaw/sdmreader'],
    install_requires=['cython'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
