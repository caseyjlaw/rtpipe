from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("rtlib_cython2", ["rtlib_cython2.pyx"])]

setup(
    name = 'rtlib_cython2 app',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
