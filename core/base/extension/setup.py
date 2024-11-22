
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

"""
build command
>>> cd core/xface/extension
>>> python setup.py build_ext --inplace
>>> ls *.so
funK.cpython-310-x86_64-linux-gnu.so  
funP.cpython-310-x86_64-linux-gnu.so  
funS.cpython-310-x86_64-linux-gnu.so  
interp_cython.cpython-310-x86_64-linux-gnu.so
"""


setup(
    name='matting_cython',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([
        Extension('funS', ['matting/funS.pyx']),
        Extension('funP', ['matting/funP.pyx']),
        Extension('funK', ['matting/funK.pyx']),
    ]),
    include_dirs=[numpy.get_include()]
)


setup(
    name='interp_cython',
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            name='interp_cython',
            sources=['interp/interp_cython.pyx', 'interp/interp.cpp'],
            language='c++',
            include_dirs=[numpy.get_include()],
        )
    ],
)
