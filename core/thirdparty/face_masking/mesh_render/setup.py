
import platform
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

"""
build command
>>> python setup.py build_ext --inplace
>>> ls *.so
"""

setup(
    name='mesh_render_cython',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([
        Extension(
            name='mesh_render_cython',
            sources=['mesh_render/mesh_render_cython.pyx', 'mesh_render/mesh_render.cpp', ],
            language='c++',
            define_macros=[("NPY_NO_DEPRECATED_API", None), ],  # disable numpy deprecation warnings
            include_dirs=[numpy.get_include()],
            extra_compile_args=['/openmp:experimental', '/arch:AVX2', '/O2'],
            extra_link_args=['/NODEFAULTLIB:libcmt', 'vcomp.lib'],
        )
    ]),
    zip_safe=False,
)
