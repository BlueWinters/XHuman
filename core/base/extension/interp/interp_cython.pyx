import numpy as np
cimport numpy as np
from libcpp.string cimport string

# use the Numpy-C-API from Cython
np.import_array()

# cdefine the signature of our c function
cdef extern from "interp.h":
    void _bicubic_byte(
            const unsigned char *src,
            int width, int height, int nColor,
            int neww, int newh,
            unsigned char *dst
    )

    void _bicubic_float(
            const float *src,
            int width, int height, int nColor,
            int neww, int newh,
            float *dst
    )

def bicubic_byte(np.ndarray[const unsigned char, ndim=3, mode = "c"] src not None,
                int width, int height, int nColor,
                int neww, int newh,
                np.ndarray[unsigned char, ndim=3, mode = "c"] dst not None,
                ):
        _bicubic_byte(
        <const unsigned char*> np.PyArray_DATA(src),
        width, height, nColor,
        neww, newh,
        <unsigned char *> np.PyArray_DATA(dst),
    )

def bicubic_float(np.ndarray[const float, ndim=3, mode = "c"] src not None,
                int width, int height, int nColor,
                int neww, int newh,
                np.ndarray[float, ndim=3, mode = "c"] dst not None,
                ):
        _bicubic_float(
        <const float*> np.PyArray_DATA(src),
        width, height, nColor,
        neww, newh,
        <float*> np.PyArray_DATA(dst),
    )


