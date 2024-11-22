cimport cython
# import numpy as np
# cimport numpy as np


@cython.infer_types(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)



def cacuP(int[:] ih_inds,
          int[:] iw_inds,
          double[:] values,
          double[:, :, :] bgr,
          int[:, :] knows):

    cdef int ih, iw, kh, kw, k, index_in_hw, spa_h, spa_w, ic, iwh, iww

    cdef double a00, a01, a02, a03, a11, a12, a13, a22, a23
    cdef double m4x4[4][4]

    cdef Py_ssize_t h = bgr.shape[0]
    cdef Py_ssize_t w = bgr.shape[1]
    k = 0

    for ih in range(1, h-1):
        for iw in range(1, w-1):
            if knows[ih, iw] == 1:
                continue
            else:
                index_in_hw = ih * w + iw

                a00, a01, a02, a03, a11, a12, a13, a22, a23 = 0.0001,0,0,0,0.0001,0,0,0.0001,0

                for iwh in range(-1, 2):
                    for iww in range(-1, 2):
                        a00 = a00 + bgr[ih+iwh, iw+iww, 0] * bgr[ih+iwh, iw+iww, 0]
                        a11 = a11 + bgr[ih+iwh, iw+iww, 1] * bgr[ih+iwh, iw+iww, 1]
                        a22 = a22 + bgr[ih+iwh, iw+iww, 2] * bgr[ih+iwh, iw+iww, 2]
                        a01 = a01 + bgr[ih+iwh, iw+iww, 0] * bgr[ih+iwh, iw+iww, 1]
                        a02 = a02 + bgr[ih+iwh, iw+iww, 0] * bgr[ih+iwh, iw+iww, 2]
                        a12 = a12 + bgr[ih+iwh, iw+iww, 1] * bgr[ih+iwh, iw+iww, 2]
                        a03 = a03 + bgr[ih+iwh, iw+iww, 0]
                        a13 = a13 + bgr[ih+iwh, iw+iww, 1]
                        a23 = a23 + bgr[ih+iwh, iw+iww, 2]


                m4x4[0][0] = a00
                m4x4[0][1] = a01
                m4x4[0][2] = a02
                m4x4[0][3] = a03

                m4x4[1][0] = a01
                m4x4[1][1] = a11
                m4x4[1][2] = a12
                m4x4[1][3] = a13

                m4x4[2][0] = a02
                m4x4[2][1] = a12
                m4x4[2][2] = a22
                m4x4[2][3] = a23

                m4x4[3][0] = a03
                m4x4[3][1] = a13
                m4x4[3][2] = a23
                m4x4[3][3] = 9.0


                for kh in range(4):
                    spa_h = kh * h * w + index_in_hw
                    for kw in range(4):
                        spa_w = kw * h * w + index_in_hw
                        ih_inds[k] = spa_h
                        iw_inds[k] = spa_w
                        values[k] = m4x4[kh][kw]
                        k = k + 1

    return k




