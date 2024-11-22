cimport cython



@cython.infer_types(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)



def cacuS(int[:] ih_inds,
          int[:] iw_inds,
          double[:] values,
          int[:] dx,
          int[:] dy,
          double[:] weights,
          int[:, :] knows):

    cdef int ih, iw, kh, kw, ic, k, index_in_hw, index_h, index_w, spa_h, spa_w
    cdef int step_h, step_w
    cdef Py_ssize_t h = knows.shape[0]
    cdef Py_ssize_t w = knows.shape[1]
    cdef double weight
    k = 0


    for ic in range(2):
        weight = weights[ic]
        step_h = dy[ic]
        step_w = dx[ic]
        for ih in range(1, h-1):
            index_h = ih + step_h
            for iw in range(1, w-1):
                index_w = iw + step_w
                if knows[ih, iw] == 1:
                    continue
                else:
                    spa_h = ih * w + iw
                    spa_w = index_h * w + index_w
                    ih_inds[k] = spa_h
                    iw_inds[k] = spa_w
                    values[k] = weight
                    k = k + 1

    return k




