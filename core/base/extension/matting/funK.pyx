cimport cython



@cython.infer_types(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)



def cacuK(int[:] ih_inds,
          int[:] iw_inds,
          double[:] values,
          double[:, :, :] bgr,
          int[:, :] knows):

    cdef int ih, iw, kh, kw, ic, k, index_in_hw, index_h, index_w, spa_h, spa_w

    cdef Py_ssize_t h = bgr.shape[0]
    cdef Py_ssize_t w = bgr.shape[1]
    k = 0


    for ih in range(1, h-1):
        for iw in range(1, w-1):
            if knows[ih, iw] == 1:
                continue
            else:
                index_in_hw = ih * w + iw
                for kh in range(-1, 2):
                    index_h = ih + kh
                    for kw in range(-1, 2):
                        index_w = iw + kw
                        spa_h = index_h * w + index_w
                        # for ic in range(3):
                        #     spa_w = ic * h * w + index_in_hw
                        #     ih_inds[k] = spa_h
                        #     iw_inds[k] = spa_w
                        #     values[k] = bgr[index_h, index_w, ic]
                        #     k = k + 1

                        ih_inds[k] = spa_h
                        iw_inds[k] = 0 * h * w + index_in_hw
                        values[k] = bgr[index_h, index_w, 0]
                        k = k + 1

                        ih_inds[k] = spa_h
                        iw_inds[k] = 1 * h * w + index_in_hw
                        values[k] = bgr[index_h, index_w, 1]
                        k = k + 1

                        ih_inds[k] = spa_h
                        iw_inds[k] = 2 * h * w + index_in_hw
                        values[k] = bgr[index_h, index_w, 2]
                        k = k + 1

                        ih_inds[k] = spa_h
                        iw_inds[k] = 3 * h * w + index_in_hw
                        values[k] = 1.0
                        k = k + 1

    return k




