
import numpy as np
import numba


def elementwise_multiply(fine_result, hair_mask, combine_mask, coarse_result, parallel=True):
    if parallel is True:
        return elementwise_multiply_jit(fine_result, hair_mask, combine_mask, coarse_result)
    return fine_result * (1 - hair_mask) + hair_mask * \
        (combine_mask * (fine_result * 0.5 + 0.5 * coarse_result) + (1 - combine_mask) * coarse_result)

@numba.jit(nopython=True, nogil=True, parallel=True)
def elementwise_multiply_jit(fine_result, hair_mask, combine_mask, coarse_result):
    result = np.empty_like(fine_result)
    # H, W, C = fine_result.shape
    for i in numba.prange(fine_result.shape[0]):
        for j in numba.prange(fine_result.shape[1]):
            for c in numba.prange(fine_result.shape[2]):
                result[i, j, c] = fine_result[i, j, c] * (1 - hair_mask[i, j, 0]) + \
                    hair_mask[i, j, 0] * (combine_mask[i, j, 0] * (fine_result[i, j, c] * 0.5 +
                        0.5 * coarse_result[i, j, c]) +(1 - combine_mask[i, j, 0]) * coarse_result[i, j, c])
    return result

def alpha_multiply(matrix_a, matrix_b, matrix_c, parallel=True):
    if parallel is True:
        return alpha_multiply_jit(matrix_a, matrix_b, matrix_c)
    return matrix_a * (1 - matrix_b) + matrix_c * matrix_b


@numba.jit(nopython=True, nogil=True, parallel=True)
def alpha_multiply_jit(matrix_a, matrix_b, matrix_c):
    result = np.empty_like(matrix_a)
    H, W, C = matrix_a.shape
    for i in numba.prange(H):
        for j in numba.prange(W):
            for c in numba.prange(C):
                result[i, j, c] = matrix_a[i, j, c] * (1 - matrix_b[i, j, 0]) + matrix_c[i, j, c] * matrix_b[i, j, 0]
    return result