
import cv2
import numpy as np
import scipy.sparse
from .ichol import ichol
from .conjugategradient import cg
from ...extension import funS as fs


def grid_coordinates(width, height, flatten=False):
    if flatten:
        x = np.tile(np.arange(width), height)
        y = np.repeat(np.arange(height), width)
    else:
        x = np.arange(width)
        y = np.arange(height)

        x, y = np.meshgrid(x, y)

    return x, y


def sparse_conv_matrix_with_offsets(width, height, kernel, dx, dy, knows):

    weights = np.asarray(kernel).flatten()
    count = len(weights)
    N = width * height

    knows = knows.astype(np.int32)
    n = np.sum(1 - knows)
    n = n.astype(np.int32) + 1

    i_inds = np.zeros(n * count, dtype=np.int32)
    j_inds = np.zeros(n * count, dtype=np.int32)
    values = np.zeros(n * count, dtype=np.float64)

    k = fs.cacuS(i_inds, j_inds, values, np.array(dx).astype(np.int32),
                   np.array(dy).astype(np.int32), np.array(weights), knows)

    A = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(N, N))

    return A

def genKnow(alpha):
    ksize = 11
    thre1 = 0.007
    thre2 = 0.99
    mask = cv2.GaussianBlur(alpha, (ksize, ksize), 0)
    mask2 = mask.copy()
    mask[mask2<thre1] = 1
    mask[mask2>thre2] = 1
    mask[(mask2>=thre1)&(mask2<=thre2)] = 0

    return mask

def estimate_foreground_cf(
    image,
    alpha,
    regularization=1e-5,
    rtol=1e-5,
    neighbors=[(-1, 0), (1, 0), (0, -1), (0, 1)],
    return_background=False,
    foreground_guess=None,
    background_guess=None,
    ichol_kwargs={},
    cg_kwargs={},
):




    h, w, d = image.shape

    assert alpha.shape == (h, w)
    n = w * h



    knows = genKnow(alpha)
    knows2 = knows.copy()
    index_h, index_w = np.where(knows2 == 0)
    index_in_hw = index_h * w + index_w
    index_in_hw2 = np.concatenate((index_in_hw, index_in_hw + n), axis=0)





    a = alpha.flatten()

    S = None
    for dx, dy in neighbors:
        D = sparse_conv_matrix_with_offsets(w, h, [1.0, -1.0], [0, dx], [0, dy], knows)
        S2 = D.T.dot(scipy.sparse.diags(regularization + np.abs(D.dot(a)))).dot(D)
        S2 = S2[index_in_hw, :]
        S2 = S2[:, index_in_hw]


        S = S2 if S is None else S + S2

        del D, S2

    V = scipy.sparse.bmat([[S, None],
                           [None, S]])

    del S

    U = scipy.sparse.bmat([[scipy.sparse.diags(a), scipy.sparse.diags(1 - a)]])

    T = U.T.dot(U)
    T = T[index_in_hw2, :]
    T = T[:, index_in_hw2]

    A = T + V

    A.sum_duplicates()

    del V

    # precondition = ichol(A, **ichol_kwargs)

    foreground = (image if foreground_guess is None else foreground_guess).copy()
    background = (image if background_guess is None else background_guess).copy()

    for channel in range(d):
        image_channel = image[:, :, channel].flatten()

        b = U.T.dot(image_channel)
        b = b[index_in_hw2]

        f0 = foreground[:, :, channel].flatten()
        b0 = background[:, :, channel].flatten()
        fb = np.concatenate([f0, b0])
        fb_init = fb[index_in_hw2]


        fb_iter = cg(A, b, x0=fb_init, rtol=rtol, atol=rtol)
        # fb_iter = cg(A, b, x0=fb_init, rtol=rtol, **cg_kwargs)
        fb[index_in_hw2] = fb_iter

        foreground[:, :, channel] = fb[:n].reshape(h, w)
        background[:, :, channel] = fb[n:].reshape(h, w)

    foreground = np.clip(foreground, 0, 1)
    background = np.clip(background, 0, 1)

    if return_background:
        return foreground, background

    return foreground













