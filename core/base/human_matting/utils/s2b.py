
import cv2
import numpy as np
import scipy.sparse
from .conjugategradient import cg
from ...extension import funK as fk
from ...extension import funP as fp
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





def fun(bgr, alpha, knows, epsilon=0.0000001):

    h, w = np.shape(alpha)


    knows = knows.astype(np.int32)
    n = np.sum(1 - knows)
    n = n.astype(np.int32) + 1

    ih_inds = np.zeros(n * 36, dtype=np.int32)
    iw_inds = np.zeros(n * 36, dtype=np.int32)
    values  = np.zeros(n * 36, dtype=np.float64)

    knows = knows.astype(np.int32)

    k = fk.cacuK(ih_inds, iw_inds, values, bgr, knows)

    A = scipy.sparse.csr_matrix((values, (ih_inds, iw_inds)), shape=(h*w, 4*h*w))

    return A



def fun2(bgr, alpha, knows, epsilon=0.0000001):
    h, w = np.shape(alpha)

    knows = knows.astype(np.int32)
    n = np.sum(1 - knows)
    n = n.astype(np.int32) + 1

    ih_inds = np.zeros(n * 16, dtype=np.int32)
    iw_inds = np.zeros(n * 16, dtype=np.int32)
    values  = np.zeros(n * 16, dtype=np.float64)


    k = fp.cacuP(ih_inds, iw_inds, values, bgr, knows)

    B = scipy.sparse.csr_matrix((values, (ih_inds, iw_inds)), shape=(4 * h * w, 4 * h * w))
    return B



def boxfilter(I,rad):
    N = cv2.blur(I, rad)
    return N


def guideFilter2(I, P, rads, eps):

    hgt = I.shape[0]
    wid = I.shape[1]
    N = np.ones(np.shape(I))

    meanI = boxfilter(I.astype(np.float32),rads)
    meanP = boxfilter(P.astype(np.float32),rads)
    corrIP = boxfilter(I*P,rads)
    corrI = boxfilter(I * I, rads)


    varI = corrI - meanI*meanI
    covIP = corrIP - meanI * meanP

    a = covIP / (varI+eps)
    b = meanP - a*meanI

    meanA = boxfilter(a,rads)/N
    meanB = boxfilter(b,rads)/N

    res = meanA * I + meanB
    return res, meanA, meanB




def fun5(bgr, alpha, knows, epsilon=0.00001):
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    h, w, c = np.shape(bgr)
    a = alpha.flatten()
    n = h*w


    knows2 = knows.copy()
    index_h, index_w = np.where(knows2 == 0)
    index_in_hw = index_h * w + index_w
    index_in_hw2 = np.concatenate((index_in_hw, index_in_hw+n), axis=0)
    index_in_hw4 = np.concatenate((index_in_hw2, index_in_hw2 + 2*n), axis=0)



    S = None
    for dx, dy in neighbors:
        D = sparse_conv_matrix_with_offsets(w, h, [1.0, -1.0], [0, dx], [0, dy], knows)
        S2 = D.T.dot(scipy.sparse.diags(epsilon + np.abs(D.dot(a)))).dot(D)
        S2 = S2[index_in_hw, :]
        S2 = S2[:, index_in_hw]
        S = S2 if S is None else S + S2
        del D, S2


    V = scipy.sparse.bmat([[S, None, None, None],
                           [None, S, None, None],
                           [None, None, S, None],
                           [None, None, None, S]])

    del S


    K = fun(bgr, alpha, knows)
    P = fun2(bgr, alpha, knows)
    P = P[index_in_hw4, :]
    P = P[:, index_in_hw4]



    A = P + V

    del V
    del P


    ############################
    gray = bgr[:, :, 0] * 0.114 + bgr[:, :, 1] * 0.587 + bgr[:, :, 2] * 0.299
    eps = 0.01
    winSize = (3, 3)
    gi, ga, gb = guideFilter2(gray, alpha, winSize, eps)
    ga = np.reshape(ga, (-1))
    gb = np.reshape(gb, (-1))
    iab = ga * 0.114
    iag = ga * 0.587
    iar = ga * 0.299
    fb = np.concatenate((iab, iag, iar, gb), axis=0)
    ############################

    rtol = 1e-5
    atol = 1e-5


    b = K.T.dot(a)
    b = b[index_in_hw4]

    del K


    fb_init = fb[index_in_hw4]


    fb_iter = cg(A, b, x0=fb_init, atol=atol, rtol=rtol)


    fb[index_in_hw4] = fb_iter

    a1 = fb[:n]
    a2 = fb[n:2*n]
    a3 = fb[2*n:3*n]
    d  = fb[3*n::]

    return a1, a2, a3, d

def genKnow(alpha):
    ksize = 11
    thre1 = 0.01
    thre2 = 0.99
    mask = cv2.GaussianBlur(alpha, (ksize, ksize), 0)
    mask2 = mask.copy()
    mask[mask2<thre1] = 1
    mask[mask2>thre2] = 1
    mask[(mask2>=thre1)&(mask2<=thre2)] = 0

    return mask

def _interp_coef(coef, hs ,ws, hb, wb):
    coef = np.reshape(coef, (hs, ws))
    coef = cv2.resize(coef, (wb, hb), interpolation=cv2.INTER_LINEAR)
    return coef

def finer(images, alphas, imageb):
    knows = genKnow(alphas)
    hs, ws = np.shape(images)[0:2]
    hb, wb = np.shape(imageb)[0:2]
    a1, a2, a3, d = fun5(images, alphas, knows)
    a1_b = _interp_coef(a1, hs, ws, hb, wb)
    a2_b = _interp_coef(a2, hs, ws, hb, wb)
    a3_b = _interp_coef(a3, hs, ws, hb, wb)
    d_b  = _interp_coef(d , hs, ws, hb, wb)
    alphab = a1_b * imageb[:, :, 0] + a2_b * imageb[:, :, 1] + a3_b * imageb[:, :, 2] + d_b
    alphab = alphab * 1.02
    alphab = np.clip(alphab, 0, 1)
    return alphab

































