
import logging
import os
import cv2
import numpy as np
import numba
from .masking_blur import MaskingBlur


class MaskingBlurDiffusion(MaskingBlur):
    """
    """
    NameEN = 'blur_diffusion'
    NameCN = '扩散模糊'

    @staticmethod
    def benchmark():
        pass

    """
    """
    def __init__(self, k_neigh=11, align_type='head', *args, **kwargs):
        super(MaskingBlurDiffusion, self).__init__(*args, **kwargs)
        self.k_neigh = k_neigh
        self.pre_blur_kernel = 0
        self.post_blur_kernel = 0
        self.align_type = align_type

    def __str__(self):
        return '{}(k_neigh={}, pre_blur_kernel={}, post_blur_kernel={} align_type={})'.format(
            self.NameEN, self.k_neigh, self.pre_blur_kernel, self.post_blur_kernel, self.align_type)

    """
    """
    @staticmethod
    def doBlur(bgr, k):
        blured_bgr = cv2.GaussianBlur(bgr, (k, k), k // 2, k // 2)
        return blured_bgr

    @staticmethod
    @numba.jit(nopython=True, nogil=True, parallel=True)
    def mapCoordinatesWithJit(bgr, bgr_copy, k_neigh):
        h, w, c = bgr.shape
        noise = np.random.rand(h, w, 2)
        for hj in numba.prange(k_neigh, h - k_neigh, 1):
            for wi in numba.prange(k_neigh, w - k_neigh, 1):
                jj = int((noise[hj, wi, 0] - 0.5) * (k_neigh * 2 - 1))
                ii = int((noise[hj, wi, 1] - 0.5) * (k_neigh * 2 - 1))
                hh = (hj + jj) % h
                ww = (wi + ii) % w
                bgr_copy[hj, wi, :] = bgr[hh, ww, :]
        return bgr_copy

    @staticmethod
    def mapCoordinates(bgr, bgr_copy, k_neigh):
        h, w, c = bgr.shape
        noise = np.random.rand(h, w, 2)
        for hj in range(k_neigh, h - k_neigh, 1):
            for wi in range(k_neigh, w - k_neigh, 1):
                jj = int((noise[hj, wi, 0] - 0.5) * (k_neigh * 2 - 1))
                ii = int((noise[hj, wi, 1] - 0.5) * (k_neigh * 2 - 1))
                hh = (hj + jj) % h
                ww = (wi + ii) % w
                bgr_copy[hj, wi, :] = bgr[hh, ww, :]
        return bgr_copy

    def inference(self, bgr):
        if self.pre_blur_kernel > 0:
            blur_bgr = self.doBlur(bgr, self.pre_blur_kernel)
        else:
            blur_bgr = np.copy(bgr)
        result_bgr = MaskingBlurDiffusion.mapCoordinates(blur_bgr, np.copy(blur_bgr), self.k_neigh)
        if self.post_blur_kernel > 0:
            result_bgr = self.doBlur(result_bgr, self.post_blur_kernel)
        return result_bgr

