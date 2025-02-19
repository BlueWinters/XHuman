
import logging
import os
import cv2
import numpy as np
import numba
from .masking_blur import MaskingBlur


class MaskingBlurWater(MaskingBlur):
    """
    """
    NameEN = 'blur_water'
    NameCN = '水纹模糊'

    @staticmethod
    def benchmark():
        pass

    """
    """
    def __init__(self, a=2., b=8., align_type='head', *args, **kwargs):
        super(MaskingBlurWater, self).__init__(*args, **kwargs)
        self.a = a
        self.b = b
        self.align_type = align_type
        self.xy_map = None

    @staticmethod
    def getBlurWaterParameters(bgr, a=2.0, b=8.0):
        # a = 2.0  # rotation degree
        # b = 8.0  # each water length
        h, w, c = bgr.shape
        center_x = (w - 1) / 2.0
        center_y = (h - 1) / 2.0
        xx = np.arange(w)
        yy = np.arange(h)
        x_mask = np.repeat(xx[None, :], h, 0)
        y_mask = np.repeat(yy[:, None], w, 1)
        xx_dif = x_mask - center_x
        yy_dif = center_y - y_mask
        theta = np.arctan2(yy_dif, xx_dif)
        r = np.sqrt(xx_dif * xx_dif + yy_dif * yy_dif)
        r1 = r + a * w * 0.01 * np.sin(b * 0.1 * r)
        x_new = r1 * np.cos(theta) + center_x
        y_new = center_y - r1 * np.sin(theta)
        int_x = np.floor(x_new)
        int_x = int_x.astype(int)
        int_y = np.floor(y_new)
        int_y = int_y.astype(int)
        return int_x, int_y, x_new, y_new

    def __str__(self):
        return '{}(a={:.2f}, b={:.2f} align_type={})'.format(self.NameEN, self.a, self.b, self.align_type)

    """
    """
    @staticmethod
    @numba.jit(nopython=True, nogil=True, parallel=True)
    def mapCoordinatesWithJit(bgr, bgr_copy, int_x, int_y, x_new, y_new):
        h, w, c = bgr.shape
        for ii in numba.prange(h):
            for jj in numba.prange(w):
                new_xx = int_x[ii, jj]
                new_yy = int_y[ii, jj]
                if x_new[ii, jj] < 0 or x_new[ii, jj] > w - 1:
                    continue
                if y_new[ii, jj] < 0 or y_new[ii, jj] > h - 1:
                    continue
                bgr_copy[ii, jj, :] = bgr[new_yy, new_xx, :]
        return bgr_copy

    def inference(self, bgr):
        if self.xy_map is None:
            self.xy_map = self.getBlurWaterParameters(bgr, self.a, self.b)  # update x,y mapping coordinates
        assert len(self.xy_map) == 4, len(self.xy_map)
        int_x, int_y, x_new, y_new = self.xy_map
        return self.mapCoordinatesWithJit(bgr, np.copy(bgr), int_x, int_y, x_new, y_new)

