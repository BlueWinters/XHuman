
import logging
import os
import cv2
import numpy as np
from .masking_blur import MaskingBlur
from ..helper.masking_helper import MaskingHelper


class MaskingBlurGaussian(MaskingBlur):
    """
    """
    NameEN = 'blur_gaussian'
    NameCN = '高斯模糊'

    @staticmethod
    def benchmark():
        pass

    """
    """
    def __init__(self, kernel=25, align_type='head', *args, **kwargs):
        super(MaskingBlurGaussian, self).__init__(*args, **kwargs)
        kernel = kernel if kernel % 2 == 1 else (kernel + 1)  # should be odd
        assert 0 < kernel <= 64, kernel
        self.kernel = kernel
        self.align_type = align_type

    def __str__(self):
        return '{}(kernel={}, align_type={})'.format(self.NameEN, self.kernel, self.align_type)

    """
    """
    def inference(self, bgr, format_size=256):
        k = self.kernel
        format_bgr, padding = MaskingHelper.formatSizeWithPaddingForward(bgr, format_size, format_size)
        blured_bgr = cv2.GaussianBlur(format_bgr, (k, k), sigmaX=k // 2, sigmaY=k // 2)
        reformat_blur_bgr = MaskingHelper.formatSizeWithPaddingBackward(bgr, blured_bgr, padding)
        return reformat_blur_bgr

