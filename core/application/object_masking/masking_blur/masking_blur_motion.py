
import logging
import os
import cv2
import numpy as np
from .masking_blur import MaskingBlur
from ..helper.masking_helper import MaskingHelper


class MaskingBlurMotion(MaskingBlur):
    """
    """
    NameEN = 'blur_motion'
    NameCN = '运动模糊'

    @staticmethod
    def benchmark():
        pass

    """
    """
    def __init__(self, kernel=15, align_type='head', *args, **kwargs):
        super(MaskingBlurMotion, self).__init__(*args, **kwargs)
        kernel = kernel if kernel % 2 == 1 else (kernel + 1)  # should be odd
        assert 0 < kernel <= 64, kernel
        self.kernel_size = kernel
        self.kernel_image = self.getBlurMotionParameters(self.kernel_size)
        self.align_type = align_type

    @staticmethod
    def getBlurMotionParameters(kernel_size, x=16, y=16):
        kernel = kernel_size
        c = int(kernel / 2)
        blur_kernel = np.zeros((kernel, kernel), dtype=np.uint8)
        blur_kernel = cv2.line(blur_kernel, (c + x, c + y), (c, c), (1,), 1)
        blur_kernel = blur_kernel / int(np.count_nonzero(blur_kernel))
        return blur_kernel

    def __str__(self):
        return '{}(kernel_size={}, align_type={})'.format(self.NameEN, self.kernel_size, self.align_type)

    """
    """
    def inference(self, bgr, format_size=256):
        format_bgr, padding = MaskingHelper.formatSizeWithPaddingForward(bgr, format_size, format_size)
        blured_bgr = cv2.filter2D(format_bgr, ddepth=-1, kernel=self.kernel_image)
        reformat_blur_bgr = MaskingHelper.formatSizeWithPaddingBackward(bgr, blured_bgr, padding)
        return reformat_blur_bgr

