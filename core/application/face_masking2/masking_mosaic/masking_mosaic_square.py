
import logging
import os
import cv2
import numpy as np
from .masking_mosaic import MaskingMosaic
from ..helper.masking_helper import MaskingHelper


class MaskingMosaicSquare(MaskingMosaic):
    """
    """
    NameEN = 'mosaic_square'
    NameCN = '正方形马赛克'

    @staticmethod
    def benchmark():
        pass

    """
    """
    def __init__(self, num_pixels=48, align_type='head', *args, **kwargs):
        super(MaskingMosaicSquare, self).__init__(*args, **kwargs)
        assert num_pixels > 0, num_pixels
        self.nh = num_pixels
        self.nw = num_pixels
        self.align_type = align_type

    def __str__(self):
        return '{}(height={}, width={}, align_type={})'.format(self.NameEN, self.nh, self.nw, self.align_type)

    """
    """
    def inference(self, bgr, format_size=128, **kwargs):
        h, w, c = bgr.shape
        format_bgr, padding = MaskingHelper.formatSizeWithPaddingForward(bgr, format_size, format_size)
        sub = cv2.resize(format_bgr, (self.nw, self.nh), interpolation=cv2.INTER_LINEAR)
        up = cv2.resize(sub, (w, h), interpolation=cv2.INTER_NEAREST)
        reformat_mosaic_bgr = MaskingHelper.formatSizeWithPaddingBackward(bgr, up, padding)
        return reformat_mosaic_bgr


