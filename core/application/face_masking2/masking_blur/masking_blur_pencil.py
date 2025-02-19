
import logging
import os
import cv2
import numpy as np
import numba
from .masking_blur_diffusion import MaskingBlurDiffusion


class MaskingBlurPencil(MaskingBlurDiffusion):
    """
    """
    NameEN = 'blur_pencil'
    NameCN = '画笔模糊'

    @staticmethod
    def benchmark():
        pass

    """
    """
    def __init__(self, *args, **kwargs):
        super(MaskingBlurPencil, self).__init__(*args, **kwargs)
        self.pre_blur_kernel = 5
        self.post_blur_kernel = 5
