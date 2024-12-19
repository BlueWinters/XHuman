
import copy
import logging
import os
import numpy as np
import cv2
from .perlin2d import *
from .perlin3d import *



class LibImageFlow:
    """
    """
    @staticmethod
    def getResources():
        return []

    @staticmethod
    def benchmark():
        from core.utils.video import XVideoWriter
        num_frames = 16 * 3
        bgr = cv2.imread('benchmark/asset/application/flow/input.png')
        bgr_list = LibImageFlow.getFlowImages(bgr, num_frames=num_frames, k_speed=24)
        writer = XVideoWriter(dict(fps=16))
        writer.open('benchmark/asset/application/flow/output-k24.mp4')
        writer.dump(bgr_list)

    """
    """
    def __init__(self, *args, **kwargs):
        pass

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    """
    """
    @staticmethod
    def getFlowImages(bgr, num_frames, k_speed=16):
        h, w, c = bgr.shape
        size = max(h, w)
        size = size - size % 4
        noise_x = generate_fractal_noise_3d((num_frames, size, size), (1, 4, 4), 4, tileable=(True, False, False))
        noise_y = generate_fractal_noise_3d((num_frames, size, size), (1, 4, 4), 4, tileable=(True, False, False))
        bgr_list = []
        for n in range(num_frames):
            diff_x = noise_x[n, :h, :w].astype(np.float32)  # h, w
            diff_y = noise_y[n, :h, :w].astype(np.float32)  # h, w
            diff_x = cv2.resize(diff_x, (w, h), interpolation=cv2.INTER_LINEAR)
            diff_y = cv2.resize(diff_y, (w, h), interpolation=cv2.INTER_LINEAR)
            map_x = np.repeat(np.arange(w, dtype=np.float32)[None, :], h, axis=0) + k_speed * diff_x
            map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1) + k_speed * diff_y
            warped = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR)
            bgr_list.append(warped)
        return bgr_list
