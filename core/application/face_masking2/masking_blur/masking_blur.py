
import logging
import os
import cv2
import numpy as np
import functools
from ..helper.masking_helper import MaskingHelper


class MaskingBlur:
    """
    """
    @staticmethod
    def benchmark():
        pass

    @classmethod
    def parameterize(cls, *args, **kwargs):
        return functools.partial(cls, *args, **kwargs)

    """
    """
    def __init__(self, *args, **kwargs):
        self.fmt_w = kwargs.pop('fmt_w', 256)
        self.fmt_h = kwargs.pop('fmt_h', 256)

    def __del__(self):
        pass

    def inference(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def inferenceWithMask(self, source_bgr, canvas_bgr, box, mask):
        lft, top, rig, bot = box
        part = source_bgr[top:bot, lft:rig, ...]
        resized = cv2.resize(part, (self.fmt_w, self.fmt_h))
        blured = self.inference(resized)
        copy_bgr = np.copy(source_bgr)
        copy_bgr[top:bot, lft:rig, ...] = cv2.resize(blured, part.shape[:2][::-1])
        return MaskingHelper.workOnSelectedMask(canvas_bgr, copy_bgr, mask=mask)

    def inferenceOnMaskingImage(self, source_bgr, canvas_bgr, **kwargs):
        if 'mask_info' in kwargs:
            mask_info = kwargs.pop('mask_info', None)
            assert isinstance(mask_info, dict)
            box = mask_info['box']
            mask = mask_info['mask']
            return self.inferenceWithMask(source_bgr, canvas_bgr, box, mask)
        if 'box' in kwargs:
            h, w, c = canvas_bgr.shape
            box = kwargs.pop('box')
            lft, top, rig, bot = box
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            mask[top:bot, lft:rig, ...] = 255
            return self.inferenceWithMask(source_bgr, canvas_bgr, box, mask)
        raise NotImplementedError(kwargs.keys())

    def inferenceOnMaskingVideo(self, source_bgr, canvas_bgr, **kwargs):
        if 'mask_info' in kwargs:
            mask_info = kwargs.pop('mask_info', None)
            assert isinstance(mask_info, dict)
            box = mask_info['box']
            mask = mask_info['mask']
            return self.inferenceWithMask(source_bgr, canvas_bgr, box, mask)
        if 'box' in kwargs:
            h, w, c = canvas_bgr.shape
            box = kwargs.pop('box')
            lft, top, rig, bot = box
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            mask[top:bot, lft:rig, ...] = 255
            return self.inferenceWithMask(source_bgr, canvas_bgr, box, mask)
        raise NotImplementedError(kwargs.keys())
