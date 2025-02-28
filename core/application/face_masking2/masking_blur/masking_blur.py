
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

    def inferenceOnMaskingImage(self, source_bgr, canvas_bgr, angle, box, mask_info, **kwargs):
        if isinstance(mask_info, dict):
            assert 'mask' in mask_info and 'box' in mask_info, mask_info.keys()
            mask = mask_info['mask']
            lft, top, rig, bot = mask_info['box']
        else:
            mask = mask_info
            st_x, st_y, st_w, st_h = cv2.boundingRect(mask)
            lft, top, rig, bot = st_x, st_y, st_x + st_w, st_y + st_h
        part = source_bgr[top:bot, lft:rig, ...]
        resized = cv2.resize(part, (self.fmt_w, self.fmt_h))
        blured = self.inference(resized)
        copy_bgr = np.copy(source_bgr)
        copy_bgr[top:bot, lft:rig, ...] = cv2.resize(blured, part.shape[:2][::-1])
        return MaskingHelper.workOnSelectedMask(canvas_bgr, copy_bgr, mask=mask)

    def inferenceOnMaskingVideo(self, source_bgr, canvas_bgr, face_box, face_points_xy, face_points_score, **kwargs):
        mask_info = kwargs.pop('mask_info', None)
        if isinstance(mask_info, dict):
            assert 'mask' in mask_info and 'box' in mask_info, mask_info.keys()
            mask = mask_info['mask']
            lft, top, rig, bot = mask_info['box']
            part = source_bgr[top:bot, lft:rig, ...]
            resized = cv2.resize(part, (self.fmt_w, self.fmt_h))
            blured = self.inference(resized)
            copy_bgr = np.copy(source_bgr)
            copy_bgr[top:bot, lft:rig, ...] = cv2.resize(blured, part.shape[:2][::-1])
            return MaskingHelper.workOnSelectedMask(canvas_bgr, copy_bgr, mask=mask)
        else:
            return canvas_bgr
