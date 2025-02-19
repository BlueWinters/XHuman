
import logging
import os
import cv2
import numpy as np
from .masking_sticker import MaskingSticker
from ....geometry import GeoFunction


class MaskingStickerAlignBoxStatic(MaskingSticker):
    """
    this class is only worked for masking image
    """
    NameEN = 'sticker_align_box_static'
    NameCN = '贴纸_检测框对齐_静态'  # only for masking image

    @staticmethod
    def benchmark():
        pass

    """
    """
    def __init__(self, sticker, box_tuple, *args, **kwargs):
        super(MaskingStickerAlignBoxStatic, self).__init__(*args, **kwargs)
        self.sticker = np.array(sticker, dtype=np.uint8)
        assert len(self.sticker.shape) == 3 and self.sticker.shape[2] == 4, self.sticker.shape  # H,W,4
        box_ori, box_fmt = box_tuple
        assert isinstance(box_ori, list) and len(box_ori) == 4, box_ori
        assert isinstance(box_fmt, list) and len(box_fmt) == 4, box_fmt
        self.box_ori = np.reshape(np.array(box_ori, dtype=np.int32), (-1,))
        self.box_fmt = np.reshape(np.array(box_fmt, dtype=np.int32), (-1,))

    def __str__(self):
        return '{}(sticker={}, box_ori={}, box_fmt={})'.format(
            self.NameEN, self.sticker.shape, self.box_ori, self.box_fmt)

    def inference(self, bgr, *args, **kwargs):
        raise NotImplementedError

    def inferenceOnMaskingImage(self, source_bgr, canvas_bgr, angle, box, landmark, **kwargs):
        if 'auto_rot' in kwargs and kwargs['auto_rot'] is True:
            angle_back = GeoFunction.rotateBack(angle)
            sticker = GeoFunction.rotateImage(self.sticker, angle_back)
            box = GeoFunction.rotateBoxes(box, angle, source_bgr.shape[0], source_bgr.shape[1])
            source_rot_bgr = GeoFunction.rotateImage(source_bgr, angle)
            ltrb = GeoFunction.rotateBoxes(self.box_fmt, angle_back, source_rot_bgr.shape[0], source_rot_bgr.shape[1])
        else:
            sticker = self.sticker
            ltrb = self.box_fmt
        assert np.count_nonzero(np.reshape(box, (-1,)) - self.box_ori) == 0, (box, self.box_ori)
        image_c = source_bgr[ltrb[1]:ltrb[3], ltrb[0]:ltrb[2], :]
        w_ori = ltrb[2] - ltrb[0]
        h_ori = ltrb[3] - ltrb[1]
        result_nd = sticker[:, :, :3]
        result_alpha = sticker[:, :, 3]
        result_nd_resize = cv2.resize(result_nd, (w_ori, h_ori))
        result_alpha_resize = cv2.resize(result_alpha, (w_ori, h_ori))
        result_alpha_resize = result_alpha_resize[:, :, np.newaxis] / 255.0
        image_c_cartoon = result_nd_resize * result_alpha_resize + (1 - result_alpha_resize) * image_c
        canvas_bgr[ltrb[1]:ltrb[3], ltrb[0]:ltrb[2], :] = image_c_cartoon
        return canvas_bgr

