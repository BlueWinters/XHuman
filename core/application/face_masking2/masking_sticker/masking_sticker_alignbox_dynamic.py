
import logging
import os
import cv2
import numpy as np
from .masking_sticker import MaskingSticker
from ....geometry import Rectangle


class MaskingStickerAlignBoxDynamic(MaskingSticker):
    """
    """
    NameEN = 'sticker_align_box_dynamic'
    NameCN = '贴纸_检测框对齐_动态'

    @staticmethod
    def benchmark():
        pass

    """
    """
    def __init__(self, sticker, box_tuple, *args, **kwargs):
        super(MaskingStickerAlignBoxDynamic, self).__init__(*args, **kwargs)
        self.sticker = np.array(sticker, dtype=np.uint8)
        assert len(self.sticker.shape) == 3 and self.sticker.shape[2] == 4, self.sticker.shape  # H,W,4
        # box tuple
        self.box_src, self.box_fmt = None, None
        self.sticker_bounding_rect = None
        if isinstance(box_tuple, (list, tuple)):
            box_src, box_fmt = box_tuple
            assert isinstance(box_src, (list, tuple)) and len(box_src) == 4, box_src
            assert isinstance(box_fmt, (list, tuple)) and len(box_fmt) == 4, box_fmt
            self.box_src = np.reshape(np.array(box_src, dtype=np.int32), (-1,))
            self.box_fmt = np.reshape(np.array(box_fmt, dtype=np.int32), (-1,))
            self.sticker_bounding_rect = None

    def __str__(self):
        return '{}(sticker={}, box_src={}, box_fmt={})'.format(
            self.NameEN, self.sticker.shape, self.box_src, self.box_fmt)

    @staticmethod
    def remapBBox(box_src, box_fmt, box_cur):
        if np.count_nonzero(box_cur - box_src) == 0:
            return box_fmt
        src_lft, src_top, src_rig, src_bot = box_src
        fmt_lft, fmt_top, fmt_rig, fmt_bot = box_fmt
        Rectangle.assertInside(Rectangle(np.array(box_src, dtype=np.int32)), Rectangle(np.array(box_fmt, dtype=np.int32)))
        pix_lft = src_lft - fmt_lft
        pix_rig = fmt_rig - src_rig
        pix_top = src_top - fmt_top
        pix_bot = fmt_bot - src_bot
        src_h = src_bot - src_top
        src_w = src_rig - src_lft
        ratio_lft = float(pix_lft / src_w)
        ratio_rig = float(pix_rig / src_w)
        ratio_top = float(pix_top / src_h)
        ratio_bot = float(pix_bot / src_h)
        lft, top, rig, bot = Rectangle(np.array(box_cur, dtype=np.float32)).expand4(
            ratio_lft, ratio_top, ratio_rig, ratio_bot).toSquare().decouple()
        return lft, top, rig, bot

    def inference(self, bgr, *args, **kwargs):
        raise NotImplementedError

    def refineSticker(self, dst_h, dst_w):
        if self.sticker_bounding_rect is None:
            alpha = self.sticker[:, :, 3]
            alpha[alpha < 125] = 0
            self.sticker_bounding_rect = cv2.boundingRect(alpha)
        st_x, st_y, st_w, st_h = self.sticker_bounding_rect
        if st_w == 0 and st_h == 0:
            return cv2.resize(self.sticker, (dst_w, dst_h))
        sticker_image = self.sticker[st_y:st_y + st_h, st_x:st_x + st_w, ...]
        resized_sticker = cv2.resize(sticker_image, (dst_w, dst_h))
        return resized_sticker

    def inferenceOnMaskingImage(self, source_bgr, canvas_bgr, angle, box, landmark, **kwargs):
        h, w, c = source_bgr.shape
        box_remap = self.remapBBox(self.box_src, self.box_fmt, box)
        lft, top, rig, bot = Rectangle(np.array(box_remap, dtype=np.int32)).clip(0, 0, w, h).decouple()
        nh = bot - top
        nw = rig - lft
        resized_sticker = self.refineSticker(nh, nw)
        sticker_bgr = resized_sticker[:, :, :3]
        sticker_mask = resized_sticker[:, :, 3:4]
        part = source_bgr[top:bot, lft:rig, :]
        mask = sticker_mask.astype(np.float32) / 255.
        fusion = part * (1 - mask) + sticker_bgr * mask
        fusion_bgr = np.round(fusion).astype(np.uint8)
        canvas_bgr[top:bot, lft:rig, :] = fusion_bgr
        return canvas_bgr

    def inferenceOnMaskingVideo(self, *args, **kwargs):
        raise NotImplementedError
