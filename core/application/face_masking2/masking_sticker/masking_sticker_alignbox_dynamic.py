
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
    def __init__(self, sticker, *args, **kwargs):
        super(MaskingStickerAlignBoxDynamic, self).__init__(*args, **kwargs)
        self.sticker = np.array(sticker, dtype=np.uint8)
        assert len(self.sticker.shape) == 3, self.sticker.shape  # H,W,3 or H,w,4

    def __str__(self):
        return '{}(sticker={})'.format(self.NameEN, self.sticker.shape)

    def inference(self, source_bgr, canvas_bgr, box, *args, **kwargs):
        lft, top, rig, bot = box
        h = bot - top
        w = rig - lft
        if self.sticker.shape[2] == 3:
            canvas_bgr[top:bot, lft:rig, :] = cv2.resize(self.sticker, (w, h))
            return canvas_bgr
        if self.sticker.shape[2] == 4:
            resized_sticker = cv2.resize(self.sticker, (w, h))
            sticker_bgr = resized_sticker[:, :, :3]
            sticker_mask = resized_sticker[:, :, 3:4]
            part = source_bgr[top:bot, lft:rig, :]
            mask = sticker_mask.astype(np.float32) / 255.
            fusion = part * (1 - mask) + sticker_bgr * mask
            fusion_bgr = np.round(fusion).astype(np.uint8)
            canvas_bgr[top:bot, lft:rig, :] = fusion_bgr
            return canvas_bgr
        raise NotImplementedError

    def inferenceOnMaskingImage(self, source_bgr, canvas_bgr, box, *args, **kwargs):
        return self.inference(source_bgr, canvas_bgr, box)

    def inferenceOnMaskingVideo(self, source_bgr, canvas_bgr, box, *args, **kwargs):
        return self.inference(source_bgr, canvas_bgr, box)

