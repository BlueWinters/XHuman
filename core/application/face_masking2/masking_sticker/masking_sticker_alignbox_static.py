
import logging
import os
import cv2
import numpy as np
import skimage
from .masking_sticker import MaskingSticker
from ..scanning.scanning_video import InfoVideo_PersonPreview
from ..helper.masking_helper import MaskingHelper
from ....geometry import GeoFunction


class MaskingStickerAlignBoxStatic(MaskingSticker):
    """
    this class is only worked for masking image
    """
    NameEN = 'sticker_align_box_static'
    NameCN = '贴纸_检测框对齐_静态'

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
        self.box_ori = np.reshape(np.array(box_ori, dtype=np.int32), (4,))
        self.box_fmt = np.reshape(np.array(box_fmt, dtype=np.int32), (4,))

    def __str__(self):
        return '{}(sticker={}, box_ori={}, box_fmt={})'.format(
            self.NameEN, self.sticker.shape, self.box_ori, self.box_fmt)

    def getAlignPoints(self, bgr, box, points=None):
        assert len(box) == 4, len(box)
        assert isinstance(points, np.ndarray), points
        assert len(points.shape) == 2 and points.shape[1] == 2, points.shape  # 5,2 or 68,2 or 2,2
        assert points.shape[0] == 5 or points.shape[0] == 2, points.shape
        dst_pts = points[:2, :]
        return dst_pts

    def warpSticker(self, source_bgr, src_pts, dst_pts):
        h, w, c = source_bgr.shape
        transform = skimage.transform.SimilarityTransform()
        transform.estimate(src_pts, dst_pts)
        param = dict(order=1, mode='constant', cval=0, output_shape=(h, w))
        sticker_warped = skimage.transform.warp(self.sticker.astype(np.float32), transform.inverse, **param)
        sticker_warped_bgr, sticker_warped_alpha = sticker_warped[:, :, :3], sticker_warped[:, :, 3]
        return sticker_warped_bgr, sticker_warped_alpha

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
        w_ori = ltrb[2] - ltrb[0]
        h_ori = ltrb[3] - ltrb[1]
        h, w, c = source_bgr.shape
        canvas_alpha = np.zeros(shape=(h, w), dtype=np.uint8)
        canvas_alpha[ltrb[1]:ltrb[3], ltrb[0]:ltrb[2]] = cv2.resize(sticker[:, :, 3], (w_ori, h_ori))
        canvas_sticker = np.zeros(shape=(h, w, c), dtype=np.uint8)
        canvas_sticker[ltrb[1]:ltrb[3], ltrb[0]:ltrb[2], :] = cv2.resize(sticker[:, :, :3], (w_ori, h_ori))
        multi = canvas_alpha[:, :, np.newaxis] / 255.0
        fusion_bgr = canvas_sticker * multi + (1 - multi) * canvas_bgr
        return np.round(fusion_bgr).astype(np.uint8)

    def inferenceOnMaskingVideo(self, source_bgr, canvas_bgr, face_box, face_points_xy, face_points_score, **kwargs):
        preview = kwargs['preview']
        assert isinstance(preview, InfoVideo_PersonPreview), preview
        if face_points_score[2] > 0.5 and face_points_score[1] > 0.5:
            sticker_warped_bgr, sticker_warped_alpha = self.warpSticker(
                source_bgr, preview.face_key_points_xy[:2, :2], face_points_xy[:2, :2])
            fusion_bgr = MaskingHelper.workOnSelectedMask(
                canvas_bgr, sticker_warped_bgr, sticker_warped_alpha, mask_blur_k=None)
            return fusion_bgr
        else:
            return canvas_bgr

