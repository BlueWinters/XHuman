
import logging
import os
import cv2
import numpy as np
import skimage
from .masking_sticker import MaskingSticker
from ..helper.boundingbox import BoundingBox
from ....geometry import GeoFunction


class MaskingStickerCartoon(MaskingSticker):
    """
    """
    NameEN = 'sticker_cartoon'
    NameCN = '贴纸_自定义卡通'

    @staticmethod
    def benchmark():
        pass

    """
    """
    def __init__(self, sticker, box_tuple, *args, **kwargs):
        super(MaskingStickerCartoon, self).__init__(*args, **kwargs)
        self.sticker = np.array(sticker, dtype=np.uint8)
        assert len(self.sticker.shape) == 3 and self.sticker.shape[2] == 4, self.sticker.shape  # H,W,4
        box_ori, box_fmt = box_tuple
        self.box_ori = np.reshape(np.array(box_ori, dtype=np.int32), (4,))
        self.box_fmt = np.reshape(np.array(box_fmt, dtype=np.int32), (4,))

    def __str__(self):
        return '{}(sticker={}, box_ori={}, box_fmt={})'.format(
            self.NameEN, self.sticker.shape, self.box_ori, self.box_fmt)

    def refineStickerKeyPoints(self, face_key_points_xy):
        if hasattr(self, 'eyes_key_points') is False:
            h, w, c = self.sticker.shape
            lft, top, rig, bot = self.box_fmt
            sw = rig - lft
            sh = bot - top
            eyes_key_points = np.stack([face_key_points_xy[2, :], face_key_points_xy[1, :]], axis=0)  # 2,2
            eyes_key_points[:, 0] = np.round((eyes_key_points[:, 0] - lft) / sw * w).astype(np.int32)
            eyes_key_points[:, 1] = np.round((eyes_key_points[:, 1] - top) / sh * w).astype(np.int32)
            self.eyes_key_points = eyes_key_points
        return self.eyes_key_points

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

    def inferenceOnMaskingImage(self, source_bgr, canvas_bgr, **kwargs):
        angle = kwargs['angle']
        # box = kwargs['box']
        auto_rot = kwargs['auto_rot']
        if auto_rot is True:
            angle_back = GeoFunction.rotateBack(angle)
            sticker = GeoFunction.rotateImage(self.sticker, angle_back)
            # box = GeoFunction.rotateBoxes(box, angle, source_bgr.shape[0], source_bgr.shape[1])
            source_rot_bgr = GeoFunction.rotateImage(source_bgr, angle)
            ltrb = GeoFunction.rotateBoxes(self.box_fmt, angle_back, source_rot_bgr.shape[0], source_rot_bgr.shape[1])
        else:
            sticker = self.sticker
            ltrb = self.box_fmt
        # assert np.count_nonzero(np.reshape(box, (-1,)) - self.box_ori) == 0, (box, self.box_ori)
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

    def inferenceOnMaskingVideo(self, source_bgr, canvas_bgr, **kwargs):
        # if face_points_score[2] > 0.5 and face_points_score[1] > 0.5:
        #     preview = kwargs['preview']
        #     assert isinstance(preview, InfoVideo_PersonPreview), preview
        #     points_eyes_sticker = self.refineStickerKeyPoints(preview.face_key_points_xy)
        #     points_eyes_current = np.stack([face_points_xy[2, :], face_points_xy[1, :]], axis=0)
        #     sticker_warped_bgr, sticker_warped_alpha = self.warpSticker(
        #         source_bgr, points_eyes_sticker, points_eyes_current)
        #     fusion_bgr = MaskingHelper.workOnSelectedMask(
        #         canvas_bgr, sticker_warped_bgr, sticker_warped_alpha, mask_blur_k=None)
        #     return fusion_bgr
        # else:
        #     return canvas_bgr

        face_box = kwargs.pop('face_box')
        h, w, c = source_bgr.shape
        box_remap = BoundingBox.remapBBox(self.box_ori, self.box_fmt, face_box)
        lft, top, rig, bot = BoundingBox(np.array(box_remap, dtype=np.int32)).clip(0, 0, w, h).decouple()
        h = bot - top
        w = rig - lft
        st_x, st_y, st_w, st_h = cv2.boundingRect(self.sticker[:, :, 3])
        if st_w == 0 or st_h == 0:
            return canvas_bgr
        sticker_image = self.sticker[st_y:st_y + st_h, st_x:st_x + st_w, ...]
        resized_sticker = cv2.resize(sticker_image, (w, h))
        sticker_bgr = resized_sticker[:, :, :3]
        sticker_mask = resized_sticker[:, :, 3:4]
        part = canvas_bgr[top:bot, lft:rig, :]
        mask = sticker_mask.astype(np.float32) / 255.
        fusion = part * (1 - mask) + sticker_bgr * mask
        fusion_bgr = np.round(fusion).astype(np.uint8)
        canvas_bgr[top:bot, lft:rig, :] = fusion_bgr
        return canvas_bgr

