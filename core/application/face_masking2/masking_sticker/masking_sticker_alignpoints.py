
import logging
import os
import cv2
import numpy as np
import skimage
import functools
from .masking_sticker import MaskingSticker
from ..helper.masking_helper import MaskingHelper
from ....base import XPortrait, XPortraitHelper, XPortraitException
from ....geometry import Rectangle


class MaskingStickerAlignPoints(MaskingSticker):
    """
    """
    NameEN = 'sticker_align_points'
    NameCN = '贴纸_点对齐'

    @staticmethod
    def benchmark():
        pass

    @classmethod
    def parameterize(cls, *args, **kwargs):
        if 'resource' in kwargs:
            def initialize(*a, **ka):
                from ..resource import getResourceStickerAlignPoints
                align_type, prefix = kwargs.pop('resource')
                sticker, points = getResourceStickerAlignPoints(align_type, prefix)
                sticker_params = {'sticker': sticker, align_type: points}
                return functools.partial(cls, *a, *args, **ka, **kwargs, **sticker_params)
            return initialize
        return functools.partial(cls, *args, **kwargs)

    """
    """
    def __init__(self, sticker, *args, **kwargs):
        super(MaskingStickerAlignPoints, self).__init__(*args, **kwargs)
        self.sticker = np.array(sticker, dtype=np.uint8)
        assert len(self.sticker.shape) == 3 and self.sticker.shape[2] == 4, self.sticker.shape  # H,W,4
        # option1: default, face feature with 5 points
        self.align_type = 'face_feature_affine_self'
        self.points = None
        # option2: eyes center fix(similarity transform)
        if 'eyes_center_similarity' in kwargs:
            self.align_type = 'eyes_center_similarity'
            self.points = np.reshape(np.array(kwargs['eyes_center_similarity'], dtype=np.int32), (2, 2))
        if 'mouth_corners_similarity' in kwargs:
            self.align_type = 'mouth_corners_similarity'
            self.points = np.reshape(np.array(kwargs['mouth_corners_similarity'], dtype=np.int32), (2, 2))

    def __str__(self):
        return '{}(sticker={}, align_type={}, points={})'.format(
            self.NameEN, self.sticker.shape, self.align_type, self.points.tolist())

    def getAlignPoints(self, bgr, box, points=None):
        lft, top, rig, bot = box
        assert len(box) == 4, len(box)
        # face_feature: 5 key points on face
        if self.align_type == 'eyes_center_affine_self':
            # eyes_center_affine_self: align points come from sticker(which is a cartoon portrait)
            raise NotImplementedError
        # eyes_center: 2 points, center of both eyes
        if self.align_type == 'eyes_center_similarity':
            if isinstance(points, np.ndarray):
                assert len(points.shape) == 2 and points.shape[1] == 2, points.shape  # 5,2 or 68,2 or 2,2
                if points.shape[0] == 68:
                    lft = np.mean(points[36:40, :], axis=0, keepdims=True)
                    rig = np.mean(points[42:48, :], axis=0, keepdims=True)
                    dst_pts = np.concatenate([lft, rig], axis=0)
                else:
                    assert points.shape[0] == 5 or points.shape[0] == 2, points.shape
                    dst_pts = points[:2, :]
            else:
                cache = XPortrait(bgr[top:bot, lft:rig, :], strategy='area', asserting=True)
                dst_pts = np.copy(XPortraitHelper.getCenterOfEachEyes(cache))[0]
                dst_pts[:, 0] += lft
                dst_pts[:, 1] += top
            return self.points, dst_pts
        # mouth_corners: 2 points, corners of mouth
        if self.align_type == 'mouth_corners_similarity':
            if isinstance(points, np.ndarray):
                assert len(points.shape) == 2 and points.shape[1] == 2, points.shape  # 5,2 or 68,2 or 2,2
                if points.shape[0] == 68:
                    lft = np.mean(points[48:49, :], axis=0, keepdims=True)
                    rig = np.mean(points[54:55, :], axis=0, keepdims=True)
                    dst_pts = np.concatenate([lft, rig], axis=0)
                else:
                    assert points.shape[0] == 5 or points.shape[0] == 2, points.shape
                    dst_pts = points[3:5, :]
            else:
                cache = XPortrait(bgr[top:bot, lft:rig, :], strategy='area', asserting=True)
                landmark = cache.landmark[0]
                lft = np.mean(landmark[48:49, :], axis=0, keepdims=True)
                rig = np.mean(landmark[54:55, :], axis=0, keepdims=True)
                dst_pts = np.concatenate([lft, rig], axis=0)
                dst_pts[:, 0] += lft
                dst_pts[:, 1] += top
            return self.points, dst_pts

        raise NotImplementedError(self.align_type)

    def warpSticker(self, source_bgr, box, points, expand=0.2):
        h, w, c = source_bgr.shape
        box = Rectangle(box).toSquare().expand(expand, expand).clip(0, 0, w, h).asInt()
        src_pts, dst_pts = self.getAlignPoints(source_bgr, box, points)
        transform = skimage.transform.SimilarityTransform()
        transform.estimate(src_pts, dst_pts)
        param = dict(order=1, mode='constant', cval=0, output_shape=(h, w))
        sticker_warped = skimage.transform.warp(self.sticker.astype(np.float32), transform.inverse, **param)
        sticker_warped_bgr, sticker_warped_alpha = sticker_warped[:, :, :3], sticker_warped[:, :, 3]
        return sticker_warped_bgr, sticker_warped_alpha

    def inference(self, bgr, *args, **kwargs):
        raise NotImplementedError

    def inferenceOnMaskingImage(self, source_bgr, canvas_bgr, angle, box, landmark, **kwargs):
        try:
            sticker_warped_bgr, sticker_warped_alpha = self.warpSticker(source_bgr, box, landmark)
            fusion_bgr = MaskingHelper.workOnSelectedMask(
                canvas_bgr, sticker_warped_bgr, sticker_warped_alpha, mask_blur_k=None)
            return fusion_bgr
        except XPortraitException as e:
            # note: detect no faces
            return canvas_bgr

    def inferenceOnMaskingVideo(self, source_bgr, canvas_bgr, face_box, key_points_xy, key_points_score, **kwargs):
        try:
            if key_points_score[2] > 0.5 and key_points_score[1] > 0.5:
                landmark = np.stack([key_points_xy[2, :], key_points_xy[1, :]], axis=0)  # 2,2
                sticker_warped_bgr, sticker_warped_alpha = self.warpSticker(source_bgr, face_box, landmark)
                fusion_bgr = MaskingHelper.workOnSelectedMask(
                    canvas_bgr, sticker_warped_bgr, sticker_warped_alpha, mask_blur_k=None)
                return fusion_bgr
            else:
                return canvas_bgr
        except XPortraitException as e:
            # note: detect no faces
            return canvas_bgr

