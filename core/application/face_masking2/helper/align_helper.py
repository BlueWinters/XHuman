
import cv2
import numpy as np
import skimage
from ..helper.boundingbox import BoundingBox
from ....base import XPortrait


class AlignHelper:
    @staticmethod
    def realignFacePoints(points, w, h, index):
        # template = np.array([[197, 176], [402, 176], [302, 356]], dtype=np.float32)
        template = np.array([[155, 88], [360, 91], [256, 213]], dtype=np.float32)
        dst_pts = points[np.array(index, dtype=np.int32)]
        src_pts = template[:len(dst_pts), :]
        transform = skimage.transform.SimilarityTransform()
        transform.estimate(src_pts, dst_pts)
        box = np.array([[0, 0, 1], [512, 0, 1], [512, 512, 1], [0, 512, 1]], dtype=np.float32)
        box_remap = np.dot(transform.params, box.T)[:2, :].T
        box_remap_int = np.round(box_remap).astype(np.int32)
        lft = np.min(box_remap_int[:, 0])
        rig = np.max(box_remap_int[:, 0])
        top = np.min(box_remap_int[:, 1])
        bot = np.max(box_remap_int[:, 1])
        # bbox = lft, top, rig, bot
        bbox = BoundingBox(np.array([lft, top, rig, bot], dtype=np.int32)).toSquare().clip(0, 0, w - 1, h - 1).asInt()
        return bbox, box_remap_int

    @staticmethod
    def getAlignFaceCache(bgr, src_pts) -> XPortrait:
        dst_pts = np.array([[192, 239], [318, 240], [256, 314]], dtype=np.float32)
        # dst_pts = np.array([[155, 88], [360, 91], [256, 213]], dtype=np.float32)
        transform = skimage.transform.SimilarityTransform()
        transform.estimate(np.reshape(src_pts, (3, 2)), dst_pts)
        warped_f = skimage.transform.warp(
            bgr.astype(np.float32), transform.inverse, order=1, mode='constant', cval=255, output_shape=(512, 512))
        warped = np.round(warped_f).astype(np.uint8)
        return XPortrait(warped)

    """
    """
    @staticmethod
    def transformPoints2FaceBox(bgr, key_points, box, threshold=0.5):
        h, w, c = bgr.shape
        points_xy = key_points[:, :2].astype(np.float32)
        points_score = key_points[:, 2].astype(np.float32)
        # index: left-eye, right-eye, nose
        if points_score[0] > threshold and points_score[1] > threshold and points_score[2] > threshold:
            bbox, box_rot = AlignHelper.realignFacePoints(points_xy, w, h, index=[2, 1, 0])
            return np.array(bbox, dtype=np.int32), np.reshape(box_rot, (4, 2))
        if points_score[1] > threshold and points_score[2] > threshold:
            bbox, box_rot = AlignHelper.realignFacePoints(points_xy, w, h, index=[2, 1])
            return np.array(bbox, dtype=np.int32), np.reshape(box_rot, (4, 2))
        if points_score[0] > threshold and points_score[1] > threshold:
            bbox, box_rot = AlignHelper.realignFacePoints(points_xy, w, h, index=[1, 0])
            return np.array(bbox, dtype=np.int32), np.reshape(box_rot, (4, 2))
        if points_score[0] > threshold and points_score[2] > threshold:
            bbox, box_rot = AlignHelper.realignFacePoints(points_xy, w, h, index=[2, 0])
            return np.array(bbox, dtype=np.int32), np.reshape(box_rot, (4, 2))
        return np.array([0, 0, 0, 0], dtype=np.int32), np.array([0, 0, 0, 0], dtype=np.int32)

    @staticmethod
    def transformPoints2FaceBox2(bgr, key_points, box, threshold=0.5):
        h, w, c = bgr.shape
        confidence = key_points[:, 2].astype(np.float32)
        points = key_points[:, :2].astype(np.float32)
        if confidence[3] > threshold and confidence[4] > threshold:
            lft_ear = points[4, :]
            rig_ear = points[3, :]
            len_ear = np.linalg.norm(lft_ear - rig_ear)
            lft = min(lft_ear[0], rig_ear[0])
            rig = max(lft_ear[0], rig_ear[0])
            ctr_ear = (lft_ear + rig_ear) / 2
            top = int(max(ctr_ear[1] - 0.4 * len_ear, 0))
            bot = int(min(ctr_ear[1] + 0.6 * len_ear, h))
            if points[4, 0] < points[2, 0] < points[0, 0] < points[1, 0] < points[3, 0]:
                bbox = BoundingBox(np.array([lft, top, rig, bot], dtype=np.int32)).toSquare().clip(0, 0, w - 1, h - 1).asInt()
                if confidence[0] > threshold:
                    return np.array(bbox, dtype=np.int32), confidence
                else:
                    return np.array(bbox, dtype=np.int32), confidence
            else:
                if confidence[1] > threshold and confidence[2] > threshold:
                    bbox = AlignHelper.realignFacePoints(points, w, h, index=[2, 1, 0])
                    return np.array(bbox, dtype=np.int32), confidence
        if confidence[3] > threshold and confidence[1] > threshold:
            rig = points[3, 0]  # points[1, 0] < points[3, 0]
            if confidence[2] > threshold:
                assert confidence[0] > 0, confidence[0]
                if points[2, 0] < points[0, 0] < points[1, 0] < points[3, 0]:
                    rig_ratio = float(rig - points[1, 0]) / float(rig - points[0, 0])
                    lft = points[2, 0] - float(rig - points[0, 0]) * (1 - rig_ratio)
                    lft, rig = min(lft, rig), max(lft, rig)
                    len_c2rig = abs(rig - points[0, 0])
                    top = int(max(points[0, 1] - 0.4 * len_c2rig, 0))
                    bot = int(min(points[0, 1] + 0.8 * len_c2rig, h))
                    bbox = BoundingBox(np.array([lft, top, rig, bot], dtype=np.int32)).toSquare().clip(0, 0, w - 1, h - 1).asInt()
                else:
                    bbox = AlignHelper.realignFacePoints(points, w, h, index=[2, 1, 0])
                return np.array(bbox, dtype=np.int32), confidence
            if confidence[0] > threshold:
                if points[0, 0] < points[1, 0] < points[3, 0]:
                    lft = points[0, 0] - abs(points[0, 0] - points[1, 0])
                    lft, rig = min(lft, rig), max(lft, rig)
                    len_c2rig = abs(rig - points[0, 0])
                    top = int(max(points[0, 1] - 0.4 * len_c2rig, 0))
                    bot = int(min(points[0, 1] + 0.8 * len_c2rig, h))
                    bbox = BoundingBox(np.array([lft, top, rig, bot], dtype=np.int32)).toSquare().clip(0, 0, w - 1, h - 1).asInt()
                else:
                    bbox = AlignHelper.realignFacePoints(points, w, h, index=[1, 0])
                return np.array(bbox, dtype=np.int32), confidence
        if confidence[4] > threshold and confidence[2] > threshold:
            lft = points[4, 0]
            if confidence[1] > threshold:
                assert confidence[0] > 0, confidence[0]
                if points[4, 0] < points[2, 0] < points[0, 0] < points[1, 0]:
                    lft_ratio = float(points[2, 0] - lft) / float(points[0, 0] - lft)
                    rig = points[1, 0] + float(points[0, 0] - lft) * (1 - lft_ratio)
                    lft, rig = min(lft, rig), max(lft, rig)
                    len_c2lft = abs(points[0, 0] - lft)
                    top = int(max(points[0, 1] - 0.4 * len_c2lft, 0))
                    bot = int(min(points[0, 1] + 0.8 * len_c2lft, h))
                    bbox = BoundingBox(np.array([lft, top, rig, bot], dtype=np.int32)).toSquare().clip(0, 0, w - 1, h - 1).asInt()
                else:
                    bbox = AlignHelper.realignFacePoints(points, w, h, index=[2, 1, 0])
                return np.array(bbox, dtype=np.int32), confidence  # 2 + np.mean(1-confidence[5:])
            if confidence[0] > threshold:
                if points[4, 0] < points[2, 0] < points[0, 0]:
                    rig = points[0, 0] + abs(points[2, 0] - points[0, 0])  # min(lft, points[0, 0])
                    lft, rig = min(lft, rig), max(lft, rig)
                    len_c2lft = abs(points[0, 0] - lft)
                    top = int(max(points[0, 1] - 0.4 * len_c2lft, 0))
                    bot = int(min(points[0, 1] + 0.8 * len_c2lft, h))
                    bbox = BoundingBox(np.array([lft, top, rig, bot], dtype=np.int32)).toSquare().clip(0, 0, w - 1, h - 1).asInt()
                else:
                    bbox = AlignHelper.realignFacePoints(points, w, h, index=[2, 0])
                return np.array(bbox, dtype=np.int32), confidence
        return np.array([0, 0, 0, 0], dtype=np.int32), confidence
