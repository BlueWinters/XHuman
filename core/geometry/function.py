
import cv2
import numpy as np


class GeoFunction:
    """
    """
    CVRotationDict = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE
    }

    @staticmethod
    def rotateBack(angle):
        if angle == 0:
            return angle
        if angle == 90:
            return 270
        if angle == 180:
            return 180
        if angle == 270:
            return 90
        raise ValueError('angle {} not in [90,180,270]'.format(angle))

    @staticmethod
    def rotateImage(bgr, angle):
        if angle == 0:
            return bgr
        if angle == 90:
            return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(bgr, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        raise ValueError('angle {} not in [90,180,270]'.format(angle))

    @staticmethod
    def rotatePoints(points, angle, h, w):
        assert points.shape[-1] == 2, points.shape  # ...,2 (N,...,K,2)
        if angle == 0:
            return np.copy(points)
        if angle == 90:
            points_copy = np.copy(points)
            points_copy[..., 0] = h - points[..., 1]
            points_copy[..., 1] = points[..., 0]
            return points_copy
        if angle == 180:
            points_copy = np.copy(points)
            points_copy[..., 0] = w - points[..., 0]
            points_copy[..., 1] = h - points[..., 1]
            return points_copy
        if angle == 270:
            points_copy = np.copy(points)
            points_copy[..., 0] = points[..., 1]
            points_copy[..., 1] = w - points[..., 0]
            return points_copy
        raise ValueError('angle {} not in [90,180,270]'.format(angle))

    @staticmethod
    def rotateBoxes(boxes, angle, h, w):
        assert boxes.shape[-1] == 4, boxes.shape  # ...,4 (N,...,K,4) --> lft,top,rig,bot
        if angle == 0:
            return np.copy(boxes)
        if angle == 90:
            boxes_copy = np.copy(boxes)
            boxes_copy[..., 0] = h - boxes[..., 3]
            boxes_copy[..., 1] = boxes[..., 0]
            boxes_copy[..., 2] = h - boxes[..., 1]
            boxes_copy[..., 3] = boxes[..., 2]
            return boxes_copy
        if angle == 180:
            boxes_copy = np.copy(boxes)
            boxes_copy[..., 0] = w - boxes[..., 2]
            boxes_copy[..., 1] = h - boxes[..., 3]
            boxes_copy[..., 2] = w - boxes[..., 0]
            boxes_copy[..., 3] = h - boxes[..., 1]
            return boxes_copy
        if angle == 270:
            boxes_copy = np.copy(boxes)
            boxes_copy[..., 0] = boxes[..., 1]
            boxes_copy[..., 1] = w - boxes[..., 2]
            boxes_copy[..., 2] = boxes[..., 3]
            boxes_copy[..., 3] = w - boxes[..., 0]
            return boxes_copy
        raise ValueError('angle {} not in [0,90,180,270]'.format(angle))

    """
    """
    @staticmethod
    def isValidBox(box, as_int=True):
        if isinstance(box, np.ndarray):
            assert len(box.shape) == 1 and len(box) == 4, box.shape
            lft, top, rig, bot = box.astype(np.int32).tolist() if as_int else box.tolist()
            return bool(lft < rig) and bool(top < bot)
        if isinstance(box, (tuple, list)):
            assert len(box) == 4, box
            lft, top, rig, bot = [int(v) for v in box] if as_int else box
            return bool(lft < rig) and bool(top < bot)
        raise NotImplementedError

    """
    """
    @staticmethod
    def estimateProjectionPoint(point1, point2, point_proj, r=None):
        v = point2 - point1
        u = point_proj - point1
        if r is not None:
            return point1 + v * (np.dot(u, v) / np.dot(v, v) - r)
        return point1 + v * (np.dot(u, v) / np.dot(v, v))

    @staticmethod
    def calculateRadian(point0, point1, point2):
        try:
            v1 = point1 - point0
            v2 = point2 - point0
            d1 = float(np.linalg.norm(v1))
            d2 = float(np.linalg.norm(v2))
            if d1 == 0. or d2 == 0.:
                return None
            dot = np.dot(v1, v2) / (d1 * d2)
            dot = np.clip(dot, -1, 1)
            return float(np.arccos(dot))
        except Exception as e:
            import logging
            logging.error('calculateRadian error: {}, {}, {}, {} --> {}'.format(v1, v2, d1, d2, dot))
            raise e
