
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


