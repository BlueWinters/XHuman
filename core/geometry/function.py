
import cv2
import numpy as np


class GeoFunction:
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
            return angle
        if angle == 90:
            return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(bgr, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        raise ValueError('angle {} not in [90,180,270]'.format(angle))

    @staticmethod
    def rotatePoints(bgr, points, angle):
        assert points.shape[-1] == 2, points.shape  # ...,2 (N,...,K,2)
        h, w, c = bgr.shape
        if angle == 0:
            return points
        if angle == 90:
            points_copy = np.copy(points)
            points[..., 0] = points_copy[..., 1]
            points[..., 1] = h - points_copy[..., 0]
            return points
        if angle == 180:
            points_copy = np.copy(points)
            points[..., 0] = w - points_copy[..., 0]
            points[..., 1] = h - points_copy[..., 1]
            return points
        if angle == 270:
            points = np.copy(np.reshape(points, (-1, 5, 2)))
            points_copy = np.copy(points)
            points[..., 0] = w - points_copy[..., 1]
            points[..., 1] = points_copy[..., 0]
            return points
        raise ValueError('angle {} not in [90,180,270]'.format(angle))

    @staticmethod
    def rotateBoxes(bgr, boxes, angle):
        assert boxes.shape[-1] == 4, boxes.shape  # ...,4 (N,...,K,4) --> lft,top,rig,bot
        h, w, c = bgr.shape
        if angle == 0:
            return boxes
        if angle == 90:
            boxes_copy = np.copy(boxes)
            boxes[..., 0] = boxes_copy[..., 1]
            boxes[..., 1] = h - boxes_copy[..., 2]
            boxes[..., 2] = boxes_copy[..., 3]
            boxes[..., 3] = h - boxes_copy[..., 0]
            return boxes
        if angle == 180:
            boxes_copy = np.copy(boxes)
            boxes[..., 0] = w - boxes_copy[..., 2]
            boxes[..., 1] = h - boxes_copy[..., 3]
            boxes[..., 2] = w - boxes_copy[..., 0]
            boxes[..., 3] = h - boxes_copy[..., 1]
            return boxes
        if angle == 270:
            boxes_copy = np.copy(boxes)
            boxes[..., 0] = w - boxes_copy[..., 3]
            boxes[..., 1] = boxes_copy[..., 0]
            boxes[..., 2] = w - boxes_copy[..., 1]
            boxes[..., 3] = boxes_copy[..., 2]
            return boxes
        raise ValueError('angle {} not in [0,90,180,270]'.format(angle))


