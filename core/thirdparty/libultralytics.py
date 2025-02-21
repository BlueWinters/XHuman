
import logging
import cv2
import numpy as np
import ultralytics
from ..geometry import GeoFunction
from .. import XManager


class LibUltralyticsWrapper:
    """
    """
    @staticmethod
    def benchmark():
        pass

    @staticmethod
    def getResources():
        return ['{}/{}.pt'.format(LibUltralyticsWrapper.CheckpointBase, name)
                for name in LibUltralyticsWrapper.ModelList]

    """
    """
    CheckpointBase = 'thirdparty/ultralytics'
    ModelList = [
        'yolo11n',  # 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x',
        'yolo11n-pose', 'yolo11m-pose', 'yolo11x-pose',  # 'yolo11s-pose', 'yolo11m-pose', 'yolo11l-pose',
        'yolo11n-seg', 'yolo11m-seg', 'yolo11x-seg',
    ]

    def __init__(self, *args, **kwargs):
        self.model_dict = dict()

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        if hasattr(self, 'root') is False:
            root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
            self.root = '{}/{}'.format(root, self.CheckpointBase)

    def getSpecific(self, name):
        assert name in LibUltralyticsWrapper.ModelList, name
        if name not in self.model_dict:
            self.model_dict[name] = ultralytics.YOLO('{}/{}'.format(self.root, name))
            # setattr(self, '_{}'.format(name), self.model_dict[name])
        return self.model_dict[name]

    def __getitem__(self, name: str):
        return self.getSpecific(name)

    def resetTracker(self, name):
        if name in self.model_dict:
            self.model_dict[name].predictor.trackers[0].reset()

    """
    """
    def detect(self, bgr, name, rotations=None, **kwargs):
        detect_kwargs = kwargs.pop('detect_kwargs', dict(classes=[0]))
        assert isinstance(detect_kwargs, dict), detect_kwargs
        scores_collect, boxes_collect, points_collect, masks_collect, angles_collect = [], [], [], [], []
        rotations = rotations if isinstance(rotations, (list, tuple)) else [0]
        for n, rot in enumerate(rotations):
            scores, boxes, angles, points, masks = self.inferenceWithRotation(
                bgr, name, rot, detect_kwargs)
            if len(scores) > 0:
                scores_collect.append(scores)
                boxes_collect.append(boxes)
                points_collect.append(points)
                masks_collect.append(masks)
                angles_collect.append(angles)
        # NMS
        if len(scores_collect) > 0:
            scores = np.concatenate(scores_collect, axis=0)
            boxes = np.concatenate(boxes_collect, axis=0)
            angles = np.concatenate(angles_collect, axis=0)
            points = np.concatenate(points_collect, axis=0) if 'pose' in name else None
            masks = np.concatenate(masks_collect, axis=0) if 'seg' in name else None
            keep = self.doNonMaximumSuppression(scores, boxes, 0.6)
            scores = scores[keep]
            boxes = boxes[keep]
            points = points[keep] if 'pose' in name else None
            masks = masks[keep] if 'seg' in name else None
            angles = angles[keep]
        else:
            h, w, c = bgr.shape
            scores = np.zeros(shape=(0,), dtype=np.float32)
            boxes = np.zeros(shape=(0, 4), dtype=np.int32)
            points = np.zeros(shape=(0, 10), dtype=np.int32) if 'pose' in name else None
            masks = np.zeros(shape=(0, h, w), dtype=np.int32) if 'seg' in name else None
            angles = np.zeros(shape=(0,), dtype=np.int32)
        return scores, boxes, angles, points, masks

    def inferenceWithRotation(self, bgr, name, image_angle, detect_kwargs):
        module = self.getSpecific(name)
        if image_angle == 0:
            result = module(bgr, **detect_kwargs, verbose=False)[0]
            scores, boxes, points, masks = self.detachResult(result)
            angles = np.zeros(shape=len(scores), dtype=np.int32)
            return scores, boxes, angles, points, masks
        if image_angle in GeoFunction.CVRotationDict:
            rot = cv2.rotate(bgr, GeoFunction.CVRotationDict[image_angle])
            result = module(rot, classes=[0], verbose=False)[0]
            scores, boxes, points, masks = self.detachResult(result)
            h, w, c = rot.shape
            angle_back = GeoFunction.rotateBack(image_angle)
            boxes = GeoFunction.rotateBoxes(np.reshape(boxes, (len(scores), 4)), angle_back, h, w)
            boxes = np.reshape(boxes, (len(scores), 4))
            if isinstance(points, np.ndarray):
                points = GeoFunction.rotatePoints(np.reshape(points, (len(scores), -1, 2)), angle_back, h, w)
                points = np.reshape(points, (len(scores), 10))
            if isinstance(masks, np.ndarray):
                masks = np.stack([GeoFunction.rotateImage(mask, angle_back) for mask in masks], axis=0)
            angles = np.ones(shape=len(scores), dtype=np.int32) * image_angle
            return scores, boxes, angles, points, masks
        raise ValueError('angle {} not in [0,90,180,270]'.format(image_angle))

    @staticmethod
    def detachResult(result):
        classify = np.reshape(np.round(result.boxes.cls.cpu().numpy()).astype(np.int32), (-1,))
        scores = np.reshape(result.boxes.conf.cpu().numpy().astype(np.float32), (-1,))
        boxes = np.reshape(np.round(result.boxes.xyxy.cpu().numpy()).astype(np.int32), (-1, 4,))
        points = None
        masks = None
        if result.keypoints is not None:
            points = np.reshape(result.keypoints.data.cpu().numpy().astype(np.float32), (-1, 17, 3))
        if result.masks is not None:
            h, w = result.orig_shape
            masks = np.zeros(shape=(len(classify), h, w), dtype=np.uint8)
            instance_masks = np.round(result.masks.cpu().numpy().data * 255).astype(np.uint8)  # note: C,H,W and [0,1]
            for n in range(len(classify)):
                masks[n, :, :] = cv2.resize(instance_masks[n, :, :], (w, h))
        return scores, boxes, points, masks

    @staticmethod
    def doNonMaximumSuppression(scores, boxes, nms_threshold):
        """Pure Python NMS baseline."""
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_threshold)[0]
            order = order[inds + 1]
        return keep

