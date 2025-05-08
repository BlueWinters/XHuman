
import json
import logging
import cv2
import numpy as np
from .. import XManager


class LibHeadPose:
    """
    """
    @staticmethod
    def plot_axis(bgr, yaw, pitch, roll, cx=None, cy=None, size=None):
        # if the input is degree, use the behind process to convert degree to radian
        from math import cos, sin
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if cx is not None and cy is not None:
            cx = cx
            cy = cy
        else:
            height, width = bgr.shape[:2]
            cx = width / 2
            cy = height / 2
        if size is None:
            size = (bgr.shape[0] + bgr.shape[1]) // 10
        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + cx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + cy
        # Y-Axis drawn in green
        x2 = size * (-cos(yaw) * sin(roll)) + cx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + cy
        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + cx
        y3 = size * (-cos(yaw) * sin(pitch)) + cy

        cv2.line(bgr, (int(cx), int(cy)), (int(x1), int(y1)), (0, 0, 255), 5)
        cv2.line(bgr, (int(cx), int(cy)), (int(x2), int(y2)), (0, 255, 0), 5)
        cv2.line(bgr, (int(cx), int(cy)), (int(x3), int(y3)), (255, 0, 0), 5)
        return bgr

    @staticmethod
    def visual(bgr, radian, points):
        yaw, pitch, roll = (radian.astype(np.float32) * 180 / np.pi).tolist()
        cx, cy = np.mean(points, axis=0).round().astype(np.int32).tolist()
        size = (np.max(points[:, 0]) - np.min(points[:, 0]) +
                np.max(points[:, 1]) - np.min(points[:, 1])) / 2
        LibHeadPose.plot_axis(bgr, yaw, pitch, roll, cx=cx, cy=cy, size=int(round(size/3)))
        return bgr
        # for radian, points in zip(radians, landmarks):
        #     yaw, pitch, roll = (radian.astype(np.float32) * 180 / np.pi).tolist()
        #     cx, cy = np.mean(points, axis=0).round().astype(np.int32).tolist()
        #     size = (np.max(points[:, 0]) - np.min(points[:, 0]) +
        #             np.max(points[:, 1]) - np.min(points[:, 1])) / 2
        #     LibHeadPose.plot_axis(bgr, yaw, pitch, roll, cx=cx, cy=cy, size=int(round(size/3)))
        # return bgr

    @staticmethod
    def getResources():
        return [
            LibHeadPose.EngineConfig['parameters'],
        ]

    """
    """
    EngineConfig = {
        'type': 'torch',
        'device': 'cuda:0',
        'parameters': 'base/head_pose.ts'
    }

    """
    """
    def __init__(self, *args, **kwargs):
        self.engine = XManager.createEngine(self.EngineConfig)
        self.H, self.W = 128, 128
        self.norm = 180 / np.pi
        self.num_points = 68

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    """
    """
    def initialize(self, *args, **kwargs):
        self.engine.initialize(*args, **kwargs)

    """
    """

    @staticmethod
    def clipAndResize(bgr, lft, rig, top, bot, d_h, d_w):
        h, w, c = bgr.shape
        lft, rig = max(lft, 0), min(rig, w)
        top, bot = max(top, 0), min(bot, h)
        new_bgr = bgr[top:bot+1, lft:rig+1, :]
        return cv2.resize(new_bgr, (d_w, d_h))

    @staticmethod
    def calculateOuterWith68pts(pts, ext: float):
        # calculate the outer rectangle and expand it
        x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
        y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        lft = int(round(x_min - w * ext))
        rig = int(round(x_max + w * ext))
        top = int(round(y_min - h * ext))
        bot = int(round(y_max + h * ext))
        return lft, rig, top, bot

    def format(self, bgr, pts68, ext=0.3):
        assert isinstance(pts68, np.ndarray)
        assert pts68.shape == (68, 2)
        lft, rig, top, bot = self.calculateOuterWith68pts(pts68, ext=ext)
        bgr_rsz = self.clipAndResize(bgr, lft, rig, top, bot, self.H, self.W)
        return bgr_rsz

    def forward(self, bgr_rsz):
        assert bgr_rsz.shape[0] == self.H, bgr_rsz.shape
        assert bgr_rsz.shape[1] == self.W, bgr_rsz.shape
        assert bgr_rsz.shape[2] == 3, bgr_rsz.shape
        batch_bgr = np.expand_dims(bgr_rsz, axis=0)
        batch_bgr = np.transpose(batch_bgr, (0, 3, 1, 2)).astype(np.float32) / 255.
        yaw, pitch, roll = self.engine.inference(batch_bgr)
        return float(yaw), float(pitch), float(roll)

    @staticmethod
    def estimateLandmark(bgr, landmarks):
        if landmarks is None:
            module = XManager.getModules('face_landmark')
            landmarks = module(bgr)
        return landmarks

    def inference(self, bgr, landmarks=None):
        landmarks = self.estimateLandmark(bgr, landmarks)
        radians = np.zeros(shape=(len(landmarks), 3), dtype=np.float32)
        for n in range(len(landmarks)):
            bgr_rsz = self.format(bgr, landmarks[n, :, :])
            yaw, pitch, roll = self.forward(bgr_rsz)
            radians[n, :] = np.array([yaw, pitch, roll], dtype=np.float32)
        return landmarks, radians

    """
    """
    def extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        landmarks = kwargs.pop('landmarks', None)
        inference_kwargs = dict(landmarks=landmarks)
        return targets, inference_kwargs

    def returnResult(self, bgr, landmarks, radians, targets):
        def _formatResult(target):
            if target == 'source':
                return radians
            if target == 'degree':
                return radians * self.norm
            if target == 'visual':
                return LibHeadPose.visual(bgr, radians, landmarks)
            if target == 'json':
                data = list()
                for n, r in enumerate(radians):
                    data.append(dict(randian=r.tolist()))
                return json.dumps(data, indent=4)
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, bgr, *args, **kwargs):
        targets, inference_kwargs = self.extractArgs(*args, **kwargs)
        landmarks, radians = self.inference(bgr, **inference_kwargs)
        return self.returnResult(bgr, landmarks, radians, targets)
