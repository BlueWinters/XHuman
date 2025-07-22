
import logging
import numpy as np
import cv2
import json
from ..geometry import GeoFunction
from .. import XManager


class LibFaceLandmark:
    @staticmethod
    def visual(bgr, pts, r=1, color=(0,0,255), id:bool=False):
        assert pts.shape[0] == 68
        assert pts.shape[1] == 2
        for n in range(68):
            x, y = pts[n].tolist()
            position = (int(round(x)), int(round(y)))
            cv2.circle(bgr, position, r, color)
            if id is True:
                cv2.putText(bgr, str(n), position, cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255), 1)
        return bgr

    @staticmethod
    def toString(pts:np.ndarray):
        assert pts.shape[0] == 68
        assert pts.shape[1] == 2
        string:str = ''
        for n in range(68): string += \
            '{:d} {:d}\n'.format(int(round(pts[n,0])), int(round(pts[n,1])))
        return string

    @staticmethod
    def fromText(path):
        data = []
        file = open(path, 'r')
        for line in file: data.append(
            [float(v) for v in line.rstrip('\n').split(' ')])
        return np.array(data, dtype=np.int32)

    @staticmethod
    def getResources():
        return [
            LibFaceLandmark.EngineConfig['parameters'],
        ]

    """
    """
    EngineConfig = {
        'type': 'torch',
        'device': 'cuda:0',
        'parameters': 'base/face_landmark.ts'
    }

    """
    """
    def __init__(self, *args, **kwargs):
        self.engine = XManager.createEngine(self.EngineConfig)
        # other configs
        self.fH, self.fW = 128, 128
        self.num_points = 68

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    """
    """
    def initialize(self, *args, **kwargs):
        self.engine.initialize(*args, **kwargs)

    """
    """
    def formatInput(self, bgr, image_angles=None, boxes=None, points=None):
        if boxes is None:
            if image_angles is not None:
                module = XManager.getModules('face_detection')
                scores, boxes, points, angles = module(bgr, image_angles=image_angles)
                return self.clipWithBox(bgr, boxes, angles, self.fH, self.fW)
            else:
                module = XManager.getModules('face_detection')
                scores, boxes, points = module(bgr, image_angles=image_angles)
                return self.clipWithBox(bgr, boxes, None, self.fH, self.fW)
        else:
            # include image_angles is None or not
            return self.clipWithBox(bgr, boxes, image_angles, self.fH, self.fW)

    @staticmethod
    def clipAndResizeB(bgr, pts, angle, box):
        assert len(pts.shape) == 2
        new_pts = pts.copy().astype(np.float32)
        bgr_rot = GeoFunction.rotateImage(bgr, angle)
        box_rot = GeoFunction.rotateBoxes(box, angle, bgr.shape[0], bgr.shape[1])
        lft, top, rig, bot = box_rot.tolist()
        w = rig - lft + 1
        h = bot - top + 1
        new_pts[:, 0] = new_pts[:, 0] * w + lft
        new_pts[:, 1] = new_pts[:, 1] * h + top
        angle_back = GeoFunction.rotateBack(angle)
        new_pts = GeoFunction.rotatePoints(new_pts, angle_back, bgr_rot.shape[0], bgr_rot.shape[1])
        return new_pts

    @staticmethod
    def clipAndResizeF(bgr, angle, box, dst_h, dst_w):
        lft, top, rig, bot = box.tolist()
        h, w, c = bgr.shape
        lft = max(0, lft)
        lft, rig = max(lft, 0), min(rig, w)
        top, bot = max(top, 0), min(bot, h)
        new_bgr = bgr[top:bot+1, lft:rig+1, :]
        new_bgr_rot = GeoFunction.rotateImage(new_bgr, angle)
        return cv2.resize(new_bgr_rot, (dst_w, dst_h))

    def clipWithBox(self, bgr, boxes, angles, dst_h, dst_w):
        data = []
        angles = [0] * len(boxes) if angles is None else angles
        assert len(angles) == len(boxes), (len(boxes), len(angles))
        for n, (box, angle) in enumerate(zip(boxes, angles)):
            assert isinstance(box, np.ndarray) and len(box) == 4, type(box)
            box_int = np.round(box).astype(np.int32)
            bgr_fmt = self.clipAndResizeF(bgr, angle, box, dst_h, dst_w)
            data.append((bgr_fmt, lambda pts, a=angle, b=box_int: self.clipAndResizeB(bgr, pts, a, b)))
        return data

    def forward(self, batch_bgr):
        assert batch_bgr.shape[1] == self.fH
        assert batch_bgr.shape[2] == self.fW
        assert batch_bgr.shape[3] == 3
        N = batch_bgr.shape[0]
        batch_bgr = np.transpose(batch_bgr, (0, 3, 1, 2)) / 255.
        output = self.engine.inference(batch_bgr)
        out = np.reshape(output, (N, self.num_points, 2))
        return out

    def inference(self, bgr, image_angles=None, boxes=None, points=None):
        data = self.formatInput(bgr, image_angles, boxes, points)
        if len(data) > 0:
            landmarks = np.zeros(shape=(len(data), 68, 2), dtype=np.float32)
            format_bgr_list = [pair[0] for pair in data]
            batch_pts68 = self.forward(np.array(format_bgr_list, dtype=np.uint8))
            for n, pair in enumerate(data):
                assert isinstance(pair, tuple), type(pair)
                format_bgr, transform_points = pair
                points68 = np.reshape(batch_pts68[n, :, :], (68, 2))
                points68_format = transform_points(points68)
                landmarks[n, :, :] = points68_format
            return np.round(landmarks).astype(np.int32)  # N,68,2 --> N = len(landmarks)
        return list()

    """
    """
    def extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        boxes = kwargs.pop('boxes', None)
        points = kwargs.pop('points', None)
        image_angles = kwargs.pop('image_angles', None)
        inference_kwargs = dict(boxes=boxes, points=points, image_angles=image_angles)
        return targets, inference_kwargs

    def returnResult(self, bgr, output, targets):
        def _formatResult(target):
            if target == 'source':
                return output
            if target == 'json':
                data = list()
                for n, pts in enumerate(output):
                    data.append(dict(landmarks=pts.tolist()))
                return json.dumps(data, indent=4)
            if target == 'text':
                if len(output) == 1:
                    return LibFaceLandmark.toString(output[0])
                if len(output) == 0:
                    raise Exception('no face in the image')
                if len(output) >= 2:
                    raise Exception('too many faces() in the image'.format(len(output)))
            if target == 'visual':
                for n, pts in enumerate(output):
                    LibFaceLandmark.visual(bgr, np.round(pts).astype(np.int32))
                return bgr
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, bgr, *args, **kwargs):
        targets, inference_kwargs = self.extractArgs(*args, **kwargs)
        output = self.inference(bgr, **inference_kwargs)
        return self.returnResult(bgr, output, targets)
