
import logging
import numpy as np
import cv2
import json
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
            if id == True:
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
    def _detectFace(self, bgr, boxes):
        if boxes is None:
            module = XManager.getModules('face_detection')
            scores, boxes, points = module(bgr)
        return boxes

    def _mapBack(self, pts, lft, rig, top, bot):
        assert len(pts.shape) == 2
        W = rig - lft + 1
        H = bot - top + 1
        new_pts = pts.copy().astype(np.float32)
        new_pts[:, 0] = pts[:, 0] * W + lft
        new_pts[:, 1] = pts[:, 1] * H + top
        return new_pts

    def _clipAndResize(self, bgr, lft, top, rig, bot, dH, dW, ratio:float=0.1):
        H, W, C = bgr.shape
        lft = max(0, lft)
        lft, rig = max(lft, 0), min(rig, W)
        top, bot = max(top, 0), min(bot, H)
        new_bgr = bgr[top:bot+1, lft:rig+1, :]
        return cv2.resize(new_bgr, (dW, dH))

    def _forward(self, batch_bgr):
        assert batch_bgr.shape[1] == self.fH
        assert batch_bgr.shape[2] == self.fW
        assert batch_bgr.shape[3] == 3
        N = batch_bgr.shape[0]
        batch_bgr = np.transpose(batch_bgr, (0, 3, 1, 2)) / 255.
        output = self.engine.inference(batch_bgr)
        out = np.reshape(output, (N, self.num_points, 2))
        return out

    def inference(self, bgr, boxes=None):
        list_bgr_rsz = list()
        list_clip_box = list()
        boxes = self._detectFace(bgr, boxes)
        for n, box in enumerate(boxes):
            assert isinstance(box, np.ndarray), type(box)
            if len(box) == 4:
                lft, top, rig, bot = list(map(int, box))
                list_clip_box.append((lft, top, rig, bot))
                bgr_rsz = self._clipAndResize(bgr, lft, top, rig, bot, self.fH, self.fW)
                list_bgr_rsz.append(bgr_rsz)
            elif len(box) == 68:
                points = np.array(box, dtype=np.int32)
                lft, top = points[:,0].min(), points[:,1].min()
                rig, bot = points[:,0].max(), points[:,1].max()
                list_clip_box.append((lft, top, rig, bot))
                bgr_rsz = self._clipAndResize(bgr, lft, top, rig, bot, self.fH, self.fW)
                list_bgr_rsz.append(bgr_rsz)
            else:
                raise NotImplementedError('invalid input box shape: {}'.format(box.shape))
        # network inference
        landmarks = np.zeros(shape=(len(boxes), 68, 2), dtype=np.float32)
        if len(list_bgr_rsz) > 0:
            batch_pts68 = self._forward(np.array(list_bgr_rsz, dtype=np.uint8))
            for n in range(len(boxes)):
                lft, top, rig, bot = list_clip_box[n]
                points = np.reshape(batch_pts68[n, :, :], (68, 2))
                points = self._mapBack(points, lft=lft, rig=rig, top=top, bot=bot)
                landmarks[n, :, :] = points
        return np.round(landmarks).astype(np.int32)  # N,68,2 --> N = len(landmarks)

    """
    """
    def _extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        boxes = kwargs.pop('boxes', None)
        inference_kwargs = dict(boxes=boxes)
        return targets, inference_kwargs

    def _returnResult(self, bgr, output, targets):
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
        targets, inference_kwargs = self._extractArgs(*args, **kwargs)
        output = self.inference(bgr, **inference_kwargs)
        return self._returnResult(bgr, output, targets)
