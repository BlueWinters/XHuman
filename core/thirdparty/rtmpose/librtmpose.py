
import logging
import os
import cv2
import numpy as np
from rtmlib import YOLOX, RTMPose
from rtmlib import draw_skeleton
from ... import XManager



class LibRTMPoseWrapper:
    """
    """
    @staticmethod
    def assertingFiles(root):
        for key, value in LibRTMPoseWrapper.ConfigDict:
            path = '{}/{}'.format(root, value['onnx_model'])
            assert os.path.exists(path), path

    @staticmethod
    def visualBoxes(bgr, boxes, color=(0, 255, 0)):
        for bbox in boxes:
            bgr = cv2.rectangle(bgr, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        return bgr

    @staticmethod
    def visualSkeleton(bgr, keypoints, scores, openpose_skeleton=True, threshold=0.5, radius=2, line_width=2):
        return draw_skeleton(bgr, keypoints, scores, openpose_skeleton, threshold, radius, line_width)

    @staticmethod
    def getResources():
        return [
            LibRTMPoseWrapper.ConfigDict['RTMO-m']['onnx_model'],
            LibRTMPoseWrapper.ConfigDict['RTMPose-m']['onnx_model'],
        ]

    @staticmethod
    def benchmark():
        module = LibRTMPoseWrapper()
        module.initialize()
        bgr = cv2.imread('benchmark/asset/rtmpose/input.png')
        points, prob = module(bgr, targets='source')
        print(points.shape, prob.shape)
        bgr = module.visualSkeleton(bgr, points, prob)
        cv2.imwrite('benchmark/asset/rtmpose/output_rtmlib.png', bgr)

    """
    """
    # reference: https://github.com/Tau-J/rtmlib
    ConfigDict = {
        'RTMPose-m': {
            'onnx_model': 'thirdparty/rtmpose/rtmpose_onnx/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288/end2end.onnx',
            'model_input_size': (288, 384),
        }
    }

    """
    """
    def __init__(self, *args, **kwargs):
        self.max_size = 512 + 256
        self.backend = 'onnxruntime'
        self.device = 'cuda'
        self.default_p = 'RTMPose-m'

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        self.root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
        if hasattr(self, '_pose') is False and 'pose' in kwargs:
            assert kwargs['pose'] in self.ConfigDict, kwargs['pose']
            self.default_p = kwargs['pose']

    @property
    def pose(self):
        if hasattr(self, '_pose') is False:
            config = self.ConfigDict[self.default_p]
            config['onnx_model'] = '{}/{}'.format(self.root, config['onnx_model'])
            self._pose = RTMPose(**config, backend=self.backend, device=self.device)
        return self._pose

    """
    """
    def detect(self, bgr, scores=None, boxes=None):
        if boxes is None:
            scores, boxes = XManager.getModules('human_detection_yolox')(bgr)
            assert len(scores) == len(boxes), (scores.shape, boxes.shape)
            return scores, boxes
        return scores, np.round(boxes).astype(np.int32)

    def inference(self, bgr, boxes):
        scores, boxes = self.detect(bgr, boxes=boxes)
        if len(boxes) == 0:
            return [], []
        points, prob = self.pose(bgr, bboxes=boxes)
        points = np.round(points).astype(np.int32)
        return points, prob

    """
    """
    def _extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        boxes = kwargs.pop('boxes', None)
        return targets, dict(boxes=boxes)

    def _returnResult(self, bgr, output, targets):
        def _formatResult(target):
            points, prob = output
            if target == 'source':
                return points, prob
            if target == 'visual':
                return self.visualSkeleton(np.copy(bgr), points, prob)
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, bgr, *args, **kwargs):
        target, inference_kwargs = self._extractArgs(*args, **kwargs)
        output = self.inference(bgr, **inference_kwargs)
        return self._returnResult(bgr, output, target)
