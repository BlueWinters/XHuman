
import logging
import cv2
import numpy as np
import ultralytics
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
        'yolo11n-pose', 'yolo11m-pose',  # 'yolo11s-pose', 'yolo11m-pose', 'yolo11l-pose', 'yolo11x-pose',
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

