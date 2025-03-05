
import logging
import copy
import os
import cv2
import numpy as np
import json


class ScanningImage:
    """
    """
    @classmethod
    def createImageScanning(cls, **kwargs):
        if 'path_in_json' in kwargs and isinstance(kwargs['path_in_json'], str):
            path_in_json = kwargs['path_in_json']
            assert os.path.exists(path_in_json), path_in_json
            scanning_image = cls(bgr=None)
            for info_dict in json.load(open(path_in_json, 'r')):
                scanning_image.info_object_list.append(cls.getObjectClass().createFromDict(info_dict))
            return scanning_image
        if 'info_string' in kwargs and isinstance(kwargs['info_string'], str):
            info_string = kwargs['info_string']
            scanning_image = cls(bgr=None)
            for info_dict in json.loads(info_string):
                scanning_image.info_object_list.append(cls.getObjectClass().createFromDict(info_dict))
            return scanning_image
        if 'objects_list' in kwargs and isinstance(kwargs['objects_list'], list):
            scanning_image = cls(bgr=None)
            for info in kwargs['objects_list']:
                scanning_image.info_object_list.append(cls.getObjectClass().createFromDict(info))
            return scanning_image
        raise NotImplementedError('"objects_list", "path_in_json", "info_string" not in kwargs')

    @classmethod
    def getObjectClass(cls):
        raise NotImplementedError

    """
    """
    def __init__(self, bgr, *args, **kwargs):
        self.bgr = np.copy(bgr) if isinstance(bgr, np.ndarray) else None
        self.info_object_list = []

    def __iter__(self):
        return iter(self.info_object_list)

    def __len__(self):
        return len(self.info_object_list)

    """
    scanning
    """
    def doScanningSelf(self, *args, **kwargs):
        raise NotImplementedError

    """
    json
    """
    def formatImageObjectsAsDict(self) -> list:
        return [info_object.formatAsDict() for info_object in self.info_object_list]

    """
    visual
    """
    def visualImageScanning(self, canvas) -> np.ndarray:
        for info_object in self.info_object_list:
            canvas = info_object.visual(canvas)
        return canvas

    """
    summary
    """
    def summaryImageScanning(self, **kwargs) -> dict:
        preview_dict = {}
        for info_object in self.info_object_list:
            preview_dict[info_object.identity] = info_object.summaryObjectAsDict(self.bgr, **kwargs)
        return preview_dict


