
import logging
import copy
import os
import cv2
import numpy as np
import json
from .scanning_image import ScanningImage
from ..visor import Visor
from ....geometry import Rectangle
from .... import XManager


class InfoImage_Plate:
    """
    """
    CategoryName = 'plate'
    LabelString = ['single', 'double']

    """
    """
    def __init__(self, identity, label, score, box, points):
        self.identity = identity
        self.label = int(label)
        self.score = float(score)
        self.box = np.round(box).astype(np.int32)
        self.points = np.round(points).astype(np.int32)

    @property
    def classification(self) -> str:
        return self.LabelString[self.label]

    def formatAsDict(self) -> dict:
        return dict(
            category=self.CategoryName,
            identity=self.identity,
            label=self.label,
            score=self.score,
            box=self.box.tolist(),
            points=self.points.tolist()
        )

    @staticmethod
    def createFromDict(info_dict):
        assert isinstance(info_dict, dict)
        assert info_dict['category'] == InfoImage_Plate.CategoryName, \
            (info_dict['category'], InfoImage_Plate.CategoryName)
        return InfoImage_Plate(
            identity=int(info_dict['identity']),
            label=int(info_dict['label']),
            score=float(info_dict['score']),
            box=np.reshape(np.array(info_dict['box'], dtype=np.int32), (4,)),
            points=np.zeros(shape=(4, 2), dtype=np.int32),
        )

    def cropPreview(self, bgr):
        h, w, c = bgr.shape
        lft, top, rig, bot = Rectangle(self.box).clip(0, 0, w, h).asInt()
        return bgr[top:bot, lft:rig]

    def summaryObjectAsDict(self, bgr, as_text=True):
        summary = dict(
            category=self.CategoryName,
            preview=self.cropPreview(bgr),)
        if as_text is True:
            # TODO: recognize plate to text
            summary['text'] = '苏B-12345'
        return summary

    """
    visual
    """
    def visual(self, canvas):
        return Visor.visualSinglePlate(canvas, self.identity, self.classification, self.box)


class ScanningImage_Plate(ScanningImage):
    """
    """
    @classmethod
    def getObjectClass(cls):
        return InfoImage_Plate

    """
    """
    def __init__(self, *args, **kwargs):
        super(ScanningImage_Plate, self).__init__(*args, **kwargs)

    def doScanningSelf(self, identity_seq):
        result = XManager.getModules('ultralytics')['yolo8s-plate.pt'](self.bgr, verbose=False)[0]
        number = len(result)
        if number > 0:
            labels = np.reshape(np.round(result.boxes.cls.cpu().numpy()).astype(np.int32), (-1,))
            scores = np.reshape(result.boxes.conf.cpu().numpy().astype(np.float32), (-1,))
            boxes = np.reshape(np.round(result.boxes.xyxy.cpu().numpy()).astype(np.int32), (-1, 4,))
            for n in range(number):
                info_plate = InfoImage_Plate(
                    identity_seq + n + 1, labels[n], scores[n], boxes[n], np.zeros(shape=(4, 2), dtype=np.int32))
                self.info_object_list.append(info_plate)
        return number
