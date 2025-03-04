
import logging
import copy
import os
import cv2
import numpy as np
from ....geometry import Rectangle


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

    def summaryAsDict(self, bgr, *args, **kwargs):
        summary = dict(
            category=self.CategoryName,
            image=np.copy(bgr),
            box=self.box.tolist(),
            preview=self.cropPreview(bgr))
        return summary
