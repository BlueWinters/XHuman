
import logging
import copy
import os
import cv2
import numpy as np
import json
from .scanning_image import ScanningImage
from ..helper import AngleHelper
from ..visor import Visor
from ....geometry import Rectangle, GeoFunction
from ....base import XPortrait


class InfoImage_Person:
    """
    """
    CategoryName = 'person'

    """
    """
    def __init__(self, identity, box, points, landmark, angle):
        self.identity = identity
        self.box = np.round(box).astype(np.int32)
        self.points = np.round(points).astype(np.int32)
        self.landmark = np.round(landmark).astype(np.int32)
        self.angle = AngleHelper.getAngleRollByLandmark(self.landmark)

    def formatAsDict(self) -> dict:
        return dict(
            category=self.CategoryName,
            identity=self.identity,
            box=self.box.tolist(),
            points=self.points.tolist(),
            landmark=self.landmark.tolist(),
            angle=self.angle,
        )

    @staticmethod
    def createFromDict(info_dict):
        assert isinstance(info_dict, dict)
        assert info_dict['category'] == InfoImage_Person.CategoryName, \
            (info_dict['category'], InfoImage_Person.CategoryName)
        return InfoImage_Person(
            identity=int(info_dict['identity']),
            box=np.reshape(np.array(info_dict['box'], dtype=np.int32), (4,)),
            points=np.reshape(np.array(info_dict['points'], dtype=np.int32), (5, 2)),
            landmark=np.reshape(np.array(info_dict['landmark'], dtype=np.int32), (68, 2)),
            angle=int(info_dict['angle']),
        )

    @staticmethod
    def autoRotateForCartoon(bgr, box, angle):
        h, w, c = bgr.shape
        box_rot = GeoFunction.rotateBoxes(box, angle, h, w)
        bgr_rot = GeoFunction.rotateImage(bgr, angle)
        return bgr_rot, box_rot.astype(np.int32)

    @staticmethod
    def transformImage(bgr, size, is_bgr):
        resized = cv2.resize(bgr, (size, size))
        return resized if is_bgr is True else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def cropPreviewFace(self, bgr, size, is_bgr=True, ext=0.2, auto_rot=False):
        h, w, c = bgr.shape
        lft, top, rig, bot = Rectangle(self.landmark).toSquare().expand(ext, ext).clip(0, 0, w, h).asInt()
        crop = np.copy(bgr[top:bot, lft:rig, :])
        crop_rot = GeoFunction.rotateImage(crop, self.angle) if auto_rot is True else crop
        resized = cv2.resize(crop_rot, (size, size))
        return resized if is_bgr is True else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def summaryObjectAsDict(self, bgr, size=256, is_bgr=True, ext=0.2, auto_rot=False):
        bgr_c, box_c = self.autoRotateForCartoon(bgr, self.box, self.angle)
        summary = dict(
            # interface
            category=self.CategoryName,
            preview=self.cropPreviewFace(bgr, size, is_bgr, ext, auto_rot),
            # for cartoon
            angle=self.angle,
            cartoon_bgr=bgr_c,
            cartoon_box=box_c.tolist())
        return summary

    """
    visual
    """
    def visual(self, canvas):
        return Visor.visualSinglePerson(canvas, self.identity, self.box, key_points=self.landmark)


class ScanningImage_Person(ScanningImage):
    """
    """
    @classmethod
    def getObjectClass(cls):
        return InfoImage_Person

    """
    """
    def __init__(self, *args, **kwargs):
        super(ScanningImage_Person, self).__init__(*args, **kwargs)
        self.cache = self.bgr if self.bgr is None else \
            XPortrait(self.bgr, rotations=[0, 180], detect_handle='InsightFace')  # 0, 90, 180, 270; InsightFace

    def doScanningSelf(self, identity_seq):
        assert isinstance(self.cache, XPortrait)
        for n in range(self.cache.number):
            info_person = InfoImage_Person(
                identity_seq + n + 1, self.cache.box[n, :], self.cache.points[n, :, :],
                self.cache.landmark[n, :, :], self.cache.angles[n])
            self.info_object_list.append(info_person)
        return self.cache.number
