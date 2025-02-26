
import logging
import copy
import os
import cv2
import numpy as np
import json
from ..helper.angle_helper import AngleHelper
from .scanning_visor import ScanningVisor
from ....base import XPortrait
from ....geometry import Rectangle, GeoFunction


class InfoImage_Person:
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
            identity=self.identity,
            box=self.box.tolist(),
            points=self.points.tolist(),
            landmark=self.landmark.tolist(),
            angle=self.angle,
        )

    @staticmethod
    def createFromDict(info_dict):
        assert isinstance(info_dict, dict)
        return InfoImage_Person(
            int(info_dict['identity']),
            np.reshape(np.array(info_dict['box'], dtype=np.int32), (4,)),
            np.reshape(np.array(info_dict['points'], dtype=np.int32), (5, 2)),
            np.reshape(np.array(info_dict['landmark'], dtype=np.int32), (68, 2)),
            int(info_dict['angle']),
        )


class InfoImage:
    """
    """
    @staticmethod
    def createFromJson(**kwargs):
        if 'path_in_json' in kwargs and isinstance(kwargs['path_in_json'], str):
            path_in_json = kwargs['path_in_json']
            assert os.path.exists(path_in_json), path_in_json
            info_image = InfoImage(bgr=None)
            for info_dict in json.load(open(path_in_json, 'r')):
                info_image.info_person_list.append(InfoImage_Person.createFromDict(info_dict))
            return info_image
        if 'video_info_string' in kwargs and isinstance(kwargs['video_info_string'], str):
            info_string = kwargs['video_info_string']
            assert isinstance(info_string, str)
            info_image = InfoImage(bgr=None)
            for info_dict in json.loads(info_string):
                info_image.info_person_list.append(InfoImage_Person.createFromDict(info_dict))
            return info_image
        raise NotImplementedError('both "path_in_json" and "video_info_string" not in kwargs')

    """
    """
    def __init__(self, bgr):
        self.cache = bgr if bgr is None else \
            XPortrait(bgr, rotations=[0, 180], detect_handle='InsightFace')  # 0, 90, 180, 270; InsightFace
        self.info_person_list = []  # [InfoPerson,]

    def __iter__(self):
        return iter(self.info_person_list)

    def __len__(self):
        return len(self.info_person_list)

    @property
    def bgr(self):
        assert isinstance(self.cache, XPortrait)
        return self.cache.bgr

    """
    """
    def doScanning(self, schedule_call):
        assert isinstance(self.cache, XPortrait)
        if schedule_call is not None:
            schedule_call('扫描图片-运行中', None)
        if len(self.info_person_list) > 0:
            raise ValueError('scanning had already done')
        for n in range(self.cache.number):
            info_person = InfoImage_Person(
                n + 1, self.cache.box[n, :], self.cache.points[n, :, :], self.cache.landmark[n, :, :], self.cache.angles[n])
            self.info_person_list.append(info_person)
        return len(self.info_person_list)

    def visualScanning(self) -> np.ndarray:
        canvas = np.copy(self.cache.bgr)
        for person in self.info_person_list:
            assert isinstance(person, InfoImage_Person)
            canvas = ScanningVisor.visualSinglePerson(canvas, person.identity, person.box, key_points=person.landmark)
        return canvas

    def saveVisualScanning(self, path_out_image):
        if isinstance(path_out_image, str):
            cv2.imwrite(path_out_image, self.visualScanning())

    """
    """
    def formatAsJson(self):
        info_dict_list = []
        for info_person in self.info_person_list:
            assert isinstance(info_person, InfoImage_Person)
            info_dict_list.append(info_person.formatAsDict())
        return json.dumps(info_dict_list, indent=4)

    def saveAsJson(self, path_out_json, schedule_call):
        if isinstance(path_out_json, str):
            schedule_call('扫描图片-后处理', None)
            open(path_out_json, 'w').write(self.formatAsJson())

    def getInfoJson(self, *args, **kwargs) -> str:
        return self.formatAsJson()

    """
    """
    @staticmethod
    def cropPreviewFace(bgr, info_person, size, is_bgr, ext=0.2, auto_rot=False):
        h, w, c = bgr.shape
        lft, top, rig, bot = Rectangle(info_person.landmark).toSquare().expand(ext, ext).clip(0, 0, w, h).asInt()
        crop = np.copy(bgr[top:bot, lft:rig, :])
        crop_rot = GeoFunction.rotateImage(crop, info_person.angle) if auto_rot is True else crop
        resized = cv2.resize(crop_rot, (size, size))
        return resized if is_bgr is True else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

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

    def getIdentityPreviewDict(self, size=256, is_bgr=True, ext=0.2, auto_rot=False) -> dict:
        preview_dict = {}
        for info_person in self.info_person_list:
            assert isinstance(info_person, InfoImage_Person)
            bgr_c, box_c = self.autoRotateForCartoon(self.bgr, info_person.box, info_person.angle)
            preview_dict[info_person.identity] = dict(
                # interface
                image=bgr_c,
                box=box_c.tolist(),
                face=self.cropPreviewFace(self.bgr, info_person, size, is_bgr, ext, auto_rot),
                # for debug
                angle=info_person.angle,
                cartoon_image=bgr_c,
                cartoon_box=box_c.tolist())
        return preview_dict


