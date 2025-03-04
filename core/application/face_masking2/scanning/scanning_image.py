
import logging
import copy
import os
import cv2
import numpy as np
import json
from .infoimage_person import InfoImage_Person
from .infoimage_plate import InfoImage_Plate
from .scanning_visor import ScanningVisor
from ....base import XPortrait
from .... import XManager


class InfoImage:
    """
    """
    ObjectDict = dict(person=InfoImage_Person, plate=InfoImage_Plate)

    @staticmethod
    def createFromJson(**kwargs):
        if 'path_in_json' in kwargs and isinstance(kwargs['path_in_json'], str):
            path_in_json = kwargs['path_in_json']
            assert os.path.exists(path_in_json), path_in_json
            info_image = InfoImage(bgr=None)
            for info_dict in json.load(open(path_in_json, 'r')):
                category = info_dict['category']
                info_image.info_object_list.append(InfoImage.ObjectDict[category].createFromDict(info_dict))
            return info_image
        if 'video_info_string' in kwargs and isinstance(kwargs['video_info_string'], str):
            info_string = kwargs['video_info_string']
            assert isinstance(info_string, str)
            info_image = InfoImage(bgr=None)
            for info_dict in json.loads(info_string):
                category = info_dict['category']
                info_image.info_object_list.append(InfoImage.ObjectDict[category].createFromDict(info_dict))
            return info_image
        raise NotImplementedError('both "path_in_json" and "video_info_string" not in kwargs')

    """
    """
    def __init__(self, bgr):
        self.source_bgr = np.copy(bgr) if isinstance(bgr, np.ndarray) else None
        self.info_object_list = []
        self.scanning_handle_dict = dict(person=self.doScanningPersons, plate=self.doScanningPlates)
        # just for person
        self.cache = bgr if bgr is None else \
            XPortrait(bgr, rotations=[0, 180], detect_handle='SDK')  # 0, 90, 180, 270; InsightFace

    def __iter__(self):
        return iter(self.info_object_list)

    def __len__(self):
        return len(self.info_object_list)

    @property
    def bgr(self):
        assert isinstance(self.source_bgr, np.ndarray), self.source_bgr
        return self.source_bgr

    """
    """
    def doScanning(self, schedule_call, category_list):
        assert isinstance(self.cache, XPortrait)
        assert len(category_list) > 0, category_list
        if schedule_call is not None:
            schedule_call('扫描图片-运行中', None)
        if len(self.info_object_list) > 0:
            raise ValueError('scanning had already done')
        for each in category_list:
            self.scanning_handle_dict[each]()
        return len(self.info_object_list)

    def doScanningPersons(self):
        current_len = len(self.info_object_list)
        for n in range(self.cache.number):
            info_person = InfoImage_Person(
                current_len + n + 1, self.cache.box[n, :], self.cache.points[n, :, :],
                self.cache.landmark[n, :, :], self.cache.angles[n])
            self.info_object_list.append(info_person)
        return self.cache.number

    def doScanningPlates(self):
        result = XManager.getModules('ultralytics')['yolo8s-plate.pt'](self.bgr, verbose=False)[0]
        number = len(result)
        if number > 0:
            labels = np.reshape(np.round(result.boxes.cls.cpu().numpy()).astype(np.int32), (-1,))
            scores = np.reshape(result.boxes.conf.cpu().numpy().astype(np.float32), (-1,))
            boxes = np.reshape(np.round(result.boxes.xyxy.cpu().numpy()).astype(np.int32), (-1, 4,))
            current_len = len(self.info_object_list)
            for n in range(number):
                info_plate = InfoImage_Plate(
                    current_len + n + 1, labels[n], scores[n], boxes[n], np.zeros(shape=(4, 2), dtype=np.int32))
                self.info_object_list.append(info_plate)
        return number

    """
    """
    def visualScanning(self) -> np.ndarray:
        canvas = np.copy(self.cache.bgr)
        for info_object in self.info_object_list:
            if isinstance(info_object, InfoImage_Person):
                canvas = ScanningVisor.visualSinglePerson(
                    canvas, info_object.identity, info_object.box, key_points=info_object.landmark)
            if isinstance(info_object, InfoImage_Plate):
                canvas = ScanningVisor.visualSinglePlate(
                    canvas, info_object.identity, info_object.classification, info_object.box)
        return canvas

    def saveVisualScanning(self, path_out_image):
        if isinstance(path_out_image, str):
            cv2.imwrite(path_out_image, self.visualScanning())

    """
    """
    def formatAsJson(self):
        info_dict_list = []
        for info_object in self.info_object_list:
            info_dict_list.append(info_object.formatAsDict())
        return json.dumps(info_dict_list, indent=4)

    def saveAsJson(self, path_out_json, schedule_call):
        if isinstance(path_out_json, str):
            schedule_call('扫描图片-后处理', None)
            open(path_out_json, 'w').write(self.formatAsJson())

    def getInfoAsJson(self, *args, **kwargs) -> str:
        return self.formatAsJson()

    """
    """
    def getPreviewSummaryAsDict(self, size=256, auto_rot=False) -> dict:
        preview_dict = {}
        for info_object in self.info_object_list:
            preview_dict[info_object.identity] = info_object.summaryAsDict(
                self.bgr, size=size, auto_rot=auto_rot)
        return preview_dict


