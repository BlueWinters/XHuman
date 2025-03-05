
import logging
import os
import cv2
import numpy as np
import json
import pickle
from .masking_function import MaskingFunction
from .helper.masking_helper import MaskingHelper
from .scanning_image import ScanningImage, ScanningImage_Person, ScanningImage_Plate
from ...utils import Resource, XContextTimer


class LibMaskingImage:
    """
    """
    ScanningClassDict = dict(person=ScanningImage_Person, plate=ScanningImage_Plate)

    """
    """
    def __init__(self, *args, **kwargs):
        self.scanning_dict = dict()
        self.info_string = None

    """
    scanning
    """
    def doScanning(self, bgr, category_list, schedule_call):
        schedule_call('扫描图片-运行中', None)
        assert len(category_list) > 0, category_list
        objects_counter = 0
        for category in category_list:
            scanning = self.ScanningClassDict[category](bgr=bgr)
            objects_counter += scanning.doScanningSelf(objects_counter)
            self.scanning_dict[category] = scanning
        return self.scanning_dict

    def formatAsJson(self, path_out_json):
        if isinstance(self.info_string, str) is False:
            format_list = list()
            for category, scanning in self.scanning_dict.items():
                assert isinstance(scanning, ScanningImage), scanning
                format_list.append(dict(category=category, objects=scanning.formatImageObjectsAsDict()))
            self.info_string = json.dumps(format_list, indent=4)
            if isinstance(path_out_json, str) and path_out_json.endswith('.json'):
                open(path_out_json, 'w').write(self.info_string)

    def visualScanning(self, bgr, path_out_image):
        if isinstance(path_out_image, str):
            canvas = np.copy(bgr)
            assert len(self.scanning_dict) > 0
            for category, scanning in self.scanning_dict.items():
                assert isinstance(scanning, ScanningImage), scanning
                canvas = scanning.visualImageScanning(canvas)
            cv2.imwrite(path_out_image, canvas)

    def scanningImage(self, bgr, **kwargs):
        category_list = kwargs.pop('category_list', ['person'])
        path_out_json = kwargs.pop('path_out_json', None)
        schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
        path_out_image = kwargs.pop('visual_scanning', None)
        # scanning
        self.doScanning(bgr, category_list, schedule_call)
        # format
        self.formatAsJson(path_out_json)
        # visual
        self.visualScanning(bgr, path_out_image)

    """
    preview
    """
    def savePreviewAsPickle(self, path_out_pkl):
        preview_dict = self.getImageSummaryAsDict()
        pickle.dump(preview_dict, open(path_out_pkl, 'wb'))

    @staticmethod
    def loadPreviewFromPickle(path_in_pkl):
        return pickle.load(open(path_in_pkl, 'rb'))

    def getImageSummaryAsDict(self, **kwargs):
        summary_dict = dict()
        for category, scanning in self.scanning_dict.items():
            parameters = kwargs.pop(category, dict())
            summary_dict.update(scanning.summaryImageScanning(**parameters))
        return summary_dict

    """
    json
    """
    def getImageInfoAsJson(self, *args, **kwargs) -> str:
        assert isinstance(self.info_string, str), self.info_string
        return self.info_string

    """
    masking
    """
    @staticmethod
    def loadScanningDict(**kwargs):
        if 'path_in_json' in kwargs and isinstance(kwargs['path_in_json'], str):
            scanning_dict = dict()
            for info in json.load(open(kwargs.pop('path_in_json'), 'r')):
                category = info['category']
                info_objects = info['objects']
                scanning_dict[category] = LibMaskingImage.ScanningClassDict[category].createImageScanning(objects_list=info_objects)
            return scanning_dict
        if 'info_string' in kwargs and isinstance(kwargs['info_string'], str):
            scanning_dict = dict()
            for info in json.loads(kwargs.pop('info_string')):
                category = info['category']
                info_objects = info['objects']
                scanning_dict[category] = LibMaskingImage.ScanningClassDict[category].createImageScanning(objects_list=info_objects)
            return scanning_dict
        raise NotImplementedError('both "path_in_json" and "info_string" not in kwargs')

    @staticmethod
    def maskingImage(path_image_or_bgr, options_dict, **kwargs):
        schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
        with_hair = kwargs.pop('with_hair', True)

        bgr = cv2.imread(path_image_or_bgr, cv2.IMREAD_COLOR) if isinstance(path_image_or_bgr, str) \
            else np.array(path_image_or_bgr, dtype=np.uint8)
        with XContextTimer(True):
            scanning_dict = LibMaskingImage.loadScanningDict(**kwargs)
            objects_list = []
            for category, scanning in scanning_dict.items():
                objects_list.extend(scanning.info_object_list)
            # MaskingHelper.getPortraitMaskingWithInfoImage(
            #     bgr, info_image, options_dict, with_hair=with_hair, expand=0.8)
            MaskingHelper.getPortraitMaskingWithInfoImagePlus(
                bgr, objects_list, options_dict, with_hair=with_hair, expand=0.8)
            canvas_bgr = np.copy(bgr)
            for n, info_object in enumerate(objects_list):
                if info_object.identity in options_dict:
                    masking_option = options_dict[info_object.identity]
                    canvas_bgr = MaskingFunction.maskingImage(bgr, canvas_bgr, info_object, masking_option)
                    schedule_call('打码图片', float((n + 1) / len(objects_list)))
            return canvas_bgr



