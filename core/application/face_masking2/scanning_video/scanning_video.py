
import logging
import copy
import os
import cv2
import numpy as np
import json
import pickle
from ....utils import XContextTimer, XVideoReader
from .... import XManager


class ScanningVideo:
    """
    """
    IOU_Threshold = 0.3

    @classmethod
    def createFromDict(cls, **kwargs):
        raise NotImplementedError

    def __init__(self, **kwargs):
        # video reader
        self.reader = None
        self.object_identity_seq = kwargs.pop('object_identity_seq', -1)
        self.object_num_max = kwargs.pop('person_num_max', -1)
        self.object_identity_history = []
        self.object_list_current = []
        # tracking config
        self.tracking_model = 'yolo11x-pose'
        self.tracking_classes = []
        self.tracking_config = dict(
            persist=True, conf=0.5, iou=0.7, classes=self.tracking_classes, tracker='bytetrack.yaml', verbose=False)

    """
    tracking
    """
    def doScanning(self, path_in_video, schedule_call, online=True):
        if isinstance(self.reader, XVideoReader) is False:
            self.reader = XVideoReader(path_in_video)
            assert self.reader.isOpen(), path_in_video
        # tracking
        if online is True:
            return self.doScanningOnline(path_in_video, schedule_call)
        else:
            return self.doScanningOffline(path_in_video, schedule_call)

    def doScanningOffline(self, path_in_video, schedule_call):
        with XContextTimer(True):
            module = XManager.getModules('ultralytics')[self.tracking_model]
            result = module.track(path_in_video, **self.tracking_config)
            assert isinstance(result, list)
            # review the result
            self.reader.resetPositionByIndex(0)
            for n in range(len(result)):
                flag, frame_bgr = self.reader.read()
                if flag is True:
                    self.updateWithYOLO(n, frame_bgr, result[n])
                schedule_call('扫描视频-运行中', float((n + 1) / len(self.reader)))
            self.finishTracking()
            XManager.getModules('ultralytics').resetTracker(self.tracking_model)
            return len(self.object_identity_history)

    def doScanningOnline(self, path_in_video, schedule_call):
        module = XManager.getModules('ultralytics')[self.tracking_model]
        self.reader.resetPositionByIndex(0)
        for n, frame_bgr in enumerate(self.reader):
            result = module.track(frame_bgr, **self.tracking_config)[0]
            self.updateWithYOLO(n, frame_bgr, result)
            schedule_call('扫描视频-运行中', float((n + 1) / len(self.reader)))
        self.finishTracking()
        XManager.getModules('ultralytics').resetTracker(self.tracking_model)
        return len(self.object_identity_history)

    def updateWithYOLO(self, frame_index, frame_bgr, result):
        raise NotImplementedError

    def updateObjectList(self, person_list_new):
        for n in range(len(self.object_list_current)):
            obj = self.object_list_current.pop()
            obj.setActivate(False)
            self.object_identity_history.append(obj)
        self.object_list_current += person_list_new

    def finishTracking(self):
        self.updateObjectList([])

    """
    preview
    """
    def formatVideoObjectsAsDict(self, with_frame_info=False) -> list:
        sorted_history = self.getSortedHistory()
        return [info_object.formatAsDict(with_frame_info) for info_object in sorted_history]

    def getSortedHistory(self):
        return sorted(self.object_identity_history, key=lambda info_object: info_object.identity)

    """
    masking
    """
    @staticmethod
    def getInfoCursorList(info_object_list, num_frames, num_split, min_frames=0):
        object_remain_list = [info_object for info_object in info_object_list if len(info_object.frame_info_list) > min_frames]
        each_len = int(np.ceil(num_frames / num_split))
        beg_end_list = [(n*each_len, min((n+1)*each_len-1, num_frames)) for n in range(num_split)]
        cursor_list_all = []
        for n, (beg, end) in enumerate(beg_end_list):
            cursor_list = []
            for info_object in object_remain_list:
                cursor = info_object.getFrameInfoCursor(beg, end)
                if cursor.valid() is True:
                    cursor_list.append((info_object, cursor))
            cursor_list_all.append(dict(beg=beg, end=end, cursor_list=cursor_list))
        return cursor_list_all

    @staticmethod
    def getPreviewAsDict(info_object_list) -> dict:
        assert isinstance(info_object_list, list), info_object_list
        return {info_object.identity: info_object.preview for info_object in info_object_list}

    """
    summary
    """
    def getPreviewSummaryAsDict(self, *args, **kwargs):
        raise NotImplementedError

    """
    visual
    """
    def visualVideoScanning(self, *args, **kwargs):
        raise NotImplementedError

    def saveVisualScanning(self, path_in_video, path_out_video, schedule_call, **kwargs):
        raise NotImplementedError
