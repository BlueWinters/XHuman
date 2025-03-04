
import logging
import copy
import os
import cv2
import numpy as np
import json
import typing
import pickle
from .scanning_video import ScanningVideo
from .infovideo_plate import InfoVideo_Plate_Frame, InfoVideo_Plate, InfoVideo_Plate_Preview
from ..helper import AsynchronousCursor, BoundingBox, AlignHelper
from ....utils import XContextTimer, XVideoReader, XVideoWriter
from .... import XManager


class ScanningVideo_Plate(ScanningVideo):
    """
    """
    IOU_Threshold = 0.3

    @staticmethod
    def createFromDict(**kwargs):
        if 'path_in_json' in kwargs and isinstance(kwargs['path_in_json'], str):
            info_video = ScanningVideo_Plate()
            info_video.object_identity_history = [InfoVideo_Plate.createFromDict(info) for info in json.load(open(kwargs['path_in_json'], 'r'))]
            return info_video
        if 'video_info_string' in kwargs and isinstance(kwargs['video_info_string'], str):
            info_video = ScanningVideo_Plate()
            info_video.object_identity_history = [InfoVideo_Plate.createFromDict(info) for info in json.loads(kwargs['video_info_string'])]
            return info_video
        if 'objects_list' in kwargs and isinstance(kwargs['objects_list'], list):
            info_video = ScanningVideo_Plate()
            info_video.object_identity_history = [InfoVideo_Plate.createFromDict(info) for info in kwargs['objects_list']]
            return info_video
        raise NotImplementedError('"objects_list", "path_in_json", "video_info_string" not in kwargs')

    def __init__(self, **kwargs):
        super(ScanningVideo_Plate, self).__init__(**kwargs)
        # tracking config
        self.tracking_model = 'yolo8s-plate.pt'
        self.tracking_config = dict(
            persist=True, conf=0.3, iou=0.5, tracker='bytetrack.yaml', verbose=False)

    """
    tracking pipeline
    """
    def updateWithYOLO(self, frame_index, frame_bgr, result):
        person_list_new = []
        number = len(result)
        if number > 0 and result.boxes.id is not None:
            classify = np.reshape(np.round(result.boxes.cls.cpu().numpy()).astype(np.int32), (-1,))
            scores = np.reshape(result.boxes.conf.cpu().numpy().astype(np.float32), (-1,))
            identity = np.reshape(result.boxes.id.cpu().numpy().astype(np.int32), (-1,))
            boxes = np.reshape(np.round(result.boxes.xyxy.cpu().numpy()).astype(np.int32), (-1, 4,))
            # update common(tracking without lose)
            index_list = np.argsort(scores)[::-1].tolist()
            for i, n in enumerate(index_list):
                cur_one_box_tracker = boxes[n, :]  # 4: lft,top,rig,bot
                index3 = self.matchPreviousByIdentity(self.object_list_current, int(identity[n]))
                if index3 != -1:
                    object_cur = self.object_list_current.pop(index3)
                    assert isinstance(object_cur, InfoVideo_Plate)
                    object_cur.appendInfo(frame_index, frame_bgr, cur_one_box_tracker)
                    person_list_new.append(object_cur)
                    continue
                index4 = self.matchPreviousByIdentity(self.object_identity_history, int(identity[n]))
                if index4 != -1:
                    object_cur = self.object_identity_history.pop(index4)
                    assert isinstance(object_cur, InfoVideo_Plate)
                    object_cur.appendInfo(frame_index, frame_bgr, cur_one_box_tracker)
                    person_list_new.append(object_cur)
                    continue
                # create new person
                if np.sum(cur_one_box_tracker) > 0:
                    # create a new person
                    person_new = self.createNewObject(
                        frame_index, frame_bgr, int(identity[n]), cur_one_box_tracker)
                    if person_new is not None:
                        person_list_new.append(person_new)
                else:
                    # skip the unreliable box
                    pass
        # update current person list
        self.updateObjectList(person_list_new)

    @staticmethod
    def matchPreviousByIdentity(person_list, identity):
        for n, info_object in enumerate(person_list):
            assert isinstance(info_object, InfoVideo_Plate)
            if info_object.yolo_identity == identity:
                return n
        return -1

    def createNewObject(self, frame_index, frame_bgr, track_identity, box_track):
        self.object_identity_seq += 1
        info_object = InfoVideo_Plate(self.object_identity_seq, track_identity)
        info_object.appendInfo(frame_index, frame_bgr, box_track)
        info_object.setIdentityPreview(frame_index, frame_bgr, box_track)
        return info_object

    """
    summary
    """
    def getPreviewSummaryAsDict(self, *args, **kwargs):
        as_text = kwargs.pop('as_text', True)
        preview_dict = {}
        for info_object in self.getSortedHistory():
            assert isinstance(info_object, InfoVideo_Plate), info_object
            logging.info(str(info_object))
            preview_dict[info_object.identity] = info_object.getPreviewSummary(as_text)
        return preview_dict

    """
    visual tracking
    """
    def saveVisualScanning(self, path_in_video, path_out_video, schedule_call, **kwargs):
        from .scanning_visor import ScanningVisor
        if isinstance(path_out_video, str):
            schedule_call('扫描视频-可视化追踪', None)
            reader = XVideoReader(path_in_video)
            writer = XVideoWriter(reader.desc(True))
            writer.open(path_out_video)
            writer.visual_index = True
            cursor_list = [(info_object, AsynchronousCursor(info_object.frame_info_list))
                for info_object in self.object_identity_history]
            for frame_index, frame_bgr in enumerate(reader):
                canvas = frame_bgr
                for n, (person, cursor) in enumerate(cursor_list):
                    info = cursor.current()
                    if info.frame_index == frame_index:
                        canvas = ScanningVisor.visualSinglePerson(canvas, person.identity, info.box_track)
                        cursor.next()
                writer.write(canvas)
            writer.release(reformat=True)
