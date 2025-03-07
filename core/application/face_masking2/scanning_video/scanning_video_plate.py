
import logging
import cv2
import numpy as np
import json
import dataclasses
from .scanning_video import ScanningVideo
from ..helper import AsynchronousCursor
from ....utils import XVideoReader, XVideoWriter


@dataclasses.dataclass(frozen=False)
class InfoVideo_Plate_Frame:
    frame_index: int
    box_track: np.ndarray  # 4,
    box_copy: bool

    @property
    def key_points(self):
        return None

    @staticmethod
    def fromString(string):
        string_split = string.split(';')
        return InfoVideo_Plate_Frame(
            frame_index=int(string_split[0]),
            box_track=InfoVideo_Plate_Frame.formatSeqToInt(string_split[1], np.int32).reshape(4,),
            box_copy=False)

    @staticmethod
    def formatSeqToInt(string_seq, dtype=None) -> np.ndarray:
        int_list = [int(v) for v in string_seq.split(',')]
        return np.array(int_list, dtype=dtype)

    @staticmethod
    def formatArrayToString(array) -> list:
        int_list = np.reshape(array, (-1)).astype(np.int32).tolist()
        return [str(v) for v in int_list]

    def formatAsString(self) -> str:
        string_box_track = ','.join(self.formatArrayToString(self.box_track))
        info_string = '{};{}'.format(self.frame_index, string_box_track)
        return info_string


class InfoVideo_Plate_Preview:
    """
    """
    ScoreStrategy = 'pose'  # 'pose' or 'size' or 'mix'

    @staticmethod
    def createFromDict(info_dict):
        frame_index = info_dict['frame_index']
        box_track = np.array(info_dict['box_track'], dtype=np.int32).reshape(-1,)  # 4,
        return InfoVideo_Plate_Preview(frame_index, None, box_track)

    def __init__(self, frame_index, frame_bgr, box_track):
        assert len(box_track) == 4, box_track
        self.frame_index = frame_index
        self.frame_bgr = np.copy(frame_bgr) if isinstance(frame_bgr, np.ndarray) else None
        self.box_track = box_track

    def __str__(self):
        return 'frame_index={}, box_track={}'.format(self.frame_index, self.box_track.tolist())

    @staticmethod
    def transformImage(bgr, size, is_bgr) -> np.ndarray:
        resized = cv2.resize(bgr, (size, size))
        return resized if is_bgr is True else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def summaryAsDict(self, as_text) -> dict:
        summary = dict(
            frame_index=int(self.frame_index),
            box_track=self.box_track.tolist(),
        )
        # if size > 0 and isinstance(is_bgr, bool):
        #     assert isinstance(self.frame_bgr, np.ndarray), self.frame_bgr
        #     summary['image'] = np.copy(self.frame_bgr)
        #     summary['box'] = self.box_track.tolist()
        #     summary['preview'] = self.transformImage(self.face_align_cache.bgr, size, is_bgr)
        if as_text is True:
            # TODO: recognize plate to text
            summary['text'] = '苏B-12345'
            lft, top, rig, bot = self.box_track.tolist()
            summary['preview'] = np.copy(self.frame_bgr[top:bot, lft:rig, :])
        return summary


class InfoVideo_Plate:
    """
    """
    CategoryName = 'plate'

    @staticmethod
    def createFromDict(info_dict):
        plate = InfoVideo_Plate(info_dict['identity'], yolo_identity=-1)
        plate.frame_info_list = [InfoVideo_Plate_Frame.fromString(each) for each in info_dict['frame_info_list']]
        plate.preview = InfoVideo_Plate_Preview.createFromDict(info_dict['preview'])
        return plate

    """
    """
    def __init__(self, identity, yolo_identity):
        self.identity = identity
        self.preview = None
        self.activate = True
        self.frame_info_list = []  # [InfoVideo_Frame]
        # for yolo
        self.yolo_identity = yolo_identity

    def __len__(self):
        return len(self.frame_info_list)

    def __str__(self):
        return 'identity={}, yolo_identity={}, frame_length={}, preview({})'.format(
            self.identity, self.yolo_identity, len(self), str(self.preview))

    """
    """
    def setActivate(self, activate):
        self.activate = bool(activate)

    def appendInfo(self, frame_index, frame_bgr, box_track):
        lft, top, rig, bot = box_track
        if np.sum(box_track.astype(np.int32)) != 0 and lft < rig and top < bot:
            self.frame_info_list.append(InfoVideo_Plate_Frame(
                frame_index=frame_index, box_track=box_track, box_copy=False))
        else:
            assert len(self.frame_info_list) > 0
            info_last = self.frame_info_list[-1]
            if info_last.frame_index == frame_index - 1:
                self.frame_info_list.append(InfoVideo_Plate_Frame(
                    frame_index=frame_index, box_track=box_track, box_copy=True))
        self.setIdentityPreview(frame_index, frame_bgr, box_track)

    def formatAsDict(self, with_frame_info=True):
        time_beg = self.frame_info_list[0].frame_index
        time_end = self.frame_info_list[-1].frame_index
        time_len = len(self.frame_info_list)
        preview_dict = self.getPreviewSummary(as_text=False)
        info = dict(
            category=self.CategoryName, identity=self.identity,
            time_beg=time_beg, time_end=time_end, time_len=time_len, preview=preview_dict)
        if with_frame_info is True:
            info['frame_info_list'] = [info.formatAsString() for info in self.frame_info_list]
        return info

    def setIdentityPreview(self, frame_index, frame_bgr, box_track):
        if self.preview is None:
            self.preview = InfoVideo_Plate_Preview(frame_index, frame_bgr, box_track)

    def getPreviewSummary(self, as_text) -> dict:
        assert isinstance(self.preview, InfoVideo_Plate_Preview), self.preview
        return self.preview.summaryAsDict(as_text)

    def checkIdentity(self, face_size_min, num_frame_min) -> bool:
        object_frame_valid = bool(len(self) > num_frame_min)
        return bool(object_frame_valid)

    def getFrameInfoCursor(self, idx_beg, idx_end):
        frame_index_array = np.array([info.frame_index for info in self.frame_info_list], dtype=np.int32)
        idx_beg_arg_where = np.argwhere(idx_beg <= frame_index_array)
        idx_end_arg_where = np.argwhere(frame_index_array <= idx_end)
        if len(idx_beg_arg_where) > 0 and len(idx_end_arg_where) > 0:
            beg_pre_idx = int(idx_beg_arg_where[0])
            end_aft_idx = int(idx_end_arg_where[-1])
            return AsynchronousCursor(self.frame_info_list, beg_pre_idx, end_aft_idx)
        else:
            return AsynchronousCursor(self.frame_info_list, 0, 0)

    def interpolateFramesAtGap(self, max_frame_gap):
        assert isinstance(max_frame_gap, int) and max_frame_gap > 0, max_frame_gap
        n = 0
        while n < len(self.frame_info_list)-1:
            info1 = self.frame_info_list[n]
            info2 = self.frame_info_list[n+1]
            if info1.frame_index+1 < info2.frame_index and info1.frame_index+max_frame_gap >= info2.frame_index:
                index_beg = info1.frame_index + 1
                index_end = info2.frame_index
                array = np.linspace(0, 1, index_end - index_beg + 2, dtype=np.float32)[1:-1]
                for i, idx in enumerate(range(index_beg, index_end)):
                    r = float(array[i])
                    frame_index = idx
                    box_track = (r * info1.box_track + (1 - r) * info2.box_track).astype(np.int32)
                    self.frame_info_list.insert(n+1+i, InfoVideo_Plate_Frame(
                        frame_index=frame_index, box_track=box_track, box_copy=True))
                    logging.info('interpolate frames: identity-{}, insert {} into ({}, {})'.format(
                        self.identity, frame_index, index_beg, index_end))
                n += index_end - index_beg
            else:
                n += 1


class ScanningVideo_Plate(ScanningVideo):
    """
    """
    IOU_Threshold = 0.3

    @staticmethod
    def createVideoScanning(**kwargs):
        if 'path_in_json' in kwargs and isinstance(kwargs['path_in_json'], str):
            info_video = ScanningVideo_Plate()
            info_video.object_identity_history = [InfoVideo_Plate.createFromDict(info) for info in json.load(open(kwargs['path_in_json'], 'r'))]
            return info_video
        if 'info_string' in kwargs and isinstance(kwargs['info_string'], str):
            info_video = ScanningVideo_Plate()
            info_video.object_identity_history = [InfoVideo_Plate.createFromDict(info) for info in json.loads(kwargs['info_string'])]
            return info_video
        if 'objects_list' in kwargs and isinstance(kwargs['objects_list'], list):
            info_video = ScanningVideo_Plate()
            info_video.object_identity_history = [InfoVideo_Plate.createFromDict(info) for info in kwargs['objects_list']]
            return info_video
        raise NotImplementedError('"objects_list", "path_in_json", "info_string" not in kwargs')

    def __init__(self, **kwargs):
        super(ScanningVideo_Plate, self).__init__(**kwargs)
        # tracking config
        self.tracking_model = 'yolo8s-plate.pt'
        self.tracking_config = dict(
            persist=True, conf=0.3, iou=0.5, tracker='bytetrack.yaml', verbose=False)

    """
    tracking pipeline
    """
    def finishTracking(self):
        self.updateObjectList([])
        self.interpolateFrame()

    def updateWithYOLO(self, frame_index, frame_bgr, result):
        person_list_new = []
        number = len(result)
        if number > 0 and result.boxes.id is not None:
            # classify = np.reshape(np.round(result.boxes.cls.cpu().numpy()).astype(np.int32), (-1,))
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
    interpolate frames
    """
    def interpolateFrame(self, max_frame_gap=16):
        # interpolate frames into inner gap
        for n, plate in enumerate(self.object_identity_history):
            assert isinstance(plate, InfoVideo_Plate), plate
            plate.interpolateFramesAtGap(max_frame_gap)

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
    visual
    """
    def visualVideoScanning(self, frame_index, frame_canvas, cursor_list, **kwargs):
        from ..visor import Visor
        for n, (info_object, cursor) in enumerate(cursor_list):
            info: InfoVideo_Plate_Frame = cursor.current()
            if info.frame_index == frame_index:
                Visor.visualSinglePlate(frame_canvas, info_object.identity, '', box=info.box_track)
                cursor.next()
        return frame_canvas
