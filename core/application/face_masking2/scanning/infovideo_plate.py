
import logging
import copy
import os
import cv2
import numpy as np
import json
import dataclasses
import typing
from ..helper.cursor import AsynchronousCursor
from ..helper.boundingbox import BoundingBox
from ..helper.align_helper import AlignHelper
from ....base import XPortrait
from .... import XManager


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
