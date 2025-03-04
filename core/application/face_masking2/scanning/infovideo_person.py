
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
class InfoVideo_Person_Frame:
    frame_index: int
    box_face: np.ndarray  # 4,
    box_track: np.ndarray  # 4,
    key_points_xy: np.ndarray  # 5,2
    key_points_score: np.ndarray  # 5,
    box_copy: bool

    @property
    def key_points(self) -> np.ndarray:
        return np.concatenate([self.key_points_xy, self.key_points_score[:, None]], axis=1)  # 5,3

    @staticmethod
    def fromString(string):
        string_split = string.split(';')
        return InfoVideo_Person_Frame(
            frame_index=int(string_split[0]),
            box_face=InfoVideo_Person_Frame.formatSeqToInt(string_split[1], np.int32).reshape(-1, ),
            key_points_xy=InfoVideo_Person_Frame.formatSeqToInt(string_split[2], np.int32).reshape(-1, 2),
            key_points_score=InfoVideo_Person_Frame.formatSeqToInt(string_split[3], np.float32).reshape(-1) / 100.,
            box_track=np.zeros(shape=(4,), dtype=np.int32),
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
        string_box_face = ','.join(self.formatArrayToString(self.box_face))
        string_points_xy = ','.join(self.formatArrayToString(self.key_points_xy))
        string_points_score = ','.join(self.formatArrayToString(np.round(self.key_points_score*100)))  # float:[0,1] --> int:[0,100]
        info_string = '{};{};{};{}'.format(self.frame_index, string_box_face, string_points_xy, string_points_score)
        return info_string


class InfoVideo_Person_Preview:
    """
    """
    ScoreStrategy = 'pose'  # 'pose' or 'size' or 'mix'

    @staticmethod
    def createFromDict(info_dict):
        frame_index = info_dict['frame_index']
        face_key_points_xy = np.array(info_dict['face_key_points_xy'], np.float32).reshape(5, 2)
        face_key_points_score = np.array(info_dict['face_key_points_score'], np.float32).reshape(5, 1) / 100.
        face_key_points = np.concatenate([face_key_points_xy, face_key_points_score], axis=1)
        face_box = np.array(info_dict['face_box'], dtype=np.int32).reshape(-1,)
        return InfoVideo_Person_Preview(frame_index, None, face_box, face_key_points, None)

    def __init__(self, frame_index, frame_bgr, face_box, face_key_points, face_align_cache):
        assert len(face_box) == 4, face_box
        assert face_key_points.shape[0] == 5 and face_key_points.shape[1] == 3, face_key_points.shape
        self.frame_index = frame_index
        self.frame_bgr = np.copy(frame_bgr)
        self.face_box = face_box
        self.face_key_points = face_key_points
        self.face_align_cache = face_align_cache
        # others
        self.face_size = self.getFaceSize(face_box)
        self.face_key_points_xy = face_key_points[:, :2]  # 5,2
        self.face_key_points_score = face_key_points[:, 2]  # 5,
        self.face_key_points_score_sum = float(np.sum(self.face_key_points_score))
        self.face_valid_points_num = int(np.count_nonzero((self.face_key_points_score > 0.5).astype(np.int32)))

    def __str__(self):
        return 'frame_index={}, face_box={}, face_valid_points_num={}, face_score={}, face_size={}'.format(
            self.frame_index, self.face_box.tolist(), int(self.face_valid_points_num), int(self.face_score), int(self.face_size),)

    @staticmethod
    def transformImage(bgr, size, is_bgr) -> np.ndarray:
        resized = cv2.resize(bgr, (size, size))
        return resized if is_bgr is True else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    @staticmethod
    def getFaceScore(cache, face_box) -> int:
        if isinstance(cache, XPortrait):
            # 1.select the front face
            if InfoVideo_Person_Preview.ScoreStrategy == 'pose':
                return 180 - int(np.sum(np.abs(cache.radian[0, :2]) * 180 / np.pi))
            # 2.select the biggest face
            if InfoVideo_Person_Preview.ScoreStrategy == 'size':
                return InfoVideo_Person_Preview.getFaceSize(face_box)
            # 3.
            if InfoVideo_Person_Preview.ScoreStrategy == 'mix':
                # TODO
                pose = int(np.sum(np.abs(cache.radian[0, :2]) * 180 / np.pi))
                size = InfoVideo_Person_Preview.getFaceSize(face_box)
                if pose <= 30 and size >= 64:
                    return int(size * float((90 - pose) / 10))
                else:
                    return InfoVideo_Person_Preview.getFaceSize(face_box)
        return 0

    @staticmethod
    def getFaceSize(face_box) -> int:
        lft, top, rig, bot = face_box
        return min((rig - lft), (bot - top))

    def isValid(self) -> bool:
        return np.all(self.face_key_points_score[:3] >= 0.5)

    def summaryAsDict(self, size, is_bgr) -> dict:
        summary = dict(
            frame_index=int(self.frame_index),
            face_box=self.face_box.tolist(),
            face_key_points_xy=self.face_key_points_xy.astype(np.int32).tolist(),
            face_key_points_score=(self.face_key_points_score*100).astype(np.int32).tolist(),
            face_valid_points_num=int(self.face_valid_points_num),
            face_score=float(self.face_score),
            face_size=int(self.face_size),
        )
        if size > 0 and isinstance(is_bgr, bool):
            assert isinstance(self.frame_bgr, np.ndarray), self.frame_bgr
            summary['image'] = np.copy(self.frame_bgr)
            summary['box'] = self.face_box.tolist()
            assert isinstance(self.face_align_cache, XPortrait), self.face_align_cache
            summary['preview'] = self.transformImage(self.face_align_cache.bgr, size, is_bgr)
        return summary

    """
    """
    @property
    def identity_embedding(self):
        if hasattr(self, '_identity_embedding') is False:
            pad_image = np.pad(self.face_align_cache.bgr, [[256, 256], [256, 256], [0, 0]])
            self._identity_embedding = XManager.getModules('insightface')(pad_image)[0]['embedding']
        return self._identity_embedding

    @property
    def face_score(self):
        if hasattr(self, '_face_score') is False:
            self._face_score = 0
            if isinstance(self.face_align_cache, XPortrait):
                if self.face_align_cache.number > 0:
                    self._face_score = self.getFaceScore(self.face_align_cache, self.face_box)
        return self._face_score


class InfoVideo_Person:
    """
    """
    CategoryName = 'person'

    @staticmethod
    def createFromDict(info_dict):
        person = InfoVideo_Person(info_dict['identity'], yolo_identity=-1)
        person.frame_info_list = [InfoVideo_Person_Frame.fromString(each) for each in info_dict['frame_info_list']]
        person.preview = InfoVideo_Person_Preview.createFromDict(info_dict['preview'])
        person.face_size_max = info_dict['face_size_max']
        return person

    """
    """
    def __init__(self, identity, yolo_identity):
        self.identity = identity
        self.activate = True
        self.preview = None
        self.frame_info_list: typing.List[InfoVideo_Person_Frame] = []  # [InfoVideo_Frame]
        self.smooth_box_tracker = 0.8
        self.smooth_box_face = 0.8
        # for yolo
        self.yolo_identity = yolo_identity
        # for preview
        self.face_size_max = 0

    def __len__(self):
        return len(self.frame_info_list)

    def __str__(self):
        return 'identity={}, yolo_identity={}, frame_length={}, face_size_max={}, preview({})'.format(
            self.identity, self.yolo_identity, len(self), self.face_size_max, str(self.preview))

    """
    """
    def setActivate(self, activate):
        self.activate = bool(activate)

    def smoothing(self):
        length = len(self.frame_info_list)
        if length >= 2:
            info1 = self.frame_info_list[length - 1]
            info2 = self.frame_info_list[length - 2]
            if info1.box_copy is False and info2.box_copy is True:
                # search
                index = 0
                for n in range(length - 2, -1, -1):
                    info_cur = self.frame_info_list[n]  # -2-(len-2) ==> -len ==> the last
                    if info_cur.box_copy is False:
                        index = n
                        break
                # refine
                info_head = self.frame_info_list[index]
                info_tail = self.frame_info_list[length - 1]
                copy_length = length - 1 - index - 1
                for n in range(1, copy_length + 1):
                    info_cur = self.frame_info_list[index + n]
                    assert info_cur.box_copy is True, (index, n, length)
                    r = 1. - float(n / (copy_length + 1))
                    info_cur.box_face = (r * info_head.box_face + (1 - r) * info_tail.box_face).astype(np.int32)
                    info_cur.box_copy = False
            if info1.box_copy is True and info2.box_copy is True:
                r = 0.5
                info1.box_face = (r * info2.box_face + (1 - r) * info1.box_face).astype(np.int32)

    def appendInfo(self, frame_index, frame_bgr, key_points, box_track, box_face, box_face_rot):
        lft, top, rig, bot = box_face
        key_points_xy = key_points[:5, :2]  # 5,2
        key_points_score = key_points[:5, 2]  # 5,
        if np.sum(box_face.astype(np.int32)) != 0 and lft < rig and top < bot:
            self.frame_info_list.append(InfoVideo_Person_Frame(
                frame_index=frame_index, box_track=box_track, box_face=box_face,
                key_points_xy=key_points_xy, key_points_score=key_points_score, box_copy=False))
            bbox = BoundingBox(box_face)
            face_size = min(bbox.width, bbox.height)
            self.face_size_max = int(max(face_size, self.face_size_max))
        else:
            assert len(self.frame_info_list) > 0
            info_last = self.frame_info_list[-1]
            if info_last.frame_index == frame_index - 1:
                box_face = np.copy(info_last.box_face)
                self.frame_info_list.append(InfoVideo_Person_Frame(
                    frame_index=frame_index, box_track=box_track, box_face=box_face,
                    key_points_xy=key_points_xy, key_points_score=key_points_score, box_copy=True))
        # enforce to smoothing
        self.smoothing()
        self.setIdentityPreview(frame_index, frame_bgr, key_points, box_face)

    def getLastInfo(self) -> InfoVideo_Person_Frame:
        return self.frame_info_list[-1]

    def formatAsDict(self, with_frame_info=True):
        time_beg = self.frame_info_list[0].frame_index
        time_end = self.frame_info_list[-1].frame_index
        time_len = len(self.frame_info_list)
        preview_dict = self.getPreviewSummary(size=0, is_bgr=None)
        info = dict(
            category=self.CategoryName, identity=self.identity, time_beg=time_beg, time_end=time_end, time_len=time_len,
            face_size_max=self.face_size_max, preview=preview_dict)
        if with_frame_info is True:
            info['frame_info_list'] = [info.formatAsString() for info in self.frame_info_list]
        return info

    def setIdentityPreview(self, frame_index, frame_bgr, key_points, face_box):
        assert isinstance(key_points, np.ndarray), key_points
        face_key_points_xy = key_points[:5, :2]
        face_key_points_score = key_points[:5, 2]
        if self.preview is None:
            face_align_cache = None
            if np.all(face_key_points_score[:3] > 0.5):
                face_align_cache = AlignHelper.getAlignFaceCache(
                    frame_bgr, face_key_points_xy[[np.array([2, 1, 0], dtype=np.int32)]])
                # cv2.imwrite(R'N:\archive\2025\0215-masking\error_video\task-2440\first-{}-{}-{}.png'.format(
                #     frame_index, self.identity, face_key_points_xy[:3, :].astype(np.int32)), face_align_cache.bgr)
            self.preview = InfoVideo_Person_Preview(frame_index, frame_bgr, face_box, key_points[:5, :], face_align_cache)
        else:
            # update the preview face
            face_valid_points_num = int(np.count_nonzero((face_key_points_score > 0.5).astype(np.int32)))
            if np.all(face_key_points_score[:3] > 0.5) and face_valid_points_num >= self.preview.face_valid_points_num and \
                    (np.sum(face_key_points_score) - self.preview.face_key_points_score_sum) > 0.1:
                face_align_cache = AlignHelper.getAlignFaceCache(
                    frame_bgr, face_key_points_xy[[np.array([2, 1, 0], dtype=np.int32)]])
                if face_align_cache.number > 0:
                    preview = InfoVideo_Person_Preview(frame_index, frame_bgr, face_box, key_points[:5, :], face_align_cache)
                    if preview.face_score > self.preview.face_score:
                        logging.info('identity={}, {} --> {}'.format(self.identity, self.preview, preview))
                        self.preview = preview  # just update
            else:
                pass  # nothing to do

    def getPreviewSummary(self, size, is_bgr) -> dict:
        assert isinstance(self.preview, InfoVideo_Person_Preview), self.preview
        return self.preview.summaryAsDict(size, is_bgr)

    def checkPerson(self, face_size_min, num_frame_min) -> bool:
        person_frame_valid = bool(self.face_size_max > face_size_min and len(self) > num_frame_min)
        person_preview_valid = self.preview.isValid()
        return bool(person_frame_valid and person_preview_valid)

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