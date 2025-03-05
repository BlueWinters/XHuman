import copy
import logging
import cv2
import numpy as np
import json
import dataclasses
import typing
from .scanning_video import ScanningVideo
from ..helper import AsynchronousCursor, BoundingBox, AlignHelper
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
            summary['cartoon_bgr'] = np.copy(self.frame_bgr)
            summary['cartoon_box'] = self.face_box.tolist()
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
        self.num_valid_frames = 0
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
            self.num_valid_frames += 1
        else:
            assert len(self.frame_info_list) > 0
            info_last = self.frame_info_list[-1]
            if info_last.frame_index == frame_index - 1:
                box_face = np.copy(info_last.box_face)
                self.frame_info_list.append(InfoVideo_Person_Frame(
                    frame_index=frame_index, box_track=box_track, box_face=box_face,
                    key_points_xy=key_points_xy, key_points_score=key_points_score, box_copy=True))
        # enforce to smoothing
        # self.smoothing()
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
                        logging.info('identity={}, ({} --> {})'.format(self.identity, self.preview, preview))
                        self.preview = preview  # just update
            else:
                pass  # nothing to do

    def getPreviewSummary(self, size, is_bgr) -> dict:
        assert isinstance(self.preview, InfoVideo_Person_Preview), self.preview
        return self.preview.summaryAsDict(size, is_bgr)

    def checkPerson(self, face_size_min, num_frame_min) -> bool:
        person_frame_valid = bool(self.face_size_max > face_size_min and self.num_valid_frames > num_frame_min)
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
                key_points_flag = (info1.key_points_score > 0.5).astype(np.int32) * (info2.key_points_score > 0.5).astype(np.int32)
                for i, idx in enumerate(range(index_beg, index_end)):
                    r = float(array[i])
                    frame_index = idx
                    box_track = (r * info1.box_track + (1 - r) * info2.box_track).astype(np.int32)
                    box_face = (r * info1.box_face + (1 - r) * info2.box_face).astype(np.int32)
                    key_points_xy = (r * info1.key_points_xy + (1 - r) * info2.key_points_xy).astype(np.float32) * key_points_flag[:, None]
                    key_points_score = (r * info1.key_points_score + (1 - r) * info2.key_points_score).astype(np.float32) * key_points_flag
                    self.frame_info_list.insert(n+1+i, InfoVideo_Person_Frame(
                        frame_index=frame_index, box_track=box_track, box_face=box_face,
                        key_points_xy=key_points_xy, key_points_score=key_points_score, box_copy=True))
                    logging.info('interpolate frames: identity-{}, insert {} into ({}, {}), ({}, {}, {})'.format(
                        self.identity, frame_index, index_beg, index_end,
                        np.round(key_points_score*100).astype(np.int32),
                        np.round(info1.key_points_score*100).astype(np.int32),
                        np.round(info2.key_points_score*100).astype(np.int32)))
                n += index_end - index_beg
            else:
                n += 1

    def interpolateFramesBegAndEnd(self, num_interp_beg, num_interp_end, num_frames):
        assert isinstance(num_interp_beg, int) and isinstance(num_interp_end, int)
        assert num_interp_beg >= 0 and num_interp_end >= 0, (num_interp_beg, num_interp_end)
        for n in range(num_interp_beg):
            if self.frame_info_list[0].frame_index > 1:
                info_frame = copy.deepcopy(self.frame_info_list[0])
                info_frame.frame_index -= 1
                info_frame.box_copy = True
                self.frame_info_list.insert(0, info_frame)
        for n in range(num_interp_end):
            if self.frame_info_list[-1].frame_index < num_frames:
                info_frame = copy.deepcopy(self.frame_info_list[-1])
                info_frame.frame_index += 1
                info_frame.box_copy = True
                self.frame_info_list.append(info_frame)


class ScanningVideo_Person(ScanningVideo):
    """
    """
    IOU_Threshold = 0.3

    @classmethod
    def createVideoScanning(cls, **kwargs):
        if 'path_in_json' in kwargs and isinstance(kwargs['path_in_json'], str):
            info_video = ScanningVideo_Person()
            info_video.object_identity_history = [InfoVideo_Person.createFromDict(info) for info in json.load(open(kwargs['path_in_json'], 'r'))]
            return info_video
        if 'info_string' in kwargs and isinstance(kwargs['info_string'], str):
            info_video = ScanningVideo_Person()
            info_video.object_identity_history = [InfoVideo_Person.createFromDict(info) for info in json.loads(kwargs['info_string'])]
            return info_video
        if 'objects_list' in kwargs and isinstance(kwargs['objects_list'], list):
            info_video = ScanningVideo_Person()
            info_video.object_identity_history = [InfoVideo_Person.createFromDict(info) for info in kwargs['objects_list']]
            return info_video
        raise NotImplementedError('"objects_list", "path_in_json", "info_string" not in kwargs')

    def __init__(self, **kwargs):
        super(ScanningVideo_Person, self).__init__(**kwargs)
        self.tracking_model = 'yolo11x-pose'
        self.tracking_classes = [0]
        self.tracking_config = dict(
            persist=True, conf=0.5, iou=0.7, classes=self.tracking_classes, tracker='bytetrack.yaml', verbose=False)

    """
    tracking pipeline
    """
    def finishTracking(self):
        self.updateObjectList([])
        self.interpolateFrame()
        self.concatenateIdentity()

    def updateWithYOLO(self, frame_index, frame_bgr, result):
        person_list_new = []
        number = len(result)
        if number > 0 and result.boxes.id is not None:
            classify = np.reshape(np.round(result.boxes.cls.cpu().numpy()).astype(np.int32), (-1,))
            scores = np.reshape(result.boxes.conf.cpu().numpy().astype(np.float32), (-1,))
            identity = np.reshape(result.boxes.id.cpu().numpy().astype(np.int32), (-1,))
            boxes = np.reshape(np.round(result.boxes.xyxy.cpu().numpy()).astype(np.int32), (-1, 4,))
            points = np.reshape(result.keypoints.data.cpu().numpy().astype(np.float32), (-1, 17, 3))
            # update common(tracking without lose)
            index_list = np.argsort(scores)[::-1].tolist()
            for i, n in enumerate(index_list):
                if classify[n] != 0:
                    continue  # only person id needed
                cur_one_box_tracker = boxes[n, :]  # 4: lft,top,rig,bot
                cur_one_key_points = points[n, :, :]
                cur_one_box_face, cur_one_box_rot_face = AlignHelper.transformPoints2FaceBox(frame_bgr, cur_one_key_points, cur_one_box_tracker)
                if self.findByFaceBox(person_list_new, frame_index, cur_one_box_face):
                    continue
                #
                index1, iou1 = self.findBestMatchOfFaceBox(frame_index, self.object_list_current, cur_one_box_face)
                if index1 != -1 and iou1 > 0.5:
                    person_cur = self.object_list_current.pop(index1)
                    assert isinstance(person_cur, InfoVideo_Person)
                    person_cur.appendInfo(frame_index, frame_bgr, cur_one_key_points, cur_one_box_tracker, cur_one_box_face, cur_one_box_rot_face)
                    person_list_new.append(person_cur)
                    continue
                index2, iou2 = self.findBestMatchOfFaceBox(frame_index, self.object_identity_history, cur_one_box_face)
                if index2 != -1 and iou2 > 0.5:
                    person_cur = self.object_identity_history.pop(index2)
                    assert isinstance(person_cur, InfoVideo_Person)
                    person_cur.appendInfo(frame_index, frame_bgr, cur_one_key_points, cur_one_box_tracker, cur_one_box_face, cur_one_box_rot_face)
                    person_list_new.append(person_cur)
                    continue
                #
                index3 = self.matchPreviousByIdentity(self.object_list_current, int(identity[n]))
                if index3 != -1:
                    person_cur = self.object_list_current.pop(index3)
                    assert isinstance(person_cur, InfoVideo_Person)
                    person_cur.appendInfo(frame_index, frame_bgr, cur_one_key_points, cur_one_box_tracker, cur_one_box_face, cur_one_box_rot_face)
                    person_list_new.append(person_cur)
                    continue
                index4 = self.matchPreviousByIdentity(self.object_identity_history, int(identity[n]))
                if index4 != -1:
                    person_cur = self.object_identity_history.pop(index4)
                    assert isinstance(person_cur, InfoVideo_Person)
                    person_cur.appendInfo(frame_index, frame_bgr, cur_one_key_points, cur_one_box_tracker, cur_one_box_face, cur_one_box_rot_face)
                    person_list_new.append(person_cur)
                    continue
                # create new person
                if np.sum(cur_one_box_face) > 0:
                    # create a new person
                    person_new = self.createNewObject(
                        frame_index, frame_bgr, int(identity[n]), cur_one_key_points, cur_one_box_tracker, cur_one_box_face, cur_one_box_rot_face)
                    if person_new is not None:
                        person_list_new.append(person_new)
                else:
                    # skip the unreliable face box
                    pass
        # update current person list
        self.updateObjectList(person_list_new)

    def createNewObject(self, frame_index, bgr, track_identity, key_points, box_track, box_face, box_face_rot):
        self.object_identity_seq += 1
        info_object = InfoVideo_Person(self.object_identity_seq, track_identity)
        info_object.appendInfo(frame_index, bgr, key_points, box_track, box_face, box_face_rot)
        info_object.setIdentityPreview(frame_index, bgr, key_points, box_face)
        return info_object

    @staticmethod
    def findByFaceBox(person_list, frame_index, cur_face_box):
        for person in person_list:
            frame_info = person.getLastInfo()
            if frame_info.frame_index == frame_index:
                iou = BoundingBox.computeIOU(np.reshape(frame_info.box_face, (1, 4)), np.reshape(cur_face_box, (1, 4)))
                if float(iou) > 0.5:
                    return True
        return False

    @staticmethod
    def findBestMatchOfFaceBox(frame_index, person_list, cur_one_box_face):
        iou_max_val = 0.
        iou_max_idx = -1
        for n, person in enumerate(person_list):
            assert isinstance(person, InfoVideo_Person)
            frame_info = person.getLastInfo()
            if frame_index == frame_info.frame_index + 1:
                iou = BoundingBox.computeIOU(np.reshape(frame_info.box_face, (1, 4)), np.reshape(cur_one_box_face, (1, 4)))
                if iou > iou_max_val:
                    iou_max_val = iou
                    iou_max_idx = n
        return iou_max_idx, iou_max_val

    @staticmethod
    def matchPreviousByIdentity(person_list, identity):
        for n, person in enumerate(person_list):
            assert isinstance(person, InfoVideo_Person)
            if person.yolo_identity == identity:
                return n
        return -1

    """
    interpolate frames
    """
    def interpolateFrame(self, max_frame_gap=16, num_interp_beg=0, num_interp_end=0):
        # interpolate frames into inner gap
        for n, person in enumerate(self.object_identity_history):
            assert isinstance(person, InfoVideo_Person), person
            person.interpolateFramesAtGap(max_frame_gap)
        # interpolate frames at begin and end
        for n, person in enumerate(self.object_identity_history):
            assert isinstance(person, InfoVideo_Person), person
            person.interpolateFramesBegAndEnd(num_interp_beg, num_interp_end, len(self.reader))

    """
    merge person by face embedding
    """
    def concatenateIdentity(self):
        def locate(person_identity):
            for n, person in enumerate(self.object_identity_history):
                if person.identity == person_identity:
                    return n
            return -1

        exclude_pair_list = []
        while True:
            obj_cat_pair_list = self.getOnePair(exclude_pair_list)
            if len(obj_cat_pair_list) == 0:
                break  # stop concatenation
            pair_dict = obj_cat_pair_list[0]
            obj_id_src = pair_dict['obj_id_src']
            obj_id_tar = pair_dict['obj_id_tar']
            obj_info_src = pair_dict['obj_info_src']
            obj_info_tar = pair_dict['obj_info_tar']
            obj_info_cat = obj_info_src if obj_id_src < obj_id_tar else obj_info_tar
            self.doConcatenation(obj_info_src, obj_info_tar, obj_info_cat)
            del_index = locate(int(obj_id_src if obj_id_src > obj_id_tar else obj_id_tar))
            obj_del = self.object_identity_history.pop(del_index)
            logging.info('delete object identity-{}'.format(obj_del.identity))
            del obj_del
        return None

    def getOnePair(self, exclude_pair_list):
        obj_cat_pair_list = list()
        for obj_info_src in self.object_identity_history:
            for obj_info_tar in self.object_identity_history:
                exclude_pair = (obj_info_src.identity, obj_info_tar.identity)
                if exclude_pair in exclude_pair_list:
                    continue
                if obj_info_src.identity != obj_info_tar.identity:
                    flag, sim_value, dis_time = self.checkConcatenate(obj_info_src, obj_info_tar)
                    if flag is True:
                        pair_dict = dict(
                            obj_id_src=obj_info_src.identity, obj_info_src=obj_info_src,
                            obj_id_tar=obj_info_tar.identity, obj_info_tar=obj_info_tar,
                            sim_value=sim_value, dis_time=dis_time)
                        obj_cat_pair_list.append(pair_dict)
        if len(obj_cat_pair_list) > 0:
            dtype = [('dis_time', int), ('sim_value', float)]
            value = [(each['dis_time'], 1-each['sim_value']) for each in obj_cat_pair_list]
            iou_and_dis_t = np.array(value, dtype=dtype)
            sorted_index = np.argsort(iou_and_dis_t, order=['dis_time', 'sim_value'])
            return [obj_cat_pair_list[index] for index in sorted_index]
        return obj_cat_pair_list

    @staticmethod
    def checkConcatenate(obj_info_pre, obj_info_cur, face_size_min=64):
        assert isinstance(obj_info_pre, InfoVideo_Person)
        assert isinstance(obj_info_cur, InfoVideo_Person)
        frame_index_pre_end = obj_info_pre.frame_info_list[-1].frame_index
        frame_index_cur_beg = obj_info_cur.frame_info_list[0].frame_index
        if bool(frame_index_pre_end < frame_index_cur_beg):
            obj_pre_preview = obj_info_pre.preview
            obj_cur_preview = obj_info_cur.preview
            assert isinstance(obj_pre_preview, InfoVideo_Person_Preview)
            assert isinstance(obj_cur_preview, InfoVideo_Person_Preview)
            # valid_face_pre = obj_pre_preview.isValid()
            # valid_face_cur = obj_cur_preview.isValid()
            valid_face_pre = bool(obj_pre_preview.face_valid_points_num >= 4 and obj_pre_preview.face_size > face_size_min)
            valid_face_cur = bool(obj_cur_preview.face_valid_points_num >= 4 and obj_cur_preview.face_size > face_size_min)
            # print('check: id-{}({}), id-{}({})'.format(obj_info_pre.identity, valid_face_pre, obj_info_cur.identity, valid_face_cur))
            if valid_face_pre and valid_face_cur:
                try:
                    cosine_similarity = obj_pre_preview.identity_embedding.dot(obj_cur_preview.identity_embedding) / \
                        (np.linalg.norm(obj_pre_preview.identity_embedding) * np.linalg.norm(obj_cur_preview.identity_embedding))
                    # print('id-{} vs id-{}: {:.2f}'.format(obj_info_pre.identity, obj_info_cur.identity, cosine_similarity))
                    if cosine_similarity > 0.6:
                        return True, cosine_similarity, (frame_index_cur_beg - frame_index_pre_end)
                except IndexError or AttributeError:
                    pass  # detect 0 face(s)
        return False, 0., -1

    @staticmethod
    def doConcatenation(obj_info_src, obj_info_tar, obj_info_cat):
        if obj_info_src.preview.face_score > obj_info_tar.preview.face_score:
            obj_info_cat.preview = obj_info_src.preview
        else:
            obj_info_cat.preview = obj_info_tar.preview
        obj_info_cat.frame_info_list = obj_info_src.frame_info_list + obj_info_tar.frame_info_list

    """
    """
    def getPreviewSummaryAsDict(self, *args, **kwargs):
        face_size_min = kwargs.pop('face_size_min', 32)
        num_frame_min = kwargs.pop('num_frame_min', 30)
        size = kwargs.pop('size', 256)
        is_bgr = kwargs.pop('is_bgr', True)
        preview_dict = {}
        for person in self.getSortedHistory():
            assert isinstance(person, InfoVideo_Person), person
            logging.info(str(person))
            if person.checkPerson(face_size_min, num_frame_min) is True:
                preview_dict[person.identity] = person.getPreviewSummary(size, is_bgr)
        return preview_dict

    """
    visual
    """
    def visualVideoScanning(self, frame_index, frame_canvas, cursor_list, **kwargs):
        from ..visor import Visor
        vis_box_rot = kwargs.pop('vis_box_rot', True)
        for n, (person, cursor) in enumerate(cursor_list):
            info: InfoVideo_Person_Frame = cursor.current()
            if info.frame_index == frame_index:
                if vis_box_rot is True:
                    key_points = np.concatenate([info.key_points_xy, info.key_points_score[:, None]], axis=1)
                    box_face, box_face_rot = AlignHelper.transformPoints2FaceBox(frame_canvas, key_points, None)
                    frame_canvas = Visor.visualSinglePerson(frame_canvas, person.identity, info.box_track, box_face_rot, info.key_points)
                else:
                    frame_canvas = Visor.visualSinglePerson(frame_canvas, person.identity, info.box_track, info.box_face, info.key_points)
                cursor.next()
        return frame_canvas

