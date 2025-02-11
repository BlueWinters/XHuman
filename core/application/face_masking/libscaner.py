
import copy
import logging
import os
import typing
import cv2
import numpy as np
import dataclasses
import tqdm
import json
import skimage
from .boundingbox import BoundingBox
from ...base.cache import XPortrait, XPortraitHelper
from ...thirdparty.cache import XBody, XBodyHelper
from ...utils.context import XContextTimer
from ...utils.video import XVideoReader, XVideoWriter
from ...utils.color import Colors
from ... import XManager


@dataclasses.dataclass(frozen=False)
class PersonFrameInfo:
    index_frame: int
    box_face: np.ndarray  # 4:<int>
    box_tracker: np.ndarray  # 4:<int>
    # points: np.ndarray  # 5,2:<int>
    box_copy: bool

    @staticmethod
    def fromString(string):
        info = [int(v) for v in string[1:-1].split(',')]
        return PersonFrameInfo(
            index_frame=info[0],
            box_face=np.array(info[1:5], dtype=np.int32),
            box_tracker=np.zeros(shape=(4,), dtype=np.int32),
            box_copy=False)


class Person:
    """
    """
    @staticmethod
    def getVisColor(index):
        return Colors.getColor(index)

    @staticmethod
    def loadFromDict(info_dict):
        person = Person(info_dict['identity'])
        person.frame_info_list = [PersonFrameInfo.fromString(each) for each in info_dict['frame_info_list']]
        if 'preview' in info_dict:
            person.loadPreviewFromJson(info_dict['preview'])
        return person

    """
    """
    def __init__(self, identity):
        self.identity = identity
        self.activate = True
        self.preview = None
        self.frame_info_list = []
        self.smooth_box_tracker = 0.8
        self.smooth_box_face = 0.8

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
                for n in range(length-2, -1, -1):
                    info_cur = self.frame_info_list[n]  # -2-(len-2) ==> -len ==> the last
                    if info_cur.box_copy is False:
                        index = n
                        break
                # refine
                info_head = self.frame_info_list[index]
                info_tail = self.frame_info_list[length - 1]
                copy_length = length - 1 - index - 1
                for n in range(1, copy_length+1):
                    info_cur = self.frame_info_list[index+n]
                    assert info_cur.box_copy is True, (index, n, length)
                    r = 1. - float(n / (copy_length+1))
                    info_cur.box_face = (r * info_head.box_face + (1 - r) * info_tail.box_face).astype(np.int32)
                    info_cur.box_copy = False
            if info1.box_copy is True and info2.box_copy is True:
                r = 0.5
                info1.box_face = (r * info2.box_face + (1 - r) * info1.box_face).astype(np.int32)

    def appendInfo(self, index_frame, bgr, box_tracker, box_face, box_face_score):
        lft, top, rig, bot = box_face
        if np.sum(box_face.astype(np.int32)) != 0 and lft < rig and top < bot:
            self.frame_info_list.append(PersonFrameInfo(index_frame=index_frame, box_tracker=box_tracker, box_face=box_face, box_copy=False))
        else:
            assert len(self.frame_info_list) > 0
            info_last = self.frame_info_list[-1]
            if info_last.index_frame == index_frame - 1:
                box_face = np.copy(info_last.box_face)
                self.frame_info_list.append(PersonFrameInfo(index_frame=index_frame, box_tracker=box_tracker, box_face=box_face, box_copy=True))
        # enforce to smoothing
        self.smoothing()
        self.setIdentityPreview(index_frame, bgr, box_face, box_face_score)

    def getLastInfo(self) -> PersonFrameInfo:
        return self.frame_info_list[-1]

    def getInfoDict(self, with_frame_info=True):
        time_beg = self.frame_info_list[0].index_frame
        time_end = self.frame_info_list[-1].index_frame
        time_len = len(self.frame_info_list)
        preview_dict = dict(index_frame=self.preview['index_frame'], box_face=self.preview['box'].tolist(), box_score=float(self.preview['box_score']))
        info = dict(identity=self.identity, time_beg=time_beg, time_end=time_end, time_len=time_len, preview=preview_dict)
        if with_frame_info is True:
            info['frame_info_list'] = [str([info.index_frame, *info.box_face.tolist()]) for info in self.frame_info_list]
        return info

    def setIdentityPreview(self, index_frame, bgr, box_face, box_face_score):
        h, w, c = bgr.shape
        if self.preview is None:
            # lft, top, rig, bot = box_face
            lft, top, rig, bot = BoundingBox(box_face).expand(0.2, 0.2).clip(0, 0, w, h).asInt()
            self.preview = dict(
                index_frame=index_frame, box=box_face, box_score=box_face_score,
                image=np.copy(bgr), face=np.copy(bgr[top:bot, lft:rig]))
        else:
            # just update the preview face
            if box_face_score < self.preview['box_score'] and (self.preview['box_score'] - box_face_score) > 0.1:
                # lft, top, rig, bot = box_face
                lft, top, rig, bot = BoundingBox(box_face).expand(0.2, 0.2).clip(0, 0, w, h).asInt()
                self.preview = dict(
                    index_frame=index_frame, box=box_face, box_score=box_face_score,
                    image=np.copy(bgr), face=np.copy(bgr[top:bot, lft:rig]))

    def loadPreviewFromJson(self, info):
        if self.preview is None:
            self.preview = dict(
                index_frame=info['index_frame'], box=info['box_face'], box_score=info['box_score'], image=None, face=None)

    """
    """
    def getInfoIterator(self):
        class AsynchronousCursor:
            def __init__(self, data):
                self.index = 0
                self.data = data

            def next(self):
                return self.data[self.index]

            def previous(self):
                return self.data[self.index-1]

            def update(self):
                self.index = min(self.index + 1, len(self.data) - 1)

            def __len__(self):
                assert len(self.data)

            def __str__(self):
                return '{}, {}'.format(self.index, len(self.data))

        return AsynchronousCursor(self.frame_info_list)

    def getInfoIterator2(self, idx_beg, idx_end):
        class AsynchronousCursor:
            def __init__(self, data, beg, end):
                self.beg = beg
                self.end = end
                self.index = beg
                self.data = data

            def next(self):
                return self.data[self.index]

            def update(self):
                self.index = min(self.index + 1, self.end)

            def valid(self):
                return bool(self.beg != self.end)

            def __len__(self):
                assert len(self.data)

            def __str__(self):
                return '{} - ({}, {}) == {}'.format(self.index, self.beg, self.end, len(self.data))

        frame_index_array = np.array([info.index_frame for info in self.frame_info_list], dtype=np.int32)
        idx_beg_arg_where = np.argwhere(idx_beg <= frame_index_array)
        idx_end_arg_where = np.argwhere(frame_index_array <= idx_end)
        if len(idx_beg_arg_where) > 0 and len(idx_end_arg_where) > 0:
            beg_pre_idx = int(idx_beg_arg_where[0])
            end_aft_idx = int(idx_end_arg_where[-1])
            return AsynchronousCursor(self.frame_info_list, beg_pre_idx, end_aft_idx)
        else:
            return AsynchronousCursor(self.frame_info_list, 0, 0)


class VideoInfo:
    """
    """
    def __init__(self, fixed_num=-1):
        self.person_identity_seq = 0
        self.person_identity_history = []
        self.person_fixed_num = fixed_num
        # save every frame info
        self.frame_info_list = []
        # current identity list
        self.person_list_current = []

    @property
    def isFixedNumber(self):
        return bool(self.person_fixed_num != -1)

    def addFrameInfo(self, info):
        self.frame_info_list.append(info)

    def updatePersonList(self, person_list_new):
        for n in range(len(self.person_list_current)):
            person = self.person_list_current.pop()
            person.setActivate(False)
            self.person_identity_history.append(person)
        self.person_list_current += person_list_new

    def createNewPerson(self, index_frame, bgr, box_tracker, box_face, box_face_score):
        if self.isFixedNumber and self.person_identity_seq == self.person_fixed_num:
            if len(self.person_identity_history) > 0:
                non_activate_index = [n for n, person in enumerate(self.person_identity_history) if person.activate is False]
                backtracking_index = -1
                backtracking_iou = 0.
                for index in non_activate_index:
                    person = self.person_identity_history[index]
                    info = person.getLastInfo()
                    iou = BoundingBox.iou(BoundingBox(info.box_tracker), BoundingBox(box_tracker))
                    if iou >= backtracking_iou:
                        backtracking_index = index
                        backtracking_iou = iou
                person = self.person_identity_history.pop(backtracking_index)
                person.setActivate(True)
                person.appendInfo(index_frame, bgr, box_tracker, box_face, box_face_score)
                return person
            return None
        # create person as common
        self.person_identity_seq += 1
        person = Person(self.person_identity_seq)
        person.appendInfo(index_frame, bgr, box_tracker, box_face, box_face_score)
        person.setIdentityPreview(index_frame, bgr, box_face, box_face_score)
        return person

    def getSortedHistory(self):
        return sorted(self.person_identity_history, key=lambda person: person.identity)

    def getInfoJson(self, with_frame_info) -> str:
        sorted_history = self.getSortedHistory()
        return json.dumps([person.getInfoDict(with_frame_info) for person in sorted_history], indent=4)

    def dumpInfoToJson(self, path_json, with_frame_info=True):
        with open(path_json, 'w') as file:
            format_list = [person.getInfoDict(with_frame_info) for person in self.getSortedHistory()]
            json.dump(format_list, file, indent=4)

    def getIdentityPreviewDict(self, size=256, is_bgr=True):
        def transform(bgr):
            resized = cv2.resize(bgr, (size, size))
            return resized if is_bgr is True else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        preview_dict = {}
        for person in self.getSortedHistory():
            preview = person.preview
            preview_dict[person.identity] = dict(
                box=preview['box'], image=preview['image'],
                face=transform(preview['face']), index_frame=preview['index_frame'], box_score=preview['box_score'])
        return preview_dict

    def getIdentityPreviewList(self, size=256, is_bgr=True):
        def transform(bgr):
            resized = cv2.resize(bgr, (size, size))
            return resized if is_bgr is True else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        return [transform(person.preview) for person in self.getSortedHistory()]

    def getInfoIterator(self, min_seconds):
        iterator_list = [(person, person.getInfoIterator()) for person in self.person_identity_history
                         if len(person.frame_info_list) > min_seconds]
        return iterator_list

    def getSplitInfoIterator(self, num_frames, num_split, min_seconds):
        each_len = int(np.ceil(num_frames / num_split))
        beg_end_list = [(n*each_len, min((n+1)*each_len-1, num_frames)) for n in range(num_split)]
        person_remain_list = [person for person in self.person_identity_history if len(person.frame_info_list) > min_seconds]
        iterator_list_all = []
        for n, (beg, end) in enumerate(beg_end_list):
            iterator_list = []
            for person in person_remain_list:
                assert isinstance(person, Person)
                it = person.getInfoIterator2(beg, end)
                if it.valid() is True:
                    iterator_list.append((person, it))
                # print('{}(id-{}): {}'.format(n, person.identity, str(iterator_list[-1][-1])))
            iterator_list_all.append(dict(beg=beg, end=end, iterator_list=iterator_list))
        return iterator_list_all

    @staticmethod
    def loadVideoInfo(**kwargs):
        if 'path_in_json' in kwargs and isinstance(kwargs['path_in_json'], str):
            video_info = VideoInfo()
            video_info.person_identity_history = [Person.loadFromDict(info) for info in json.load(open(kwargs['path_in_json'], 'r'))]
            return video_info
        if 'video_info_string' in kwargs and isinstance(kwargs['video_info_string'], str):
            video_info = VideoInfo()
            video_info.person_identity_history = [Person.loadFromDict(info) for info in json.loads(kwargs['video_info_string'])]
            return video_info
        raise NotImplementedError('both "path_in_json" and "path_in_json" not in kwargs')


class LibScaner:
    """
    """
    IOU_Threshold = 0.3
    CacheType = XPortrait
    CacheIteratorCreator = XPortraitHelper.getXPortraitIterator

    """
    """
    def __init__(self, *args, **kwargs):
        pass

    def __del__(self):
        pass

    def initialize(self, *args, **kwargs):
        pass

    """
    """
    @staticmethod
    def setCacheType(cache_type):
        assert cache_type == 'face' or cache_type == 'body', cache_type
        if cache_type == 'face':
            LibScaner.CacheType = XPortrait
            LibScaner.CacheIteratorCreator = XPortraitHelper.getXPortraitIterator
        if cache_type == 'body':
            LibScaner.CacheType = XBody
            LibScaner.CacheIteratorCreator = XBodyHelper.getXBodyIterator

    @staticmethod
    def packageAsCache(source) -> typing.Union[XBody, XPortrait]:
        return LibScaner.CacheType.packageAsCache(source)

    @staticmethod
    def getCacheIterator(**kwargs):
        return LibScaner.CacheIteratorCreator(**kwargs)

    @staticmethod
    def toNdarray(source):
        assert isinstance(source, (np.ndarray, str, XBody, XPortrait))
        if isinstance(source, str):
            if source.endswith('png') or source.endswith('jpg'):
                return cv2.imread(source)
            if source.endswith('pkl'):
                cache = XPortrait.load(source, verbose=False)
                return np.copy(cache.bgr)
            raise NotImplementedError(source)
        if isinstance(source, np.ndarray):
            return source

    """
    """
    @staticmethod
    def findBestMatch(person_list, cur_one_box_tracker):
        iou_max_val = 0.
        iou_max_idx = -1
        dis_min_val = 10000
        dis_min_idx = -1
        for n, person in enumerate(person_list):
            assert isinstance(person, Person)
            frame_info = person.getLastInfo()
            pre_one_rect = BoundingBox(frame_info.box_tracker)
            cur_one_rect = BoundingBox(cur_one_box_tracker)
            iou = BoundingBox.iou(pre_one_rect, cur_one_rect)
            if iou > iou_max_val:
                iou_max_val = iou
                iou_max_idx = n
            dis = BoundingBox.distance(pre_one_rect, cur_one_rect)
            if dis < dis_min_val:
                dis_min_val = dis
                dis_min_idx = n
        return iou_max_idx, iou_max_val, dis_min_idx, dis_min_val

    @staticmethod
    def updateCommon(index_frame, cache, video_info: VideoInfo):
        person_list_new = []
        num_max = min(cache.number, video_info.person_fixed_num) if video_info.isFixedNumber else cache.number
        for n in range(num_max):
            cur_one_box = cache.box[n, :]  # 4: lft,top,rig,bot
            box_face_score = np.sum(np.abs(cache.radian[n, :])) if isinstance(cache, XPortrait) else 1
            if cache.score[n] < 0.5:
                continue
            iou_max_idx, iou_max_val, dis_min_idx, dis_min_val = LibScaner.findBestMatch(video_info.person_list_current, cur_one_box)
            if iou_max_idx != -1 and (iou_max_val > LibScaner.IOU_Threshold or video_info.isFixedNumber):
                person_cur = video_info.person_list_current.pop(iou_max_idx)
                assert isinstance(person_cur, Person)
                person_cur.appendInfo(index_frame, cache.bgr, cur_one_box, cur_one_box, box_face_score)
                person_list_new.append(person_cur)
            else:
                # create a new person
                person_new = video_info.createNewPerson(index_frame, cache.bgr, cur_one_box, cur_one_box, box_face_score)
                if person_new is None:
                    person_cur = video_info.person_list_current.pop(dis_min_idx)
                    assert isinstance(person_cur, Person)
                    person_cur.appendInfo(index_frame, cache.bgr, cur_one_box, cur_one_box, box_face_score)
                    person_list_new.append(person_cur)
                else:
                    person_list_new.append(person_new)
        # update current person list
        video_info.updatePersonList(person_list_new)
        # set fixed number
        if video_info.isFixedNumber is False and len(video_info.person_identity_history) > 0:
            video_info.person_fixed_num = len(video_info.person_identity_history)

    """
    """
    @staticmethod
    def matchPrevious(person_list, identity):
        for n, person in enumerate(person_list):
            assert isinstance(person, Person)
            if person.identity == identity:
                return n
        return -1

    @staticmethod
    def realignFace(points, w, h, index):
        template = np.array([[192, 239], [318, 240], [256, 314]], dtype=np.float32)
        dst_pts = points[np.array(index, dtype=np.int32)]
        src_pts = template[:len(dst_pts), :]
        transform = skimage.transform.SimilarityTransform()
        transform.estimate(src_pts, dst_pts)
        box = np.array([[0, 0, 1], [512, 0, 1], [512, 512, 1], [0, 512, 1]], dtype=np.float32)
        box_remap = np.dot(transform.params, box.T)[:2, :].T
        box_remap_int = box_remap.astype(np.int32)
        lft = np.min(box_remap_int[:, 0])
        rig = np.max(box_remap_int[:, 0])
        top = np.min(box_remap_int[:, 1])
        bot = np.max(box_remap_int[:, 1])
        # bbox = lft, top, rig, bot
        bbox = BoundingBox(np.array([lft, top, rig, bot], dtype=np.int32)).toSquare().clip(0, 0, w-1, h-1).asInt()
        return bbox

    @staticmethod
    def transformPoints2FaceBox(bgr, key_points, box, threshold=0.5):
        h, w, c = bgr.shape
        confidence = key_points[:, 2].astype(np.float32)
        points = key_points[:, :2].astype(np.float32)
        if confidence[3] > threshold and confidence[4] > threshold:
            lft_ear = points[4, :]
            rig_ear = points[3, :]
            len_ear = np.linalg.norm(lft_ear - rig_ear)
            lft = min(lft_ear[0], rig_ear[0])
            rig = max(lft_ear[0], rig_ear[0])
            ctr_ear = (lft_ear + rig_ear) / 2
            top = int(max(ctr_ear[1] - 0.4 * len_ear, 0))
            bot = int(min(ctr_ear[1] + 0.6 * len_ear, h))
            if points[4, 0] < points[2, 0] < points[0, 0] < points[1, 0] < points[3, 0]:
                bbox = BoundingBox(np.array([lft, top, rig, bot], dtype=np.int32)).toSquare().clip(0, 0, w-1, h-1).asInt()
                if confidence[0] > threshold:
                    return np.array(bbox, dtype=np.int32), 1 + np.mean(1-confidence[5:])
                else:
                    return np.array(bbox, dtype=np.int32), 4 + np.mean(1-confidence[5:])
            else:
                if confidence[1] > threshold and confidence[2] > threshold:
                    bbox = LibScaner.realignFace(points, w, h, index=[2, 1, 0])
                    return np.array(bbox, dtype=np.int32), 4 + np.mean(1 - confidence[5:])
        if confidence[3] > threshold and confidence[1] > threshold:
            rig = points[3, 0]  # points[1, 0] < points[3, 0]
            if confidence[2] > threshold:
                assert confidence[0] > threshold, confidence[0]
                if points[2, 0] < points[0, 0] < points[1, 0] < points[3, 0]:
                    rig_ratio = float(rig - points[1, 0]) / float(rig - points[0, 0])
                    lft = points[2, 0] - float(rig - points[0, 0]) * (1 - rig_ratio)
                    lft, rig = min(lft, rig), max(lft, rig)
                    len_c2rig = abs(rig - points[0, 0])
                    top = int(max(points[0, 1] - 0.4 * len_c2rig, 0))
                    bot = int(min(points[0, 1] + 0.8 * len_c2rig, h))
                    bbox = BoundingBox(np.array([lft, top, rig, bot], dtype=np.int32)).toSquare().clip(0, 0, w-1, h-1).asInt()
                else:
                    bbox = LibScaner.realignFace(points, w, h, index=[2, 1, 0])
                return np.array(bbox, dtype=np.int32), 2 + np.mean(1-confidence[5:])
            if confidence[0] > threshold:
                if points[0, 0] < points[1, 0] < points[3, 0]:
                    lft = points[0, 0] - abs(points[0, 0] - points[1, 0])  # min(lft, points[0, 0])
                    lft, rig = min(lft, rig), max(lft, rig)
                    len_c2rig = abs(rig - points[0, 0])
                    top = int(max(points[0, 1] - 0.4 * len_c2rig, 0))
                    bot = int(min(points[0, 1] + 0.8 * len_c2rig, h))
                    bbox = BoundingBox(np.array([lft, top, rig, bot], dtype=np.int32)).toSquare().clip(0, 0, w-1, h-1).asInt()
                else:
                    bbox = LibScaner.realignFace(points, w, h, index=[1, 0])
                return np.array(bbox, dtype=np.int32), 3 + np.mean(1-confidence[5:])
        if confidence[4] > threshold and confidence[2] > threshold:
            lft = points[4, 0]
            if confidence[1] > threshold:
                assert confidence[0] > threshold, confidence[0]
                if points[4, 0] < points[2, 0] < points[0, 0] < points[1, 0]:
                    lft_ratio = float(points[2, 0]-lft) / float(points[0, 0]-lft)
                    rig = points[1, 0] + float(points[0, 0] - lft) * (1 - lft_ratio)
                    lft, rig = min(lft, rig), max(lft, rig)
                    len_c2lft = abs(points[0, 0] - lft)
                    top = int(max(points[0, 1] - 0.4 * len_c2lft, 0))
                    bot = int(min(points[0, 1] + 0.8 * len_c2lft, h))
                    bbox = BoundingBox(np.array([lft, top, rig, bot], dtype=np.int32)).toSquare().clip(0, 0, w-1, h-1).asInt()
                else:
                    bbox = LibScaner.realignFace(points, w, h, index=[2, 1, 0])
                return np.array(bbox, dtype=np.int32), 2 + np.mean(1-confidence[5:])
            if confidence[0] > threshold:
                if points[4, 0] < points[2, 0] < points[0, 0]:
                    rig = points[0, 0] + abs(points[2, 0] - points[0, 0])  # min(lft, points[0, 0])
                    lft, rig = min(lft, rig), max(lft, rig)
                    len_c2lft = abs(points[0, 0] - lft)
                    top = int(max(points[0, 1] - 0.4 * len_c2lft, 0))
                    bot = int(min(points[0, 1] + 0.8 * len_c2lft, h))
                    bbox = BoundingBox(np.array([lft, top, rig, bot], dtype=np.int32)).toSquare().clip(0, 0, w-1, h-1).asInt()
                else:
                    bbox = LibScaner.realignFace(points, w, h, index=[2, 0])
                return np.array(bbox, dtype=np.int32), 3 + np.mean(1-confidence[5:])
        return np.array([0, 0, 0, 0], dtype=np.int32), 4 + np.mean(1-confidence[5:])

    @staticmethod
    def hasOverlap(box, n):
        cur = box[n, :]
        for i in range(len(box)):
            if i != n:
                pre_one_rect = BoundingBox(cur)
                cur_one_rect = BoundingBox(box[i, :])
                iou = BoundingBox.iou(pre_one_rect, cur_one_rect)
                if iou > 0:
                    return True
        return False

    @staticmethod
    def resetYOLOTracker():
        XManager.getModules('ultralytics').resetTracker('yolo11m-pose')

    @staticmethod
    def updateWithYOLO(index_frame, frame_bgr, cache_or_result, video_info: VideoInfo):
        if isinstance(cache_or_result, LibScaner.CacheType):
            module = XManager.getModules('ultralytics')['yolo11m-pose']
            result = module.track(frame_bgr, persist=True, conf=0.25, iou=0.7, classes=[0], tracker='bytetrack.yaml', verbose=False)[0]
        else:
            result = cache_or_result
        person_list_new = []
        number = len(result)
        # num_max = min(number, video_info.person_fixed_num) if video_info.isFixedNumber else number
        if number > 0 and result.boxes.id is not None:
            cls = np.reshape(np.round(result.boxes.cls.cpu().numpy()).astype(np.int32), (-1,))
            box = np.reshape(np.round(result.boxes.xyxy.cpu().numpy()).astype(np.int32), (-1, 4,))
            points = np.reshape(result.keypoints.data.cpu().numpy().astype(np.float32), (-1, 17, 3))
            score = np.reshape(result.boxes.conf.cpu().numpy().astype(np.float32), (-1,))
            identity = np.reshape(result.boxes.id.cpu().numpy().astype(np.int32), (-1,))
            # update common(tracking without lose)
            index_list = np.argsort(score)[::-1].tolist()
            for i, n in enumerate(index_list):
                cur_one_box_tracker = box[n, :]  # 4: lft,top,rig,bot
                cur_one_box_face, box_face_score = LibScaner.transformPoints2FaceBox(frame_bgr, points[n, :, :], cur_one_box_tracker)
                box_face_score = box_face_score if LibScaner.hasOverlap(box, n) is False and number == video_info.person_fixed_num else 4
                index = LibScaner.matchPrevious(video_info.person_list_current, int(identity[n]))
                if cls[n] != 0:
                    index_list.remove(n)
                    continue  # only person id needed
                if index != -1:
                    person_cur = video_info.person_list_current.pop(index)
                    assert isinstance(person_cur, Person)
                    person_cur.appendInfo(index_frame, frame_bgr, cur_one_box_tracker, cur_one_box_face, box_face_score)
                    person_list_new.append(person_cur)
                    index_list.remove(n)
                    continue
            # update from history
            for i, n in enumerate(index_list):
                cur_one_box_tracker = box[n, :]  # 4: lft,top,rig,bot
                cur_one_box_face, box_face_score = LibScaner.transformPoints2FaceBox(frame_bgr, points[n, :, :], cur_one_box_tracker)
                box_face_score = box_face_score if LibScaner.hasOverlap(box, n) is False and number == video_info.person_fixed_num else 4
                iou_max_idx, iou_max_val, dis_min_idx, dis_min_val = LibScaner.findBestMatch(video_info.person_list_current, cur_one_box_tracker)
                if iou_max_idx != -1 and (iou_max_val > LibScaner.IOU_Threshold or video_info.isFixedNumber):
                    person_cur = video_info.person_list_current.pop(iou_max_idx)
                    assert isinstance(person_cur, Person)
                    person_cur.appendInfo(index_frame, frame_bgr, cur_one_box_tracker, cur_one_box_face, box_face_score)
                    person_list_new.append(person_cur)
                else:
                    if np.sum(cur_one_box_face) > 0:
                        # create a new person
                        person_new = video_info.createNewPerson(index_frame, frame_bgr, cur_one_box_tracker, cur_one_box_face, box_face_score)
                        if person_new is None:
                            if len(video_info.person_list_current) == 0:
                                continue
                            person_cur = video_info.person_list_current.pop(dis_min_idx)
                            assert isinstance(person_cur, Person)
                            person_cur.appendInfo(index_frame, frame_bgr, cur_one_box_tracker, cur_one_box_face, box_face_score)
                            person_list_new.append(person_cur)
                        else:
                            person_list_new.append(person_new)
                    else:
                        # skip the unreliable face box
                        pass
        # update current person list
        video_info.updatePersonList(person_list_new)
        # set fixed number
        # if video_info.isFixedNumber is False and len(video_info.person_identity_history) > 0:
        #     video_info.person_fixed_num = len(video_info.person_identity_history)

    @staticmethod
    def inferenceOnVideo(reader_iterator, **kwargs) -> VideoInfo:
        with XContextTimer(True) as context:
            with tqdm.tqdm(total=len(reader_iterator)) as bar:
                sample_step = kwargs.pop('sample_step', 1)
                fixed_num = kwargs.pop('fixed_num', -1)
                schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
                video_info = VideoInfo(fixed_num=fixed_num)
                for n, source in enumerate(reader_iterator):
                    if n % sample_step == 0:
                        cache = LibScaner.packageAsCache(source)
                        # LibScaner.updateCommon(n, cache, video_info)
                        LibScaner.updateWithYOLO(n, cache.bgr, cache, video_info)
                    bar.update(1)
                    schedule_call('扫描视频-运行中', float((n+1)/len(reader_iterator)))
                LibScaner.resetYOLOTracker()
                video_info.updatePersonList([])  # end the update
                if 'path_out_json' in kwargs and isinstance(kwargs['path_out_json'], str):
                    schedule_call('扫描视频-后处理', None)
                    video_info.dumpInfoToJson(kwargs['path_out_json'])
                return video_info

    @staticmethod
    def inferenceOnVideo2(path_in_video, **kwargs) -> VideoInfo:
        with XContextTimer(True) as context:
            parameters = dict(persist=True, conf=0.25, iou=0.7, classes=[0], tracker='bytetrack.yaml', verbose=False)
            module = XManager.getModules('ultralytics')['yolo11m-pose']
            results = module.track(path_in_video, **parameters)
            fixed_num = 0
            for n in range(len(results)):
                result = results[n]
                number = len(result)
                if number > 0 and result.boxes.id is not None:
                    fixed_num = max(fixed_num, number)

            schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
            video_info = VideoInfo(fixed_num=fixed_num)
            reader = XVideoReader(path_in_video)
            for n in range(len(results)):
                ret, bgr = reader.read()
                if ret is True:
                    LibScaner.updateWithYOLO(n, bgr, results[n], video_info)
                schedule_call('扫描视频-运行中', float((n + 1) / len(results)))
            LibScaner.resetYOLOTracker()
            video_info.updatePersonList([])  # end the update
            if 'path_out_json' in kwargs and isinstance(kwargs['path_out_json'], str):
                schedule_call('扫描视频-后处理', None)
                video_info.dumpInfoToJson(kwargs['path_out_json'])
            return video_info

    @staticmethod
    def inferenceOnImage(source, **kwargs):
        with XContextTimer(True) as context:
            cache = LibScaner.packageAsCache(source)
            video_info = VideoInfo(fixed_num=-1)
            schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
            schedule_call('扫描图片-运行中', None)
            LibScaner.updateCommon(0, cache, video_info)
            video_info.updatePersonList([])  # end the update
            if 'path_out_json' in kwargs and isinstance(kwargs['path_out_json'], str):
                schedule_call('扫描图片-后处理', None)
                video_info.dumpInfoToJson(kwargs['path_out_json'])
            return video_info

    """
    """
    @staticmethod
    def visualSinglePerson(canvas: np.ndarray, info: PersonFrameInfo, identity):
        rect_th = max(round(sum(canvas.shape) / 2 * 0.003), 2)
        text_th = max(rect_th - 1, 1)
        text_size = rect_th / 4
        box_tracker = np.array(info.box_tracker).astype(np.int32)
        point1 = np.array([box_tracker[0], box_tracker[1]], dtype=np.int32)
        point2 = np.array([box_tracker[2], box_tracker[3]], dtype=np.int32)
        canvas = cv2.rectangle(canvas, point1, point2, Person.getVisColor(identity), 2)
        box_face = np.array(info.box_face).astype(np.int32)
        point1_face = np.array([box_face[0], box_face[1]], dtype=np.int32)
        point2_face = np.array([box_face[2], box_face[3]], dtype=np.int32)
        canvas = cv2.rectangle(canvas, point1_face, point2_face, Person.getVisColor(identity), 1)
        label = str(identity)
        box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]
        outside = point1[1] - box_height - 3 >= 0  # label fits outside box
        point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
        # add bounding box text
        cv2.rectangle(canvas, point1, point2, Person.getVisColor(identity), -1, cv2.LINE_AA)  # filled
        cv2.putText(canvas, label, (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
                    0, text_size, (255, 255, 255), thickness=text_th)
        return canvas

    @staticmethod
    def visualFrameNumber(canvas, n):
        h, w, c = canvas.shape
        rect_th = max(round((w+h) / 2 * 0.003), 2)
        text_th = max(rect_th - 1, 2)
        text_size = rect_th / 4
        points_x = int(w * 0.05)
        points_y = points_x
        cv2.putText(canvas, str(n), (points_x, points_y), 0, text_size, (255, 255, 255), thickness=text_th)
        return canvas

    @staticmethod
    def visualSingleFrame(bgr, video_info):
        canvas = np.copy(bgr)
        for person in video_info.person_identity_history:
            canvas = LibScaner.visualSinglePerson(canvas, person.getLastInfo(), person.identity)
        return canvas

    @staticmethod
    def visualAllFrames(path_in_video, path_out_video, video_info: VideoInfo):
        if isinstance(path_out_video, str):
            reader = XVideoReader(path_in_video)
            writer = XVideoWriter(reader.desc(True))
            writer.open(path_out_video)
            iterator_list = [(person, person.getInfoIterator()) for person in video_info.person_identity_history]
            h, w = reader.h, reader.w
            for index_frame, source in enumerate(reader):
                canvas = LibScaner.toNdarray(source)
                canvas = LibScaner.visualFrameNumber(canvas, index_frame)
                for n, (person, it) in enumerate(iterator_list):
                    try:
                        info: PersonFrameInfo = it.next()
                        if info.index_frame == index_frame:
                            canvas = LibScaner.visualSinglePerson(canvas, info, person.identity)
                            it.update()
                    except IndexError as e:
                        pass
                writer.write(canvas)
            writer.release(reformat=True)

    """
    independent interface
    """
    # @staticmethod
    # def inference_ImagesToVideo(path_in_image, path_out_video, **kwargs):
    #     iterator = XPortraitHelper.getXPortraitIterator(path_image=path_in_image)
    #     video_info = LibScaner.inference(iterator, **kwargs)
    #     LibScaner.visualAllFrames(iterator, path_out_video, video_info)

    @staticmethod
    def inference_VideoToVideo(path_in_video, path_out_video, **kwargs):
        iterator = LibScaner.getCacheIterator(path_video=path_in_video)
        video_info = LibScaner.inference(iterator, **kwargs)
        LibScaner.visualAllFrames(path_in_video, path_out_video, video_info)
        return video_info

