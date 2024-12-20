
import logging
import os
import typing
import cv2
import numpy as np
import dataclasses
import tqdm
import json
from ...base.cache import XPortrait, XPortraitHelper
from ...thirdparty import XBody
from ...utils.context import XContextTimer
from ...utils.video import XVideoReader, XVideoWriter
from ...utils.color import Colors
from ... import XManager


@dataclasses.dataclass(frozen=True)
class PersonFrameInfo:
    index_frame: int
    box: np.ndarray  # 4:<int>
    points: np.ndarray  # 5,2:<int>

    @staticmethod
    def fromString(string):
        info = [int(v) for v in string[1:-1].split(',')]
        return PersonFrameInfo(index_frame=info[0], box=np.array(info[1:5], dtype=np.int32), points=np.zeros(shape=(5, 2), dtype=np.int32))


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
        return person

    """
    """
    def __init__(self, identity):
        self.identity = identity
        self.activate = True
        self.preview = None
        self.frame_info_list = []

    def setActivate(self, activate):
        self.activate = bool(activate)

    def appendInfo(self, index_frame, box, points):
        # TODO: interpolate frame info
        self.frame_info_list.append(PersonFrameInfo(index_frame=index_frame, box=box, points=points))

    def getLastInfo(self) -> PersonFrameInfo:
        return self.frame_info_list[-1]

    def getInfoDict(self, with_frame_info=True):
        time_beg = self.frame_info_list[0].index_frame
        time_end = self.frame_info_list[-1].index_frame
        time_len = len(self.frame_info_list)
        info = dict(identity=self.identity, time_beg=time_beg, time_end=time_end, time_len=time_len)
        if with_frame_info is True:
            info['frame_info_list'] = [str([info.index_frame, *info.box.tolist()]) for info in self.frame_info_list]
        return info

    def setIdentityPreview(self, bgr, box):
        if self.preview is None:
            lft, top, rig, bot = box
            self.preview = np.copy(bgr[top:bot, lft:rig, :])

    """
    """
    def getInfoIterator(self):
        class AsynchronousCursor:
            def __init__(self, data):
                self.index = 0
                self.data = data

            def next(self):
                return self.data[self.index]

            def update(self):
                self.index += 1

            def __len__(self):
                assert len(self.data)

        return AsynchronousCursor(self.frame_info_list)


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

    def createNewPerson(self, index_frame, bgr, box, points):
        if self.isFixedNumber and self.person_identity_seq == self.person_fixed_num:
            if len(self.person_identity_history) > 0:
                non_activate_index = [n for n, person in enumerate(self.person_identity_history) if person.activate is False]
                backtracking_index = -1
                backtracking_iou = 0.
                for index in non_activate_index:
                    person = self.person_identity_history[index]
                    info = person.getLastInfo()
                    iou = XRectangle.iou(XRectangle(info.box), XRectangle(box))
                    if iou >= backtracking_iou:
                        backtracking_index = index
                        backtracking_iou = iou
                person = self.person_identity_history.pop(backtracking_index)
                person.setActivate(True)
                person.appendInfo(index_frame, box, points)
                return person
            return None
        # create person as common
        self.person_identity_seq += 1
        person = Person(self.person_identity_seq)
        person.appendInfo(index_frame, box, points)
        person.setIdentityPreview(bgr, box)
        return person

    def getSortedHistory(self):
        return sorted(self.person_identity_history, key=lambda person: person.identity)

    def getInfoJson(self, with_frame_info):
        sorted_history = self.getSortedHistory()
        return json.dumps([person.getInfoDict(with_frame_info) for person in sorted_history], indent=4)

    def dumpInfoToJson(self, path_json, with_frame_info=True):
        with open(path_json, 'w') as file:
            format_list = [person.getInfoDict(with_frame_info) for person in self.getSortedHistory()]
            json.dump(format_list, file, indent=4)

    def getIdentityPreviewDict(self, size=256):
        return {person.identity: cv2.resize(person.preview, (size, size)) for person in self.getSortedHistory()}

    def getIdentityPreviewList(self, size=256, is_bgr=True):
        def transform(bgr):
            resized = cv2.resize(bgr, (size, size))
            return resized if is_bgr is True else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        return [transform(person.preview) for person in self.getSortedHistory()]

    @staticmethod
    def loadHistoryInfoFromJson(path_json):
        with open(path_json, 'r') as file:
            return [Person.loadFromDict(info) for info in json.load(file)]

    @staticmethod
    def loadFromJson(path_json):
        video_info = VideoInfo()
        video_info.person_identity_history = VideoInfo.loadHistoryInfoFromJson(path_json)
        return video_info


class XRectangle:
    def __init__(self, points):
        self.x_min = 0
        self.x_max = 0
        self.y_min = 0
        self.y_max = 0
        self.points = None
        # assign
        self.fromPoints(points)

    def fromPoints(self, points):
        assert isinstance(points, np.ndarray)
        if self.points is None:
            if len(points.shape) == 1:
                assert points.shape[0] == 4, points.shape
                self.points = np.copy(points)
                self.x_min = points[0]  # lft
                self.x_max = points[2]  # top
                self.y_min = points[1]  # top
                self.y_max = points[3]  # bot
            if len(points.shape) == 2:
                assert points.shape[1] == 2, points.shape
                self.points = np.copy(points)
                self.x_min = np.min(points[:, 0])
                self.x_max = np.max(points[:, 0])
                self.y_min = np.min(points[:, 1])
                self.y_max = np.max(points[:, 1])

    def area(self) -> float:
        return (self.y_max - self.y_min) * (self.x_max - self.x_min)

    @staticmethod
    def iou(a, b) -> float:
        assert isinstance(a, XRectangle)
        assert isinstance(b, XRectangle)
        xx1 = max(a.x_min, b.x_min)
        yy1 = max(a.y_min, b.y_min)
        xx2 = min(a.x_max, b.x_max)
        yy2 = min(a.y_max, b.y_max)
        inter_area = (max(0, xx2 - xx1 + 1) * max(0, yy2 - yy1 + 1))
        area_a = a.area()
        area_b = b.area()
        if area_a == 0 or area_b == 0:
            return 0.
        union_area = area_a + area_b - inter_area
        return float(inter_area / union_area) if union_area > 0. else 0.

    @staticmethod
    def distance(a, b) -> float:
        assert isinstance(a, XRectangle)
        assert isinstance(b, XRectangle)
        a_cx = (a.x_min + a.x_max) / 2.
        a_cy = (a.y_min + a.y_max) / 2.
        b_cx = (b.x_min + b.x_max) / 2.
        b_cy = (b.y_min + b.y_max) / 2.
        return np.linalg.norm(np.array([a_cx-b_cx, a_cy-b_cy], dtype=np.float32))

    @staticmethod
    def findBestMatch(pre_all_boxes, cur_one_box):
        iou_max = 0.
        idx_max = -1
        for n in range(len(pre_all_boxes)):
            pre_one_rect = XRectangle(pre_all_boxes[n, :, :])
            cur_one_rect = XRectangle(cur_one_box)
            iou = XRectangle.iou(pre_one_rect, cur_one_rect)
            if iou > iou_max:
                iou_max = iou
                idx_max = n
        return idx_max, iou_max


class LibScaner:
    """
    """
    IOU_Threshold = 0.3

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
    def packageAsCache(source) -> typing.Union[XBody, XPortrait]:
        return XPortrait.packageAsCache(source)

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
    def findBestMatch(person_list, cur_one_box):
        iou_max_val = 0.
        iou_max_idx = -1
        dis_min_val = 10000
        dis_min_idx = -1
        for n, person in enumerate(person_list):
            assert isinstance(person, Person)
            frame_info = person.getLastInfo()
            pre_one_rect = XRectangle(frame_info.box)
            cur_one_rect = XRectangle(cur_one_box)
            iou = XRectangle.iou(pre_one_rect, cur_one_rect)
            if iou > iou_max_val:
                iou_max_val = iou
                iou_max_idx = n
            dis = XRectangle.distance(pre_one_rect, cur_one_rect)
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
            if cache.score[n] < 0.5:
                continue
            iou_max_idx, iou_max_val, dis_min_idx, dis_min_val = LibScaner.findBestMatch(video_info.person_list_current, cur_one_box)
            if iou_max_idx != -1 and (iou_max_val > LibScaner.IOU_Threshold or video_info.isFixedNumber):
                person_cur = video_info.person_list_current.pop(iou_max_idx)
                assert isinstance(person_cur, Person)
                person_cur.appendInfo(index_frame, cur_one_box, cache.points[n, :, :])
                person_list_new.append(person_cur)
            else:
                # create a new person
                person_new = video_info.createNewPerson(index_frame, cache.bgr, cur_one_box, cache.points[n, :, :])
                if person_new is None:
                    person_cur = video_info.person_list_current.pop(dis_min_idx)
                    assert isinstance(person_cur, Person)
                    person_cur.appendInfo(index_frame, cur_one_box, cache.points[n, :, :])
                    person_list_new.append(person_cur)
                else:
                    person_list_new.append(person_new)
        # append all information
        # video_info.addFrameInfo((np.copy(cache.box), np.copy(cache.points)))
        # update current person list
        video_info.updatePersonList(person_list_new)

    @staticmethod
    def inference(reader_iterator, **kwargs) -> VideoInfo:
        with XContextTimer(True) as context:
            with tqdm.tqdm(total=len(reader_iterator)) as bar:
                sample_step = kwargs.pop('sample_step', 1)
                fixed_num = kwargs.pop('fixed_num', -1)
                video_info = VideoInfo(fixed_num=fixed_num)
                for n, source in enumerate(reader_iterator):
                    if n % sample_step == 0:
                        cache = LibScaner.packageAsCache(source)
                        LibScaner.updateCommon(n, cache, video_info)
                    bar.update(1)
                video_info.updatePersonList([])  # end the update
                # print(video_info.getInfoJson())
                if 'path_out_json' in kwargs:
                    video_info.dumpInfoToJson(kwargs['path_out_json'])
                return video_info

    """
    """
    @staticmethod
    def visualSinglePerson(canvas: np.ndarray, info: PersonFrameInfo, identity):
        rect_th = max(round(sum(canvas.shape) / 2 * 0.003), 2)
        text_th = max(rect_th - 1, 1)
        text_size = rect_th / 4
        bbox = np.array(info.box).astype(np.int32)
        point1 = np.array([bbox[0], bbox[1]], dtype=np.int32)
        point2 = np.array([bbox[2], bbox[3]], dtype=np.int32)
        canvas = cv2.rectangle(canvas, point1, point2, Person.getVisColor(identity), 2)
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
    def visualSingleFrame(bgr, video_info):
        canvas = np.copy(bgr)
        for person in video_info.person_list_current:
            canvas = LibScaner.visualSinglePerson(canvas, person.getLastInfo(), person.identity)
        return canvas

    @staticmethod
    def visualAllFrames(path_in_video, path_out_video, video_info: VideoInfo):
        if isinstance(path_out_video, str):
            reader = XVideoReader(path_in_video)
            writer = XVideoWriter(reader.desc(True))
            writer.open(path_out_video)
            iterator_list = [(person, person.getInfoIterator()) for person in video_info.person_identity_history]
            for index_frame, source in enumerate(reader):
                canvas = LibScaner.toNdarray(source)
                for _, (person, it) in enumerate(iterator_list):
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
    #
    # @staticmethod
    # def inference_VideoToVideo(path_in_video, path_out_video, **kwargs):
    #     iterator = XPortraitHelper.getXPortraitIterator(path_video=path_in_video)
    #     video_info = LibScaner.inference(iterator, **kwargs)
    #     LibScaner.visualAllFrames(iterator, path_out_video, video_info)
    #     return video_info

