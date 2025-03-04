
import logging
import copy
import os
import cv2
import numpy as np
import json
import typing
import pickle
from .scanning_video import ScanningVideo
from .infovideo_person import InfoVideo_Person_Frame, InfoVideo_Person, InfoVideo_Person_Preview
from ..helper import AsynchronousCursor, BoundingBox, AlignHelper
from ....utils import XContextTimer, XVideoReader, XVideoWriter
from .... import XManager


class ScanningVideo_Person(ScanningVideo):
    """
    """
    IOU_Threshold = 0.3

    @classmethod
    def createFromDict(cls, **kwargs):
        if 'path_in_json' in kwargs and isinstance(kwargs['path_in_json'], str):
            info_video = ScanningVideo_Person()
            info_video.object_identity_history = [InfoVideo_Person.createFromDict(info) for info in json.load(open(kwargs['path_in_json'], 'r'))]
            return info_video
        if 'video_info_string' in kwargs and isinstance(kwargs['video_info_string'], str):
            info_video = ScanningVideo_Person()
            info_video.object_identity_history = [InfoVideo_Person.createFromDict(info) for info in json.loads(kwargs['video_info_string'])]
            return info_video
        if 'objects_list' in kwargs and isinstance(kwargs['objects_list'], list):
            info_video = ScanningVideo_Person()
            info_video.object_identity_history = [InfoVideo_Person.createFromDict(info) for info in kwargs['objects_list']]
            return info_video
        raise NotImplementedError('"objects_list", "path_in_json", "video_info_string" not in kwargs')

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
    def getPreviewSummaryAsDict(self, face_size_min=32, num_frame_min=30, size=256, is_bgr=True):
        preview_dict = {}
        for person in self.getSortedHistory():
            assert isinstance(person, InfoVideo_Person), person
            logging.info(str(person))
            if person.checkPerson(face_size_min, num_frame_min) is True:
                preview_dict[person.identity] = person.getPreviewSummary(size, is_bgr)
        return preview_dict

    """
    """
    def visualFrame(self, frame_index, frame_canvas, vis_box_rot=True):
        from .scanning_visor import ScanningVisor
        cursor_list = [(person, AsynchronousCursor(person.frame_info_list)) for person in self.object_identity_history]
        for n, (person, cursor) in enumerate(cursor_list):
            info: InfoVideo_Person_Frame = cursor.current()
            if info.frame_index == frame_index:
                if vis_box_rot is True:
                    key_points = np.concatenate([info.key_points_xy, info.key_points_score[:, None]], axis=1)
                    box_face, box_face_rot = AlignHelper.transformPoints2FaceBox(frame_canvas, key_points, None)
                    frame_canvas = ScanningVisor.visualSinglePerson(frame_canvas, person.identity, info.box_track, box_face_rot, info.key_points)
                else:
                    frame_canvas = ScanningVisor.visualSinglePerson(frame_canvas, person.identity, info.box_track, info.box_face, info.key_points)
                cursor.next()
                yield frame_canvas
        return frame_canvas

    def saveVisualScanning(self, path_in_video, path_out_video, schedule_call, **kwargs):
        from .scanning_visor import ScanningVisor
        if isinstance(path_out_video, str):
            schedule_call('扫描视频-可视化追踪', None)
            reader = XVideoReader(path_in_video)
            writer = XVideoWriter(reader.desc(True))
            writer.open(path_out_video)
            writer.visual_index = True
            vis_box_rot = kwargs.pop('vis_box_rot', True)
            vis_points = kwargs.pop('vis_points', True)
            cursor_list = [(person, AsynchronousCursor(person.frame_info_list)) for person in self.object_identity_history]
            for frame_index, frame_bgr in enumerate(reader):
                canvas = frame_bgr
                for n, (person, cursor) in enumerate(cursor_list):
                    info: InfoVideo_Person_Frame = cursor.current()
                    if info.frame_index == frame_index:
                        if vis_box_rot is True:
                            key_points = np.concatenate([info.key_points_xy, info.key_points_score[:, None]], axis=1)
                            box_face, box_face_rot = AlignHelper.transformPoints2FaceBox(canvas, key_points, None)
                            canvas = ScanningVisor.visualSinglePerson(canvas, person.identity, info.box_track, box_face_rot, info.key_points)
                        else:
                            canvas = ScanningVisor.visualSinglePerson(canvas, person.identity, info.box_track, info.box_face, info.key_points)
                        cursor.next()
                writer.write(canvas)
            writer.release(reformat=True)
