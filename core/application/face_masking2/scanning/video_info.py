
import logging
import copy
import os
import typing
import cv2
import numpy as np
import dataclasses
import tqdm
import json


@dataclasses.dataclass(frozen=False)
class InfoPersonFrame:
    frame_index: int
    box_face: np.ndarray  # 4:<int>
    box_track: np.ndarray  # 4:<int>
    # points: np.ndarray  # 5,2:<int>
    box_copy: bool

    @staticmethod
    def fromString(string):
        info = [int(v) for v in string[1:-1].split(',')]
        return InfoPersonFrame(
            frame_index=info[0],
            box_face=np.array(info[1:5], dtype=np.int32),
            box_track=np.zeros(shape=(4,), dtype=np.int32),
            box_copy=False)


class InfoPerson:
    """
    """
    @staticmethod
    def getVisColor(index):
        return Colors.getColor(index)

    @staticmethod
    def loadFromDict(info_dict):
        person = InfoPerson(info_dict['identity'])
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
        self.smooth_box_track = 0.8
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

    def appendInfo(self, frame_index, bgr, box_track, box_face, box_face_score):
        lft, top, rig, bot = box_face
        if np.sum(box_face.astype(np.int32)) != 0 and lft < rig and top < bot:
            self.frame_info_list.append(InfoPersonFrame(frame_index=frame_index, box_track=box_track, box_face=box_face, box_copy=False))
        else:
            assert len(self.frame_info_list) > 0
            info_last = self.frame_info_list[-1]
            if info_last.frame_index == frame_index - 1:
                box_face = np.copy(info_last.box_face)
                self.frame_info_list.append(InfoPersonFrame(frame_index=frame_index, box_track=box_track, box_face=box_face, box_copy=True))
        # enforce to smoothing
        self.smoothing()
        self.setIdentityPreview(frame_index, bgr, box_face, box_face_score)

    def getLastInfo(self) -> InfoPersonFrame:
        return self.frame_info_list[-1]

    def getInfoDict(self, with_frame_info=True):
        time_beg = self.frame_info_list[0].frame_index
        time_end = self.frame_info_list[-1].frame_index
        time_len = len(self.frame_info_list)
        preview_dict = dict(frame_index=self.preview['frame_index'], box_face=self.preview['box'].tolist(), box_score=float(self.preview['box_score']))
        info = dict(identity=self.identity, time_beg=time_beg, time_end=time_end, time_len=time_len, preview=preview_dict)
        if with_frame_info is True:
            info['frame_info_list'] = [str([info.frame_index, *info.box_face.tolist()]) for info in self.frame_info_list]
        return info

    def setIdentityPreview(self, frame_index, bgr, box_face, box_face_score):
        if self.preview is None:
            lft, top, rig, bot = box_face
            self.preview = dict(
                frame_index=frame_index, box=box_face, box_score=box_face_score,
                image=np.copy(bgr), face=np.copy(bgr[top:bot, lft:rig]))
        else:
            # just update the preview face
            if box_face_score < self.preview['box_score']:
                lft, top, rig, bot = box_face
                self.preview = dict(
                    frame_index=frame_index, box=box_face, box_score=box_face_score,
                    image=np.copy(bgr), face=np.copy(bgr[top:bot, lft:rig]))

    def loadPreviewFromJson(self, info):
        if self.preview is None:
            self.preview = dict(
                frame_index=info['frame_index'], box=info['box_face'], box_score=info['box_score'], image=None, face=None)

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

            def __len__(self):
                assert len(self.data)

            def __str__(self):
                return '{}, {}'.format(self.index, len(self.data))

        frame_index_array = np.array([info.frame_index for info in self.frame_info_list], dtype=np.int32)
        idx_beg_arg_where = np.argwhere(idx_beg <= frame_index_array)
        beg_pre_idx = int(idx_beg_arg_where[0] if len(idx_beg_arg_where) > 0 else frame_index_array[0])
        idx_end_arg_where = np.argwhere(frame_index_array <= idx_end)
        end_aft_idx = int(idx_end_arg_where[-1] if len(idx_end_arg_where) > 0 else frame_index_array[-1])
        return AsynchronousCursor(self.frame_info_list, beg_pre_idx, end_aft_idx)