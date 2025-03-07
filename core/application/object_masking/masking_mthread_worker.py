
import logging
import os
import copy
import numpy as np
import threading
import functools
from .masking_function import MaskingFunction
from .helper.cursor import AsynchronousCursor
from .helper.masking_helper import MaskingHelper
from ...utils.video import XVideoReader, XVideoWriter
from ...utils.video.video_mthread_helper import \
    XVideoMThreadWorker, XVideoMThreadSession


class MaskingVideoWorker(XVideoMThreadWorker):
    def __init__(self, num_seq, frame_bgr_list, frame_index_beg,
            options_dict, cursor_dict, worker_lock, with_hair=True, preview_dict=None, verbose=False):
        super(MaskingVideoWorker, self).__init__(num_seq, frame_bgr_list, frame_index_beg)
        self.cursor_list = cursor_dict['cursor_list']
        self.index_beg = cursor_dict['beg']
        self.index_end = cursor_dict['end']
        self.options_dict = options_dict
        self.worker_lock = worker_lock
        self.with_hair = with_hair
        self.preview_dict = preview_dict
        self.result_list = []
        self.verbose = verbose

    @staticmethod
    def maskingFunction(frame_index, frame_bgr, cursor_list, options_dict, with_hair, preview_dict):
        canvas_bgr = np.copy(frame_bgr)
        mask_info_list = MaskingHelper.getPortraitMaskingWithInfoVideoPlus(
            frame_index, frame_bgr, cursor_list, options_dict, with_hair)
        for _, (person, cursor) in enumerate(cursor_list):
            assert isinstance(cursor, AsynchronousCursor)
            info = cursor.current()
            if info.frame_index == frame_index:
                if person.identity in options_dict:
                    masking_option = options_dict[person.identity]
                    # mask_info = MaskingHelper.getPortraitMaskingWithInfoVideo(
                    #     frame_index, frame_bgr, person, info, options_dict, with_hair=with_hair)
                    # if frame_index < 20:
                    #     import cv2
                    #     cv2.imwrite(R'N:\archive\2025\0215-masking\error_video\05\image\{}.png'.format(frame_index), mask_info['mask'])
                    # other parameters
                    parameters = dict()
                    if person.identity in mask_info_list:
                        parameters['mask_info'] = mask_info_list[person.identity]
                    if person.identity in preview_dict:
                        parameters['preview'] = preview_dict[person.identity]
                    canvas_bgr = MaskingFunction.maskingVideo(
                        frame_bgr, canvas_bgr, info, masking_option, **parameters)
                cursor.next()
        return canvas_bgr

    def process(self):
        frame_index = self.index_beg
        for n, frame_bgr in enumerate(self.frame_bgr_list):
            if not bool(self.index_beg <= frame_index <= self.index_end):
                break  # finish, just break
            canvas_bgr = MaskingVideoWorker.maskingFunction(
                frame_index, frame_bgr, self.cursor_list, self.options_dict, self.with_hair, self.preview_dict)
            self.worker_lock(self.num_seq, 1, frame_index=frame_index, frame_img=np.copy(canvas_bgr))
            self.result_list.append((frame_index, canvas_bgr))
            frame_index += 1
            if self.verbose is True:
                self.worker_lock.info('{:<4d}: {:<4d}({:<4d},{:<4d})'.format(self.num_seq, n, self.index_beg, self.index_end))
        self.worker_lock.warning('finish threading {:<6d}({:<6s}): {:<4d}({:<4d},{:<4d} -- {:<4d})'.format(
            self.ident, self.name, self.num_seq, self.index_beg, self.index_end, len(self.frame_bgr_list)))


class WorkerLock:
    def __init__(self, total, schedule_call):
        assert total > 0, total
        self.total = total
        self.call = schedule_call
        self.lock = threading.Lock()
        self.counter = 0

    def __call__(self, n_seq, n=1, **kwargs):
        self.lock.acquire()
        try:
            self.counter += n
            schedule = float(self.counter) / self.total
            self.call(schedule=schedule, **kwargs)
        finally:
            self.lock.release()

    def info(self, *args, **kwargs):
        self.lock.acquire()
        try:
            logging.info(*args, **kwargs)
        finally:
            self.lock.release()

    def warning(self, *args, **kwargs):
        self.lock.acquire()
        try:
            logging.warning(*args, **kwargs)
        finally:
            self.lock.release()


class MaskingVideoSession(XVideoMThreadSession):
    def __init__(self, num_workers, path_in_video, options_dict, cursor_list, schedule_call, **kwargs):
        self.options_dict = options_dict
        self.cursor_list = cursor_list
        self.schedule_call = schedule_call
        self.with_hair = kwargs.pop('with_hair', True)
        self.preview_dict = kwargs.pop('preview_dict', None)
        self.debug_mode = kwargs.pop('debug_mode', False)
        super(MaskingVideoSession, self).__init__(num_workers, path_in_video)

    def initialize(self):
        worker_list = []
        reader = XVideoReader(self.path_in_video)
        index_pair_list = self.getSplitIndex(len(reader), self.num_workers)
        frame_bgr_list = reader.sampleFrames(0, reader.num_frame, 1)
        # def schedule_call(name:str, schedule:float, all_frame_num:int, frame_index:int, frame_img:np.ndarray)
        schedule_call_partial = functools.partial(
            self.schedule_call, name='打码视频', all_frame_num=len(frame_bgr_list))
        worker_lock = WorkerLock(len(reader), schedule_call_partial)
        for n, (beg, end) in zip(range(self.num_workers), index_pair_list):
            worker = MaskingVideoWorker(
                n, frame_bgr_list[beg:end + 1], beg,
                copy.deepcopy(self.options_dict), self.cursor_list[n], worker_lock, self.with_hair, self.preview_dict)
            worker.setDaemon(False)
            worker_list.append(worker)
        return worker_list

    def dump(self, path_video_out, visual_index=False):
        reader = XVideoReader(self.path_in_video)
        writer = XVideoWriter(reader.desc(True))
        writer.open(path_video_out)
        writer.visual_index = visual_index
        for worker in self.worker_list:
            assert isinstance(worker, MaskingVideoWorker)
            writer.dump(worker.result_list)
        writer.release(reformat=True)
