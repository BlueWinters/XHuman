
import logging
import os
import copy
import numpy as np
import threading
from .masking_function import MaskingFunction
from .helper.cursor import AsynchronousCursor
from .helper.masking_helper import MaskingHelper
from ...utils.video import XVideoReader, XVideoWriter
from ...utils.video.video_mthread_helper import \
    XVideoMThreadWorker, XVideoMThreadSession


class MaskingVideoWorker(XVideoMThreadWorker):
    def __init__(self, num_seq, frame_bgr_list, frame_index_beg, options_dict, cursor_dict, worker_lock, with_hair=True, verbose=False):
        super(MaskingVideoWorker, self).__init__(num_seq, frame_bgr_list, frame_index_beg)
        self.cursor_list = cursor_dict['cursor_list']
        self.index_beg = cursor_dict['beg']
        self.index_end = cursor_dict['end']
        self.options_dict = options_dict
        self.worker_lock = worker_lock
        self.with_hair = with_hair
        self.result_list = []
        self.verbose = verbose

    def process(self):
        frame_index = self.index_beg
        for n, frame_bgr in enumerate(self.frame_bgr_list):
            if not bool(self.index_beg <= frame_index <= self.index_end):
                break  # finish, just break
            canvas_bgr = np.copy(frame_bgr)
            for _, (person, cursor) in enumerate(self.cursor_list):
                assert isinstance(cursor, AsynchronousCursor)
                info = cursor.current()
                if info.frame_index == frame_index:
                    if person.identity in self.options_dict:
                        masking_option = self.options_dict[person.identity]
                        mask_info = MaskingHelper.getPortraitMaskingWithInfoVideo(
                            frame_index, frame_bgr, person, info, self.options_dict, with_hair=self.with_hair)
                        frame_bgr = MaskingFunction.maskingVideoFace(
                            frame_bgr, canvas_bgr, info, masking_option, mask_info=mask_info)
                    cursor.next()
            self.worker_lock(self.num_seq, 1)
            self.result_list.append((frame_index, frame_bgr))
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

    def __call__(self, n_seq, n=1):
        self.lock.acquire()
        try:
            self.counter += n
            self.call(float(self.counter) / self.total)
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
        self.debug_mode = kwargs.pop('debug_mode', False)
        super(MaskingVideoSession, self).__init__(num_workers, path_in_video)

    def initialize(self):
        worker_list = []
        reader = XVideoReader(self.path_in_video)
        index_pair_list = self.getSplitIndex(len(reader), self.num_workers)
        frame_bgr_list = reader.sampleFrames(0, reader.num_frame, 1)
        worker_lock = WorkerLock(len(reader), self.schedule_call)
        for n, (beg, end) in zip(range(self.num_workers), index_pair_list):
            worker = MaskingVideoWorker(
                n, frame_bgr_list[beg:end + 1], beg,
                copy.deepcopy(self.options_dict), self.cursor_list[n], worker_lock, self.with_hair)
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
