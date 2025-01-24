
import logging
import os
import platform
import multiprocessing
import sys
import traceback
import typing
import cv2
import numpy as np
import tqdm
import threading
from .libscaner import *
from ...utils.context import XContextTimer
from ...utils.video import XVideoReader, XVideoWriter, XVideoWriterAsynchronous
from ...utils.resource import Resource
from ... import XManager


class WorkerLock:
    def __init__(self, total, schedule_call):
        assert total > 0, total
        self.total = total
        self.call = schedule_call
        self.lock = threading.Lock()
        self.counter = 0  # tqdm.tqdm(total=total)
        # self.bar = p_tqdm

    def update(self, n_seq, n=1):
        self.lock.acquire()
        try:
            # self.bar.update(n)
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


class MaskingVideoWorker(threading.Thread):
    """
    """
    def __init__(self, num_seq, path_in_video, options_dict, iterator_dict, masking_function, worker_lock, verbose=True, debug=False):
        super(MaskingVideoWorker, self).__init__(daemon=False)
        self.num_seq = num_seq
        self.reader = XVideoReader(path_in_video)
        self.iterator_list = iterator_dict['iterator_list']
        self.index_beg = iterator_dict['beg']
        self.index_end = iterator_dict['end']
        self.options_dict = options_dict
        self.masking_function = masking_function
        self.worker_lock: WorkerLock = worker_lock
        self.verbose = verbose
        self.debug = debug
        self.result_list = []
        # exception
        self.exit_code = 0
        self.exception = None
        self.exc_traceback = ''

    def run(self):
        try:
            self.masking()
        except Exception as e:
            self.exit_code = 1  # 0 is normal
            self.exception = e
            self.exc_traceback = ''.join(traceback.format_exception(*sys.exc_info()))

    def join(self, timeout=None) -> None:
        threading.Thread.join(self, timeout)
        if self.exit_code == 1:
            setattr(self.exception, 'traceback', ''.join(traceback.format_exception(*sys.exc_info())))
            raise self.exception

    def trying(self, frame_index, frame_bgr, *args, **kwargs):
        if self.debug is False:
            try:
                frame_bgr = self.masking_function(frame_index, frame_bgr, *args, **kwargs)
            finally:
                return frame_bgr
        else:
            return self.masking_function(frame_index, frame_bgr, *args, **kwargs)

    def masking(self, *args, **kwargs):
        self.reader.resetPositionByIndex(self.index_beg)
        frame_index = self.index_beg
        for n, bgr in enumerate(self.reader):
            if not bool(self.index_beg <= frame_index <= self.index_end):
                break  # finish, just break
            # masking process
            for _, (person, it) in enumerate(self.iterator_list):
                info = it.next()
                if info.index_frame == frame_index:
                    if person.identity in self.options_dict:
                        masking_option = self.options_dict[person.identity]
                        bgr = self.trying(frame_index, bgr, info.box_face, masking_option)
                    it.update()
            self.worker_lock.update(self.num_seq, 1)
            self.result_list.append((frame_index, bgr))
            frame_index += 1
            if self.verbose is True:
                self.worker_lock.info('{:<4d}: {:<4d}({:<4d},{:<4d})'.format(self.num_seq, n, self.index_beg, self.index_end))
        self.worker_lock.warning('finish threading: {:<4d}({:<4d},{:<4d})'.format(self.num_seq, self.index_beg, self.index_end))

    def getResult(self):
        return self.result_list

    """
    """
    @property
    def info(self):
        if platform.system().lower() == 'windows':
            return print
        if platform.system().lower() == 'linux':
            return logging.info

    @property
    def warning(self):
        if platform.system().lower() == 'windows':
            return print
        if platform.system().lower() == 'linux':
            return logging.warning

    @staticmethod
    def createScheduleCallWithLock(lock, update_bar, update_function):
        def function(name):
            lock.acquire()
            try:
                update_bar.update(1)
                update_function(name, update_bar.n)
            finally:
                lock.release()

        assert isinstance(update_bar, tqdm.tqdm)
        return function

    @staticmethod
    def createWorkers(num_workers, path_in_video, options_dict, iterator_list, masking_function, schedule_call, debug):
        num_split_max = multiprocessing.cpu_count()
        assert 0 < num_workers <= num_split_max, 'num split should be (0,{}], but is {}'.format(num_split_max, num_workers)
        worker_lock = WorkerLock(len(XVideoReader(path_in_video)), schedule_call)
        worker_list = []
        for n in range(num_workers):
            worker = MaskingVideoWorker(n, path_in_video, options_dict, iterator_list[n], masking_function, worker_lock, False, debug)
            worker_list.append(worker)
        return worker_list

    @staticmethod
    def doMaskingParallel(worker_list):
        try:
            for worker in worker_list:
                assert isinstance(worker, MaskingVideoWorker)
                worker.start()
            for worker in worker_list:
                assert isinstance(worker, MaskingVideoWorker)
                worker.join()
            logging.warning('finish masking parallel')
        except Exception as e:
            raise e

    @staticmethod
    def dumpAll(path_video_out, worker_list, config_or_path_video_in=None):
        config = config_or_path_video_in
        if isinstance(config_or_path_video_in, str):
            if os.path.isfile(config_or_path_video_in):
                path_video_in = config_or_path_video_in
                config = XVideoReader(path_video_in).desc(True)
        assert isinstance(config, dict)
        writer = XVideoWriter(config)
        writer.open(path_video_out)
        for worker in worker_list:
            assert isinstance(worker, MaskingVideoWorker)
            writer.dump(worker.getResult())
        writer.release(reformat=True)

