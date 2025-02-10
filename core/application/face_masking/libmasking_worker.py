
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
from ...utils.video import XVideoReader, XVideoWriter


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
    def __init__(self, num_seq, image_list, options_dict, iterator_dict, masking_function, worker_lock, verbose=True, debug=False):
        super(MaskingVideoWorker, self).__init__()
        self.num_seq = num_seq
        self.image_list = image_list
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
        frame_index = self.index_beg
        counter = 0
        for n, bgr in enumerate(self.image_list):
            if not bool(self.index_beg <= frame_index <= self.index_end):
                break  # finish, just break
            if isinstance(bgr, np.ndarray) is False:
                break  # bug for video
            # masking process
            counter += 1
            for _, (person, it) in enumerate(self.iterator_list):
                if person.identity in self.options_dict:
                    info = it.next()
                    if info.index_frame == frame_index:
                        masking_option = self.options_dict[person.identity]
                        bgr = self.trying(frame_index, bgr, info.box_face, masking_option)
                        it.update()
                    if info.index_frame < frame_index:
                        it.update()
            self.worker_lock.update(self.num_seq, 1)
            self.result_list.append((frame_index, bgr))
            frame_index += 1
            if self.verbose is True:
                self.worker_lock.info('{:<4d}: {:<4d}({:<4d},{:<4d})'.format(self.num_seq, n, self.index_beg, self.index_end))
        self.worker_lock.warning('finish threading {:<6d}({:<6s}): {:<4d}({:<4d},{:<4d} -- {:<4d})'.format(
            self.ident, self.name, self.num_seq, self.index_beg, self.index_end, counter))

    def getResult(self):
        return self.result_list

    """
    """
    @staticmethod
    def createWorkers(num_workers, path_in_video, options_dict, iterator_list, masking_function, schedule_call, debug):
        num_split_max = multiprocessing.cpu_count()
        assert 0 < num_workers <= num_split_max, 'num split should be (0,{}], but is {}'.format(num_split_max, num_workers)
        worker_lock = WorkerLock(len(XVideoReader(path_in_video)), schedule_call)
        worker_list = []
        reader = XVideoReader(path_in_video)
        frame_list = reader.sampleFrames(0, reader.num_frame)
        for n in range(num_workers):
            iterator_dict = iterator_list[n]
            frame_beg, frame_end = iterator_dict['beg'], iterator_dict['end']
            worker = MaskingVideoWorker(n, frame_list[frame_beg:frame_end+1], options_dict, iterator_dict, masking_function, worker_lock, False, debug)
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
            for worker in worker_list:
                assert isinstance(worker, MaskingVideoWorker)
                worker.worker_lock.info('seq-{}: exit code-{}, exception-{}, traceback-{}'.format(
                    worker.num_seq, worker.exit_code, worker.exception, worker.exc_traceback))
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
        # writer.visual_index = True
        for worker in worker_list:
            assert isinstance(worker, MaskingVideoWorker)
            writer.dump(worker.getResult())
        writer.release(reformat=True)

