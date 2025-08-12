
import logging
import os
import sys
import traceback
import psutil
import numpy as np
import threading
from ..video import XVideoReader


class XVideoMThreadWorker(threading.Thread):
    """
    """
    def __init__(self, num_seq, frame_stream):
        super(XVideoMThreadWorker, self).__init__()
        assert isinstance(frame_stream, (list, XVideoReader)), frame_stream
        self.num_seq = num_seq
        self.frame_input_stream = frame_stream
        # for exception
        self.exit_code = 0
        self.exception = None
        self.exc_traceback = ''

    def run(self):
        try:
            self.process()
        except Exception as e:
            self.exit_code = 1  # 0 is normal
            self.exception = e
            self.exc_traceback = ''.join(traceback.format_exception(*sys.exc_info()))

    def join(self, timeout=None) -> None:
        threading.Thread.join(self, timeout)
        if self.exit_code == 1:
            setattr(self.exception, 'traceback', ''.join(traceback.format_exception(*sys.exc_info())))
            raise self.exception

    def process(self):
        # function signature template, just overridden this function for custom process
        for frame_index, frame_bgr in enumerate(self.frame_input_stream):
            raise NotImplementedError  # TODO: process each frame


class XVideoMThreadSession:
    def __init__(self, num_workers, path_in_video):
        num_workers_max = psutil.cpu_count(logical=False)
        assert 0 < num_workers <= num_workers_max, \
            'number workers should be (0,{}], but is {}'.format(num_workers_max, num_workers)
        self.num_workers = num_workers
        self.path_in_video = path_in_video
        self.worker_list = self.initialize()

    def initialize(self):
        worker_list = []
        reader = XVideoReader(self.path_in_video)
        index_pair_list = self.getSplitIndex(len(reader), self.num_workers)
        frame_bgr_list = reader.sampleFrames(0, reader.num_frame, 1)
        for n, (beg, end) in zip(range(self.num_workers), index_pair_list):
            worker = XVideoMThreadWorker(num_seq=n, frame_stream=frame_bgr_list[beg:end+1])
            worker.setDaemon(False)
            worker_list.append(worker)
        return worker_list

    @staticmethod
    def getSplitIndex(num_frames, num_workers):
        each_len = int(np.ceil(num_frames / num_workers))
        # index include the end
        index_pair_list = [(n * each_len, min((n + 1) * each_len - 1, num_frames-1)) for n in range(num_workers)]
        return index_pair_list

    def start(self, *args, **kwargs):
        try:
            for worker in self.worker_list:
                # demon=False: to release source properly
                worker.setDaemon(False)
                worker.start()
            """ 
            Reference:
                https://docs.python.org/3/library/threading.html
            Note: 
                Other threads can call a thread’s join() method. 
                This blocks the calling thread until the thread whose join() method is called is terminated.
            """
            for worker in self.worker_list:
                worker.join()
        except Exception as e:
            raise e

