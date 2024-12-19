
import os
import logging
import cv2
import numpy as np
import tempfile
import queue, threading
from typing import List



class XVideoWriter:
    """
    """
    @staticmethod
    def default_fourcc():
        return 'm', 'p', '4', 'v'
        # return ('a', 'v', 'c', '1')

    @staticmethod
    def default_FPS():
        return 16

    """
    """
    def __init__(self, config:dict):
        self._config(config)

    def __del__(self):
        self.release()

    def _config(self, config:dict):
        self.fps = config['fps'] if 'fps' in config \
            else XVideoWriter.default_FPS()
        self.fourcc = config['fourcc'] if 'fourcc' in config \
            else XVideoWriter.default_fourcc()
        assert len(self.fourcc) == 4
        if 'h' in config and 'w' in config:
            self.w = config['w']
            self.h = config['h']
        else:
            self.w = self.h = -1

    @staticmethod
    def _getOpencvWriter(path, fps, w, h, fourcc):
        assert len(fourcc) == 4, fourcc
        writer = cv2.VideoWriter()
        code = cv2.VideoWriter_fourcc(*fourcc)
        writer.open(path, code, fps, (w, h), True)
        return writer.isOpened(), writer

    @staticmethod
    def _getWriter(path, fps, w, h, fourcc, backend='opencv'):
        is_open, video_writer = XVideoWriter._getOpencvWriter(path, fps, w, h, fourcc)
        if is_open is False:
            video_writer.release()
            logging.warning('open file fail: {}'.format(path))
            raise IOError(path)
        return video_writer

    """
    for multi step writer
    usage:
        writer.open(path_out)
        for frame in frame_list:
            writer.write(frame)
    """
    @property
    def writer(self):
        if hasattr(self, '_writer') is False:
            self._writer = self._getWriter(self.path, self.fps, self.w, self.h, self.fourcc)
            self._handle = lambda bgr: self._writer.write(bgr)  # lambda function for writing
        return self._handle

    def open(self, path:str):
        if hasattr(self, 'path') is False:
            assert os.path.exists(os.path.split(path)[0])
            self.path = path
            if self.h != -1 and self.w != -1:
                handle = self.writer  # get a writer handle
        return self

    def release(self):
        if hasattr(self, '_writer'):
            if isinstance(self._writer, cv2.VideoWriter) and self._writer.isOpened():
                self._writer.release()
                return True
        return False

    def write(self, image:np.ndarray):
        assert len(image.shape) == 3 and image.shape[2] == 3, image.shape
        if self.h == -1 or self.w == -1:
            self.h, self.w = image.shape[:2]
        assert self.h == image.shape[0] and self.w == image.shape[1], (self.h, self.w, image.shape)
        self.writer(image)

    def dump(self, bgr_list:List[np.ndarray]):
        for bgr in bgr_list:
            self.write(bgr)



class XVideoWriterSynchronous(XVideoWriter):
    """
    """
    def __init__(self, config):
        super(XVideoWriterSynchronous, self).__init__(config)

    def _serializeYield(self, writer:cv2.VideoWriter):
        self.counter = 0
        while True:
            self.counter += 1
            image = yield self
            writer.write(image)

    @property
    def writer(self):
        if hasattr(self, '_writer') is False:
            self._writer = self._getWriter(self.path, self.fps, self.w, self.h, self.fourcc)
            self._iter = self._serializeYield(self._writer)
            self._iter.__next__()
            self._handle = lambda bgr: self._iter.send(bgr)  # lambda function for writing
        return self._handle



class XVideoWriterAsynchronous(XVideoWriter):
    """
    """
    def __init__(self, config):
        super(XVideoWriterAsynchronous, self).__init__(config)

    @property
    def writer(self):
        if hasattr(self, '_writer') is False:
            self._writer = self._getWriter(self.path, self.fps, self.w, self.h, self.fourcc)
            self._queue = queue.Queue()
            self._worker = threading.Thread(target=lambda image: self._writer.write(image))
            self._worker.setDaemon(True)
            self._worker.start()
            self._handle = lambda bgr: self._queue.put_nowait(bgr)  # lambda function for writing
        return self._handle
