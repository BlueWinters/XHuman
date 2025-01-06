import copy
import os
import logging
import cv2
import numpy as np
import tempfile
import platform
import queue
import threading
import subprocess
from typing import List


class XVideoWriter:
    """
    """
    @staticmethod
    def reformatVideo(path_video_source, path_video_target):
        if platform.system().lower() == 'linux':
            # example: ffmpeg -i source.avi -c:v copy -c:a copy target.mp4
            command = ['ffmpeg', '-i', path_video_source, '-loglevel', 'warning', '-codec:v', 'libx264', '-codec:a', 'aac', path_video_target]
            subprocess.run(command)
            logging.warning('finish reformat with ffmpeg: {}'.format(command))
        else:
            logging.warning('only linux system support ffmpeg')

    @staticmethod
    def reformatVideoTo(path_video_source, suffix='.mp4'):
        path_video_target = '{}{}'.format(os.path.splitext(path_video_source)[0], suffix)
        XVideoWriter.reformatVideo(path_video_source, path_video_target)
        return path_video_target

    @staticmethod
    def default_fourcc():
        return 'X', 'V', 'I', 'D'

    @staticmethod
    def default_suffix():
        return '.avi'

    @staticmethod
    def default_FPS():
        return 16

    """
    """
    ConfigForceDefaultFormat = True

    @property
    def isForceDefaultFormat(self):
        if platform.system().lower() == 'windows':
            return False
        return XVideoWriter.ConfigForceDefaultFormat

    """
    """
    def __init__(self, config: dict):
        self._config(config)

    def __del__(self):
        self.release()

    def _config(self, config: dict):
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

    def open(self, path: str):
        if hasattr(self, 'path') is False:
            assert os.path.exists(os.path.split(path)[0])
            self.path_source = copy.copy(path)
            self.path = copy.copy(path)  # the final path for writing
            if XVideoWriter.isForceDefaultFormat:
                self.path = '{}{}'.format(os.path.splitext(self.path)[0], XVideoWriter.default_suffix())
                self.fourcc = self.default_fourcc()
            if self.h != -1 and self.w != -1:
                _ = self.writer  # to initialize, get a writer handle
        return self

    def release(self, reformat=True):
        if hasattr(self, '_writer'):
            if isinstance(self._writer, cv2.VideoWriter) and self._writer.isOpened():
                self._writer.release()
                if self.path_source != self.path and reformat is True:
                    self.reformatVideo(self.path, self.path_source)
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
