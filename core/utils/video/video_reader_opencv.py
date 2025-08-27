
import os
import logging
import cv2
import numpy as np
import tempfile
from typing import Union, List, Tuple, Dict, Any


class XVideoReaderOpenCV:
    """
    """
    @staticmethod
    def decode_Codec(raw_codec_format: int):
        decoded_codec_format = (chr((raw_codec_format & 0xFF)),
                                chr((raw_codec_format & 0xFF00) >> 8),
                                chr((raw_codec_format & 0xFF0000) >> 16),
                                chr((raw_codec_format & 0xFF000000) >> 24))
        return decoded_codec_format

    """
    """
    def __init__(self, path_or_stream: Union[str, bytes]):
        self._config(self._open(path_or_stream))
        self.cur = 0

    def _open(self, path_or_stream) -> cv2.VideoCapture:
        """
        warning: global cap.cpp:342 open VIDEOIO(FFMPEG):
            backend is generally available but can't be used to capture by index
        solution: use cv2.CAP_ANY to disable this warning
        reference: opencv/modules/videoio/src/cap.cpp
        """
        cv2.setLogLevel(0)
        capture = cv2.VideoCapture(cv2.CAP_FFMPEG)
        if isinstance(path_or_stream, bytes):
            stream = path_or_stream
            file = tempfile.TemporaryFile()
            file.dump(stream)
            path = file.name
            logging.info('open video from stream')
        else:
            assert isinstance(path_or_stream, str), type(path_or_stream)
            self.path = path = path_or_stream  # str is for both file path
            logging.info('open video from: {}'.format(self.path))
        capture.open(path)
        return capture

    def _config(self, capture: cv2.VideoCapture):
        if capture.isOpened():
            self.capture = capture
            self.w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(capture.get(cv2.CAP_PROP_FPS))
            self.fourcc = XVideoReaderOpenCV.decode_Codec(int(capture.get(cv2.CAP_PROP_FOURCC)))
            self.num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.num_sec = int(self.num_frame / self.fps)

    def suffix(self):
        if hasattr(self, 'path') is True:
            return os.path.splitext(self.path)[1][1:]
        return ''

    def prefix(self):
        if hasattr(self, 'path') is True:
            file_name = os.path.split(self.path)[1]
            return os.path.splitext(file_name)[0]
        return ''

    def desc(self, with_hw: bool = True) -> dict:
        desc = dict(
            fps=self.fps,
            num_frames=self.num_frame,
            fourcc=self.fourcc,
            num_sec=self.num_sec
        )
        if with_hw is True:
            desc['h'] = self.h
            desc['w'] = self.w
        return desc

    def __iter__(self):
        return self

    def __next__(self):
        ret, bgr = self.capture.read()
        if ret is False:
            raise StopIteration
        self.cur += 1
        return bgr

    def __len__(self):
        assert self.isOpen() is True
        return self.num_frame

    def isOpen(self) -> bool:
        if hasattr(self, 'capture'):
            return self.capture.isOpened()
        return False

    def read(self) -> Tuple[Any, np.ndarray]:
        self.cur += 1
        ret, bgr = self.capture.read()
        return ret, bgr

    def release(self):
        self.capture.release()

    def resetPositionByIndex(self, index: int):
        self.cur = min(max(index, 0), self.num_frame - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.cur)

    def resetPositionByRatio(self, ratio: float):
        self.cur = int(ratio * self.num_frame + 0.5)
        self.cur = min(max(self.cur, 0), self.num_frame - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.cur)

    """
    sample frames
    """
    def sampleFrames(self, beg=0, end=1, step=1):
        assert 0 <= beg < end <= self.num_frame, (beg, end, self.num_frame)
        assert 0 < step < self.num_frame//2, (step, self.num_frame)
        assert self.isOpen() is True
        data = list()
        self.resetPositionByIndex(beg)
        for n in range(end-beg):
            ret, bgr = self.read()
            if ret is False:
                break
            if n % step == 0:
                data.append(bgr)
        return data

    def dumpFrames(self, path_save, step=1, **kwargs):
        assert self.isOpen() is True
        assert 0 < step < self.num_frame
        format_function = kwargs['format'] if 'format' in kwargs \
            else lambda n: '{:05d}.png'.format(n)
        suffix = format_function(0).split('.')[1]
        assert len(suffix) > 0, suffix
        for n in range(self.num_frame):
            _, bgr = self.read()
            if n % step == 0:
                path = '{}/{}'.format(path_save, format_function(n))
                cv2.imencode('.{}'.format(suffix), bgr)[1].tofile(path)


class XVideoReaderOpenCVCache(XVideoReaderOpenCV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_size = kwargs.get('cache_size', 16)
        self.cache = {}
        self.cache_queue = []  # LRU, only index

    def getFrameByIndex(self, index: int) -> np.ndarray:
        if not (0 <= index < len(self)):
            raise IndexError(f"Index {index} out of range [0, {len(self)-1}]")

        # try to get from cache
        if index in self.cache:
            # update LRU
            self.cache_queue.remove(index)
            self.cache_queue.append(index)
            return self.cache[index]

        # reset reader
        self.resetPositionByIndex(index)
        ret, frame = self.read()

        if ret:
            # store in LRU cache
            if len(self.cache) >= self.cache_size:
                old_idx = self.cache_queue.pop(0)
                del self.cache[old_idx]
            self.cache[index] = frame
            self.cache_queue.append(index)
        else:
            raise IOError(f"read frame {index} failed")
        return self.cache[index]

    def __getitem__(self, item):
        """
        support method：
            reader[100] -> positive int
            reader[-1]  -> negative int
            reader[1:5] -> slice
        """
        if isinstance(item, slice):
            # slice
            indices = range(*item.indices(len(self)))
            return [self.getFrameByIndex(i) for i in indices]
        elif isinstance(item, int):
            # negative index
            if item < 0:
                item = len(self) + item
            return self.getFrameByIndex(item)
        else:
            raise TypeError(f"index must be int or slice，but got {type(item)}")

