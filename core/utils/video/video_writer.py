
import copy
import os
import logging
import cv2
import ffmpeg
import numpy as np
import platform
import typing


class XVideoWriter:
    """
    """
    @staticmethod
    def getBackend(name):
        return dict(linux='opencv', windows='opencv')[name]

    @staticmethod
    def visualFrameNumber(bgr, n: int, color=(255, 255, 255)):
        h, w, c = bgr.shape
        rect_th = max(round((w + h) / 2 * 0.003), 2)
        text_th = max(rect_th - 1, 2)
        text_size = rect_th / 4
        points_x = int(w * 0.05)
        points_y = points_x
        cv2.putText(bgr, str(n), (points_x, points_y), 0, text_size, color, thickness=text_th)
        return bgr

    """
    """
    def __init__(self, config: dict):
        self.path = None
        self.platform = platform.system().lower()
        self.backend = config.pop('backend', self.getBackend(self.platform))
        self.fourcc = config.pop('fourcc', self.getDefault('fourcc'))
        self.fps = config.pop('fps', self.getDefault('fps'))
        self.w = config.pop('w', -1)
        self.h = config.pop('h', -1)
        # function
        self.function_write = None
        self.function_initialize = getattr(self, '{}_initialize'.format(self.backend))
        self.function_open = getattr(self, '{}_open'.format(self.backend))
        self.function_release = getattr(self, '{}_release'.format(self.backend))
        self.counter = 0
        self.visual_index = bool(config.pop('visual_index', False))
        # opencv
        self.opencv_writer = None
        # ffmpeg
        self.ffmpeg_writer = None
        self.buffer_size = config.pop('buffer_size', self.getDefault('buffer_size'))

    def __del__(self):
        self.release()

    def __str__(self):
        return 'platform={}, backend={}, fourcc={}, fps={}, w={}, h={}'.format(
            self.platform, self.backend, self.fourcc, self.fourcc, self.w, self.h)

    def getDefault(self, key):
        return {
            'windows': {
                'fourcc': ('X', 'V', 'I', 'D'),
                'suffix': '.avi',
                'fps': 30,
                'buffer_size': None,
            },
            'linux': {
                'fourcc': ('X', 'V', 'I', 'D'),
                'suffix': '.avi',
                'fps': 30,
                'buffer_size': '1024K',
            },
        }[self.platform][key]

    def open(self, path: str) -> str:
        return self.function_open(path)

    def release(self) -> bool:
        return self.function_release()

    def write(self, image: np.ndarray):
        assert len(image.shape) == 3 and image.shape[2] == 3, image.shape
        if self.h == -1 or self.w == -1:
            self.h, self.w = image.shape[:2]
            self.function_initialize()
        assert self.h == image.shape[0] and self.w == image.shape[1], (self.h, self.w, image.shape)
        self.counter += 1
        if self.visual_index is True:
            image = self.visualFrameNumber(np.copy(image), self.counter)
        self.function_write(image)

    def dump(self, data_list: list):
        for index, bgr in data_list:
            assert self.counter == index, (self.counter, index)
            self.write(bgr)

    """
    opencv
    """
    def opencv_initialize(self):
        if self.opencv_writer is None and self.function_write is None:
            assert len(self.fourcc) == 4, self.fourcc
            # if self.backend == 'opencv':
            #     self.fourcc = self.getDefault('fourcc')
            video_writer = cv2.VideoWriter()
            code = cv2.VideoWriter_fourcc(*self.fourcc)
            video_writer.open(self.path, code, self.fps, (self.w, self.h), True)
            if video_writer.isOpened() is False:
                video_writer.release()
                logging.warning('open file fail: {}'.format(self.path))
                raise IOError(self.path)
            self.opencv_writer = video_writer
            self.function_write = lambda bgr: self.opencv_writer.write(bgr)
        else:
            logging.warning('the writer has initialize: {}'.format(str(self)))

    def opencv_open(self, path: str) -> str:
        if self.path is None:
            assert os.path.exists(os.path.dirname(path)), self.path
            name, suffix = os.path.splitext(path)
            suffix_new = self.getDefault('suffix')
            self.path = '{}{}'.format(name, suffix_new)
            if self.h != -1 and self.w != -1:
                logging.warning('backend opencv has reset file suffix: {} -> {}'.format(suffix, suffix_new))
                self.opencv_initialize()
        return self.path

    def opencv_release(self) -> bool:
        if isinstance(self.opencv_writer, cv2.VideoWriter) and self.opencv_writer.isOpened():
            self.opencv_writer.release()
            return True
        return False

    """
    ffmpeg
    """
    def ffmpeg_initialize(self):
        if self.ffmpeg_writer is None and self.function_write is None:
            dir_name = os.path.dirname(self.path)
            if len(dir_name) > 0:
                assert os.path.exists(dir_name), self.path
            assert self.h > 0 or self.w > 0, (self.w, self.h)
            self.ffmpeg_writer = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(self.w, self.h))
                .output(self.path, pix_fmt='yuv420p', r=self.fps, vcodec='libx264', bufsize=self.buffer_size)
                .overwrite_output()
                .run_async(pipe_stdin=True))
            self.function_write = lambda bgr: self.ffmpeg_writer.stdin.write(bgr.astype(np.uint8).tobytes())
        else:
            logging.warning('the writer has initialize: {}'.format(str(self)))

    def ffmpeg_open(self, path: str) -> str:
        if self.path is None:
            self.path = copy.copy(path)  # the final path for writing
            if self.h != -1 and self.w != -1:
                self.ffmpeg_initialize()
        return self.path

    def ffmpeg_release(self):
        if self.ffmpeg_writer is not None:
            self.ffmpeg_writer.stdin.close()
            self.ffmpeg_writer.wait()
            self.ffmpeg_writer = None
            return True
        return False


