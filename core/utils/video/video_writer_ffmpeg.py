
import copy
import os
import logging
import cv2
import numpy as np
import ffmpeg
from typing import List


class XVideoWriterFFMpeg:
    """
    """
    @staticmethod
    def visualFrameIndex(bgr, n, color=(255, 255, 255)):
        h, w, c = bgr.shape
        rect_th = max(round((w + h) / 2 * 0.003), 2)
        text_th = max(rect_th - 1, 2)
        text_size = rect_th / 4
        points_x = int(w * 0.05)
        points_y = points_x
        cv2.putText(bgr, str(n), (points_x, points_y), 0, text_size, color, thickness=text_th)
        return bgr

    """
    io-config
    """
    ConfigFastCPU = {
        'kwargs_input': {
            'format': 'rawvideo',
            'pix_fmt': 'bgr24',
            'r': 30,
            're': None,
        },
        'kwargs_output': {
            # 'r': 30,
            'pix_fmt': 'yuv420p',
            'vcodec': 'libx264',
            'crf': 0,
            'preset': 'ultrafast',
            'tune': 'zerolatency',
        },
    }

    ConfigFastGPU = {
        'kwargs_input': {
            'format': 'rawvideo',
            'pix_fmt': 'bgr24',
            'r': 30,
            're': None,
        },
        'kwargs_output': {
            'r': 30,
            'pix_fmt': 'yuv420p',
            'vcodec': 'h264_nvenc',
            'cq': 1,
            'preset': 'hq',
            # others
        },
    }

    ConfigHighPixel = {
        'kwargs_input': {
            'format': 'rawvideo',
            'pix_fmt': 'bgr24',
            'r': 30,
            're': None,
        },
        'kwargs_output': {
            'r': 30,
            'vcodec': 'ffv1',
        },
    }

    """
    """
    def __init__(self, config: dict, **kwargs):
        self.path = None
        self.ffmpeg_writer = None
        self.counter = 0
        self.kwargs_input = None
        self.kwargs_output = None
        self.visual_index = None
        self.verbose = None
        self.suffix = None  # force to replace file suffix
        self.fps = config.pop('fps', 30)
        self.w = config.pop('w', -1)
        self.h = config.pop('h', -1)
        self.verbose = False
        self.extractArgs(**kwargs)

    def __del__(self):
        self.release()

    def getFFMpegConfig(self, item):
        return dict(
            fast_cpu=self.ConfigFastCPU,
            fast_gpu=self.ConfigFastGPU,
            highpixel=self.ConfigHighPixel,
        )[item]

    def extractArgs(self, **kwargs):
        item = kwargs.pop('ffmpeg_io_config', 'fast_cpu')
        ffmpeg_config = self.getFFMpegConfig(item)
        self.kwargs_input = ffmpeg_config['kwargs_input']
        self.kwargs_output = ffmpeg_config['kwargs_output']
        self.visual_index = kwargs.pop('visual_index', False)
        self.verbose = kwargs.pop('verbose', False)
        # replace
        if 'kwargs_input' in kwargs:
            self.kwargs_input = kwargs.pop('kwargs_input')
        if 'kwargs_output' in kwargs:
            self.kwargs_output = kwargs.pop('kwargs_output')
        if self.fps > 0 and 'r' in self.kwargs_output:
            self.kwargs_output['r'] = self.fps
        if self.fps > 0 and 'r' in self.kwargs_input:
            self.kwargs_input['r'] = self.fps
        if self.h > 0 and self.w > 0:
            self.kwargs_input['s'] = '{}x{}'.format(self.w, self.h)
        if 'suffix' in kwargs:
            self.suffix = kwargs.pop('suffix')
            assert self.suffix[0] == '.', self.suffix  # '.avi' or '.mp4'
        if 'verbose' in kwargs:
            self.verbose = bool(kwargs.pop('verbose'))

    """
    """
    @property
    def writer(self):
        if self.ffmpeg_writer is None:
            dir_name = os.path.dirname(self.path)
            if len(dir_name) > 0:
                assert os.path.exists(dir_name), self.path
            assert self.h > 0 or self.w > 0, (self.w, self.h)
            self.kwargs_input['s'] = '{}x{}'.format(self.w, self.h)
            if self.verbose is True:
                logging.info('kwargs_input: ', self.kwargs_input)
                logging.info('kwargs_output: ', self.kwargs_output)
            self.ffmpeg_writer = (
                ffmpeg
                .input('pipe:', **self.kwargs_input)
                .output(self.path, **self.kwargs_output)
                .overwrite_output()
                .global_args('-loglevel', 'error')
                .global_args('-y')
                .run_async(pipe_stdin=True))
        return self.ffmpeg_writer

    def open(self, path: str):
        if self.path is None:
            self.path = copy.copy(path)
            if self.suffix is not None:
                prefix, _ = os.path.splitext(self.path)
                self.path = '{}{}'.format(prefix, self.suffix)
            if self.h != -1 and self.w != -1:
                _ = self.writer
        return self.path

    def isOpen(self):
        return bool(self.writer is not None)

    def release(self, *args, **kwargs):
        if self.ffmpeg_writer is not None:
            self.ffmpeg_writer.stdin.close()
            self.ffmpeg_writer.wait()
            self.ffmpeg_writer = None
            return True
        return False

    def write(self, image: np.ndarray):
        assert len(image.shape) == 3 and image.shape[2] == 3, image.shape
        if self.h == -1 or self.w == -1:
            self.h, self.w = image.shape[:2]
        assert self.h == image.shape[0] and self.w == image.shape[1], (self.h, self.w, image.shape)
        if self.visual_index is True:
            image = self.visualFrameIndex(np.copy(image), self.counter)  # start with 0
        self.counter += 1
        self.writer.stdin.write(image.astype(np.uint8).tobytes())

    def dump(self, image_list: List[np.ndarray]):
        for index, image in image_list:
            assert self.counter == index, (self.counter, index)
            self.write(image)

