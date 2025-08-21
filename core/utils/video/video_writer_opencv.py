
import os
import logging
import cv2
import numpy as np
import typing
import platform


class XVideoWriterOpenCV:
    """
    """
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

    Default = {
        'fourcc': ('X', 'V', 'I', 'D'),
        'suffix': '.avi',
        'fps': 30,
    }

    """
    """
    def __init__(self, config: dict):
        self.path = None
        self.fourcc = config.get('fourcc', self.Default['fourcc'])
        self.fps = config.get('fps', self.Default['fps'])
        self.w = config.get('w', -1)
        self.h = config.get('h', -1)
        self.capture = None
        self.counter = 0
        self.visual_index = bool(config.get('visual_index', False))

    def __del__(self):
        self.release()

    def __str__(self):
        return 'fourcc={}, fps={}, w={}, h={}, path={}'.format(
            self.fourcc, self.fourcc, self.w, self.h, self.path)

    @staticmethod
    def reformatFourcc(suffix, fourcc: str):
        if platform.system().lower() == 'windows':
            return dict(avi='XVID', mp4='AVC1')[suffix.lower()]
        return fourcc

    @property
    def writer(self):
        if self.capture is None:
            assert len(self.fourcc) == 4, self.fourcc
            assert isinstance(self.path, str)
            assert self.h > 0 and self.w > 0, (self.h, self.w)
            self.capture = cv2.VideoWriter()
            fourcc = self.reformatFourcc(os.path.splitext(self.path)[1][1:], self.fourcc)
            code = cv2.VideoWriter_fourcc(*fourcc)
            self.capture.open(self.path, code, self.fps, (self.w, self.h), True)
            if self.capture.isOpened() is False:
                self.capture.release()
                logging.warning('the writer has initialize: {}'.format(str(self)))
                raise IOError(self.path)
        return self.capture

    def open(self, path: str) -> str:
        if self.path is None:
            path_dir = os.path.dirname(path)
            assert os.path.exists(path_dir), path_dir
            name, suffix = os.path.splitext(path)
            suffix_new = self.Default['suffix']
            self.path = '{}{}'.format(name, suffix_new)
            if self.h != -1 and self.w != -1:
                if suffix != suffix_new:
                    logging.warning('backend opencv has reset file suffix: {} -> {}'.format(suffix, suffix_new))
                _ = self.writer  # initialize capture
        return self.path

    def release(self) -> bool:
        if isinstance(self.capture, cv2.VideoWriter) and self.capture.isOpened():
            self.capture.release()
            return True
        return False

    def write(self, image: np.ndarray):
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3 and image.shape[2] == 3, image.shape
        if self.h == -1 or self.w == -1:
            self.h, self.w = image.shape[:2]
            _ = self.writer   # initialize capture
        assert self.h == image.shape[0] and self.w == image.shape[1], (self.h, self.w, image.shape)
        if self.visual_index is True:
            image = self.visualFrameNumber(np.copy(image), self.counter)  # visualize index from 0
        self.counter += 1
        self.writer.write(image)

    def dump(self, frame_list: typing.List):
        for index, frame_bgr in frame_list:
            assert self.counter == index, (self.counter, index)
            self.write(frame_bgr)

