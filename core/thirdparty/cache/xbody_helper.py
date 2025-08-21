
import logging
import os
import cv2
import numpy as np
import typing
from .xbody import XBody
from ...utils.video import XVideoReaderOpenCV


class XBodyHelper:
    @staticmethod
    def dumpXBodyFromFolder(path_dir_in, path_dir_out, suffix='.png'):
        for name in sorted(os.listdir(path_dir_in)):
            if name.endswith(suffix):
                bgr = cv2.imread('{}/{}'.format(path_dir_in, name))
                cache = XBody(bgr)
                path_pkl = '{}/{}.pkl'.format(path_dir_out, os.path.splitext(name)[0])
                cache.save(path_pkl, name_list=['bgr', 'number', 'scores', 'boxes', 'scores26', 'points26'])
            else:
                logging.warning('skip file: {}'.format(name))

    @staticmethod
    def getXBodyIterator(**kwargs):
        if 'path_video' in kwargs:
            return XBodyIteratorVideo(kwargs['path_video'])
        if 'path_image' in kwargs:
            return XBodyIteratorImages(kwargs['path_image'])
        if 'path_pkl' in kwargs:
            return XBodyIteratorBin(kwargs['path_pkl'])
        raise NotImplementedError


class XBodyIteratorVideo:
    def __init__(self, path_video):
        self.path = path_video
        self.reader = XVideoReaderOpenCV(path_video)
        assert self.reader.isOpen(), path_video

    def __iter__(self):
        self.reader.resetPositionByIndex(0)
        return self

    def __next__(self):
        return XBody.packageAsCache(next(self.reader))

    def __len__(self):
        return len(self.reader)


class XBodyIteratorImages:
    def __init__(self, path_image):
        assert os.path.isdir(path_image), path_image
        self.path = path_image
        self.list = sorted(os.listdir(path_image))
        self.iterator = iter(self.list)

    def __iter__(self):
        return self

    def __next__(self):
        path_image = '{}/{}'.format(self.path, next(self.iterator))
        return XBody.packageAsCache(cv2.imdecode(np.fromfile(path_image, dtype=np.uint8), -1))

    def __len__(self):
        return len(self.list)


class XBodyIteratorBin:
    def __init__(self, path_pkl):
        assert os.path.isdir(path_pkl), path_pkl
        self.path = path_pkl
        self.list = sorted(os.listdir(path_pkl))
        self.iterator = iter(self.list)

    def __iter__(self):
        return self

    def __next__(self):
        path_pkl = '{}/{}'.format(self.path, next(self.iterator))
        return XBody.load(path_pkl, verbose=False)

    def __len__(self):
        return len(self.list)

