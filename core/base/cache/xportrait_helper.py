
import logging
import os
import typing
import cv2
import numpy as np
from .xportrait import XPortrait
from ...utils.video import XVideoReader


class XPortraitHelper:
    """
    """
    @staticmethod
    def getEyesLength(xcache) -> float:
        assert isinstance(xcache, XPortrait), type(xcache)
        lft_eye_len = [np.linalg.norm(xcache.landmark[n][36, :] - xcache.landmark[n][39, :]) for n in range(xcache.number)]
        rig_eye_len = [np.linalg.norm(xcache.landmark[n][42, :] - xcache.landmark[n][45, :]) for n in range(xcache.number)]
        return lft_eye_len, rig_eye_len

    @staticmethod
    def getEyesMeanLength(xcache):
        lft_eye_len, rig_eye_len = XPortraitHelper.getEyesLength(xcache)
        return [(lft_len + rig_len) / 2. for lft_len, rig_len in zip(lft_eye_len, rig_eye_len)]

    @staticmethod
    def getAjna(xcache):
        assert isinstance(xcache, XPortrait), type(xcache)
        return [np.mean(xcache.landmark[n][17:27, :], axis=0) for n in range(xcache.number)]

    @staticmethod
    def getCenterOfEyes(xcache):
        assert isinstance(xcache, XPortrait), type(xcache)
        return [np.mean(xcache.landmark[n][36:48, :], axis=0) for n in range(xcache.number)]

    """
    """
    @staticmethod
    def dumpXPortraitFromFolder(path_dir_in, path_dir_out, suffix='.png'):
        for name in sorted(os.listdir(path_dir_in)):
            if name.endswith(suffix):
                bgr = cv2.imread('{}/{}'.format(path_dir_in, name))
                cache = XPortrait(bgr, asserting=False)
                path_pkl = '{}/{}.pkl'.format(path_dir_out, os.path.splitext(name)[0])
                cache.save(path_pkl, name_list=['bgr', 'number', 'score', 'box', 'points', 'landmark', 'radian'])
            else:
                logging.warning('skip file: {}'.format(name))

    @staticmethod
    def getXPortraitIterator(**kwargs):
        if 'path_video' in kwargs:
            return XPortraitIteratorVideo(kwargs['path_video'])
        if 'path_image' in kwargs:
            return XPortraitIteratorImages(kwargs['path_image'])
        if 'path_pkl' in kwargs:
            return XPortraitIteratorBin(kwargs['path_pkl'])
        raise NotImplementedError


class XPortraitIteratorVideo:
    def __init__(self, path_video):
        self.path = path_video
        self.reader = XVideoReader(path_video)
        assert self.reader.isOpen(), path_video

    def __iter__(self):
        self.reader.resetPositionByIndex(0)
        return self

    def __next__(self):
        return XPortrait.packageAsCache(next(self.reader))

    def __len__(self):
        return len(self.reader)


class XPortraitIteratorImages:
    def __init__(self, path_image):
        assert os.path.isdir(path_image), path_image
        self.path = path_image
        self.list = sorted(os.listdir(path_image))
        self.iterator = iter(self.list)

    def __iter__(self):
        return self

    def __next__(self):
        path_image = '{}/{}'.format(self.path, next(self.iterator))
        return XPortrait.packageAsCache(cv2.imdecode(np.fromfile(path_image, dtype=np.uint8), -1))

    def __len__(self):
        return len(self.list)


class XPortraitIteratorBin:
    def __init__(self, path_pkl):
        assert os.path.isdir(path_pkl), path_pkl
        self.path = path_pkl
        self.list = sorted(os.listdir(path_pkl))
        self.iterator = iter(self.list)

    def __iter__(self):
        return self

    def __next__(self):
        path_pkl = '{}/{}'.format(self.path, next(self.iterator))
        return XPortrait.load(path_pkl, verbose=False)

    def __len__(self):
        return len(self.list)
