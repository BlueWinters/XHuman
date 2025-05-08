
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
    def getEyesLength(cache):
        assert isinstance(cache, XPortrait), type(cache)
        lft_eye_len = [np.linalg.norm(cache.landmarks[n][36, :] - cache.landmarks[n][39, :]) for n in range(cache.number)]
        rig_eye_len = [np.linalg.norm(cache.landmarks[n][42, :] - cache.landmarks[n][45, :]) for n in range(cache.number)]
        return lft_eye_len, rig_eye_len

    @staticmethod
    def getEyesMeanLength(cache):
        lft_eye_len, rig_eye_len = XPortraitHelper.getEyesLength(cache)
        return [(lft_len + rig_len) / 2. for lft_len, rig_len in zip(lft_eye_len, rig_eye_len)]

    @staticmethod
    def getAjna(cache):
        assert isinstance(cache, XPortrait), type(cache)
        ajna_list = []
        for n in range(cache.number):
            ajna_list.append(np.mean(cache.landmarks[n][17:27, :], axis=0))
        return np.reshape(np.array(ajna_list, dtype=np.float32), (cache.number, 2))  # N,2

    @staticmethod
    def getCenterOfEyes(cache):
        assert isinstance(cache, XPortrait), type(cache)
        center_both_eyes_list = []
        for n in range(cache.number):
            center_both_eyes_list.append(np.mean(cache.landmarks[n][36:48, :], axis=0))
        return np.reshape(np.array(center_both_eyes_list, dtype=np.float32), (cache.number, 2))  # N,2

    @staticmethod
    def getCenterOfEachEyes(cache):
        assert isinstance(cache, XPortrait), type(cache)
        center_each_eyes_list = []
        for n in range(cache.number):
            center_each_eyes_list.append((np.mean(cache.landmarks[n, 36:40, :], axis=0), np.mean(cache.landmarks[n, 42:48, :], axis=0)))
        return np.stack(center_each_eyes_list, axis=0)  # N,2,2

    @staticmethod
    def getFaceRegionByLandmark(h, w, landmark, top_line='brow', value=255):
        def getTopLinePoints(points):
            if top_line == 'brow':
                pts_rig = points[22:27, :][::-1, :]
                pts_lft = points[17:22, :][::-1, :]
                return pts_rig, pts_lft
            if top_line == 'eye':
                pts_rig = points[42:46, :][::-1, :]
                pts_lft = points[36:40, :][::-1, :]
                return pts_rig, pts_lft
            if top_line == 'brow-eye':
                points_eye_rig = points[42:46, :][::-1, :]
                points_eye_lft = points[36:40, :][::-1, :]
                points_brow_rig = points[22:26, :][::-1, :]
                points_brow_lft = points[18:22, :][::-1, :]
                pts_rig = np.round((points_eye_rig + points_brow_rig) / 2).astype(np.int32)
                pts_lft = np.round((points_eye_lft + points_brow_lft) / 2).astype(np.int32)
                return pts_rig, pts_lft
            raise NotImplementedError(top_line)

        assert landmark.shape[0] == 68 and landmark.shape[1] == 2, landmark.shape
        points_rig, points_lft = getTopLinePoints(landmark)
        points_profile = landmark[0:17, :]
        points_all = np.concatenate([points_profile, points_rig, points_lft], axis=0).round().astype(np.int32)
        mask_face = np.zeros(shape=(h, w), dtype=np.uint8)
        cv2.fillPoly(mask_face, [points_all], (value, value, value))
        return mask_face

    @staticmethod
    def getFaceRegion(cache, index=None, top_line='brow', value=255):
        assert isinstance(cache, XPortrait), type(cache)
        mask_list = []
        index_list = [index] if isinstance(index, int) else list(range(cache.number))
        h, w = cache.shape
        for n in index_list:
            landmark = cache.landmarks[n]
            mask_list.append(XPortraitHelper.getFaceRegionByLandmark(h, w, landmark, top_line, value))
        return mask_list

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
