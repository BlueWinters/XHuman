
import logging
import os
import cv2
import numpy as np
import json
import pickle
import skimage
from ...base import XCache



class XBody(XCache):
    """
    """
    @staticmethod
    def benchmark_property():
        bgr = cv2.imread('benchmark/asset/xbody/test.png')
        xcache = XBody(bgr)  # , url='http://192.168.130.17:8089/api/'
        print(xcache.number, xcache.scores, xcache.boxes)
        print(xcache.points26, xcache.scores26)
        # cv2.imwrite('benchmark/asset/xbody/xbody_segmentation.png', xcache.segmentation)

    """
    """
    @staticmethod
    def packageAsCache(source):
        assert isinstance(source, (np.ndarray, XBody)), type(source)
        cache = XBody(bgr=source) if isinstance(source, np.ndarray) else source
        return cache

    """
    global config
    """
    PropertyReadOnly = True

    """
    """
    def __init__(self, bgr, **kwargs):
        super(XBody, self).__init__(bgr=bgr)
        self.url = kwargs.pop('url', '127.0.0.0')
        # strategy: 'area', 'score', 'pose
        self.strategy = kwargs.pop('strategy', 'area')
        assert self.strategy in ['area', 'score'], self.strategy
        # assert flag
        self.asserting = kwargs.pop('asserting', True)

    def local(self):
        return bool(self.url == '127.0.0.0')

    def _getModule(self, module):
        if self.local() is True:
            from ... import XManager
            return XManager.getModules(module)
        else:
            from ....serving.xruntime import XRuntime
            function = lambda *args, **kwargs: \
                XRuntime(module, url=self.url)(*args, **kwargs)
            return function

    """
    """
    @staticmethod
    def _resort(strategy, scores, boxes):
        def _return(index):
            return scores[index], boxes[index]

        if strategy == 'area':
            # big --> small
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            return _return(np.argsort(area)[::-1])
        if strategy == 'score':
            # high --> low
            return _return(np.argsort(scores)[::-1])
        raise NotImplementedError('no such sorting strategy: {}'.format(strategy))

    def _detectBoxes(self, bgr):
        scores, boxes = self._getModule('human_detection_yolox')(bgr, targets='source')
        scores, boxes = self._resort(self.strategy, scores, boxes)
        if self.asserting is True:
            # XPortraitExceptionAssert.assertNoFace(len(scores))
            pass
        self._number = len(scores)
        self._scores = np.reshape(scores.astype(np.float32), (-1,))
        self._boxes = np.reshape(np.round(boxes).astype(np.int32), (-1, 4,))

    @property
    def number(self):
        if not hasattr(self, '_number'):
            self._detectBoxes(self.bgr)
        return self._number

    @property
    def scores(self):
        if not hasattr(self, '_scores'):
            self._detectBoxes(self.bgr)
        return self._scores

    @property
    def boxes(self):
        if not hasattr(self, '_boxes'):
            self._detectBoxes(self.bgr)
        return self._boxes

    def _detectPose(self):
        points26, scores26 = self._getModule('rtmpose')(self.bgr, boxes=self.boxes)
        self._points26 = np.reshape(np.array(points26, dtype=np.int32), (-1, 26, 2))
        self._scores26 = np.reshape(np.array(scores26, dtype=np.float32), (-1,))

    @property
    def points26(self):
        if not hasattr(self, '_points26'):
            self._detectPose()
        return self._points26

    @property
    def scores26(self):
        if not hasattr(self, '_scores26'):
            self._detectPose()
        return self._scores26

    """
    """
    @property
    def segmentation(self):
        if not hasattr(self, '_segmentation'):
            self._segmentation = self._getModule('sapiens')(self.bgr, 'segmentation_1b')
        return self._segmentation

    @property
    def depth(self):
        if not hasattr(self, '_depth'):
            self._depth = self._getModule('sapiens')(self.bgr, 'depth_1b')
        return self._depth

    @property
    def normal(self):
        if not hasattr(self, '_normal'):
            self._normal = self._getModule('sapiens')(self.bgr, 'normal_1b')
        return self._normal