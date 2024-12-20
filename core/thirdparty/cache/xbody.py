
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
        print(xcache.number, xcache.score, xcache.box)
        print(xcache.points26, xcache.scores26)
        # cv2.imwrite('benchmark/asset/xbody/xbody_segmentation.png', xcache.segmentation)

    """
    """
    @staticmethod
    def packageAsCache(source, **kwargs):
        assert isinstance(source, (np.ndarray, XBody)), type(source)
        cache = XBody(bgr=source, **kwargs) if isinstance(source, np.ndarray) else source
        return cache

    @staticmethod
    def toCache(source, **kwargs):
        assert isinstance(source, (str, np.ndarray, XBody)), type(source)
        if isinstance(source, str):
            if source.endswith('pkl'):
                source = XBody.load(source, verbose=False)
            if source.endswith('png') or source.endswith('jpg'):
                source = cv2.imread(source)
        return XBody.packageAsCache(source, asserting=False)

    """
    global config
    """
    PropertyReadOnly = True
    ThresholdVisualPoints26 = 0.3

    """
    """
    def __init__(self, bgr, **kwargs):
        super(XBody, self).__init__(bgr=bgr)
        self.url = kwargs.pop('url', '127.0.0.0')
        # strategy: 'area', 'score', 'pose
        self.strategy = kwargs.pop('strategy', 'area')
        assert self.strategy in ['area', 'score'], self.strategy
        # assert flag
        self.asserting = kwargs.pop('asserting', False)

    def local(self):
        return bool(self.url == '127.0.0.0')

    def _getModule(self, module):
        if self.local() is True and True:
            from ... import XManager
            return XManager.getModules(module)
        else:
            from ....serving.xruntime import XRuntime
            function = lambda *args, **kwargs: \
                XRuntime(module, url=self.url)(*args, **kwargs)
            return function

    def save(self, path: str, **kwargs):
        name_pair_list = []
        if 'name_list' in kwargs:
            name_list = kwargs['name_list']
            assert isinstance(name_list, (list, tuple))
            for name in name_list:
                assert isinstance(name, str), type(name)
                name = name[1:] if name.startswith('_') else name
                if hasattr(self, name) is False:
                    logging.warning('class {} has no attribute {}'.format(XBody.__name__, name))
                    continue
                outer_name = name
                inner_name = '_{}'.format(name)
                name_pair_list.append((inner_name, outer_name))
        if 'save_all' in kwargs and kwargs['save_all'] is True and len(name_pair_list) == 0:
            for name, value in vars(XBody).items():
                if isinstance(value, property):
                    outer_name = name
                    inner_name = '_{}'.format(name)
                    name_pair_list.append((inner_name, outer_name))
        data = dict()
        for name, _ in name_pair_list:
            data[name] = getattr(self, name)
        pickle.dump(data, open(path, 'wb'))

    @staticmethod
    def load(path: str, verbose=True):
        assert os.path.exists(path), path
        data = pickle.load(open(path, 'rb'))
        xcache = XBody(bgr=data['_bgr'])
        property_list = []
        for name in data:
            # if name == '_bgr': continue
            # name = name[1:] if isinstance(vars(XCachePortrait)[name], property) else name
            # if isinstance(vars(XBody)[name], property) and verbose is True:
            #     logging.warning('loading non-default member {} from {}'.format(name, path))
            setattr(xcache, name, data[name])
            property_list.append(name)
        if verbose is True:
            logging.warning('load from: {}, property list: {}'.format(path, property_list))
        return xcache

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
    def score(self):
        if not hasattr(self, '_scores'):
            self._detectBoxes(self.bgr)
        return self._scores

    @property
    def box(self):
        if not hasattr(self, '_boxes'):
            self._detectBoxes(self.bgr)
        return self._boxes

    def _detectPose(self):
        points26, scores26 = self._getModule('rtmpose')(self.bgr, boxes=self.box)
        self._points26 = np.reshape(np.array(points26, dtype=np.int32), (-1, 26, 2))
        self._scores26 = np.reshape(np.array(scores26, dtype=np.float32), (-1, 26))

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

    @property
    def visual_points26(self):
        if not hasattr(self, '_visual_points26'):
            module = self._getModule('rtmpose')
            self._visual_points26 = module.visualSkeleton(
                np.copy(self.bgr), self.points26, self.scores26, threshold=self.ThresholdVisualPoints26)
        return self._visual_points26

    @property
    def visual_boxes(self):
        if not hasattr(self, '_visual_boxes'):
            module = self._getModule('rtmpose')
            self._visual_boxes = module.visualBoxes(np.copy(self.bgr), self.box)
        return self._visual_boxes

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