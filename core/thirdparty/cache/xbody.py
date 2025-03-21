
import logging
import os
import cv2
import numpy as np
import json
import pickle
import skimage
from ...base import XCache
from ...utils import Colors


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
    def setDefaultBackend(backend):
        assert 'rtmlib' == backend or 'ultralytics' in backend, backend
        XBody.DefaultBackend = backend

    @staticmethod
    def packageAsCache(source, **kwargs):
        assert isinstance(source, (str, np.ndarray, XBody)), type(source)
        if isinstance(source, str):
            if source.endswith('pkl'):
                return XBody.load(source, verbose=False)
            if source.endswith('png') or source.endswith('jpg') or source.endswith('bmp'):
                return XBody(bgr=cv2.imread(source), **kwargs)
        if isinstance(source, np.ndarray):
            assert len(source.shape) == 2 or (len(source.shape) == 3 and source.shape[2] == 3), source.shape
            return XBody(bgr=source, **kwargs)
        if isinstance(source, XBody):
            return source
        raise NotImplementedError(source)

    """
    global config
    """
    PropertyReadOnly = True
    ThresholdVisualPoints26 = 0.3
    DefaultBackend = 'rtmlib'

    """
    """
    def __init__(self, bgr, **kwargs):
        super(XBody, self).__init__(bgr=bgr)
        self.url = kwargs.pop('url', '127.0.0.0')
        self.backend = kwargs.pop('backend', self.DefaultBackend)
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
        if self.backend == 'rtmlib':
            scores, boxes = self._getModule('human_detection_yolox')(bgr, targets='source')
            scores, boxes = self._resort(self.strategy, scores, boxes)
            if self.asserting is True:
                pass
            self._number = len(scores)
            self._score = np.reshape(scores.astype(np.float32), (-1,))
            self._box = np.reshape(np.round(boxes).astype(np.int32), (-1, 4,))
        if 'ultralytics' in self.backend:
            name = self.backend.split('.')[1]  # example: 'ultralytics.yolo11n-pose' ==> ultralytics.{LibUltralyticsWrapper.ModelList[n]}
            module = self._getModule('ultralytics')[name]
            result = module(bgr, classes=[0], verbose=False)[0]
            self._number = len(result)
            self._score = np.reshape(result.boxes.conf.cpu().numpy().astype(np.float32), (-1,))
            self._box = np.reshape(np.round(result.boxes.xyxy.cpu().numpy()).astype(np.int32), (-1, 4,))
            self._points17 = None
            self._scores17 = None
            self._instances_mask = None
            # assign the property
            if result.keypoints is not None:
                points17 = np.reshape(result.keypoints.data.cpu().numpy().astype(np.float32), (-1, 17, 3))
                self._points17 = np.reshape(points17[:, :, :2], (-1, 17, 2))
                self._scores17 = np.reshape(points17[:, :, 2], (-1, 17))
            if result.masks is not None:
                h, w = self.shape
                self._instances_mask = np.zeros(shape=(self._number, h, w), dtype=np.uint8)
                if len(result) > 0:
                    masks = np.round(result.masks.cpu().numpy().data * 255).astype(np.uint8)  # note: C,H,W and [0,1]
                    for n in range(self._number):
                        self._instances_mask[n, :, :] = cv2.resize(masks[n, :, :], (w, h))

    @property
    def number(self):
        if not hasattr(self, '_number'):
            self._detectBoxes(self.bgr)
        return self._number

    @property
    def score(self):
        if not hasattr(self, '_score'):
            self._detectBoxes(self.bgr)
        return self._score

    @property
    def box(self):
        if not hasattr(self, '_box'):
            self._detectBoxes(self.bgr)
        return self._box

    """
    """
    def _detectPoseWithRTMPose(self):
        points26, scores26 = self._getModule('rtmpose')(self.bgr, boxes=self.box)
        self._points26 = np.reshape(np.array(points26, dtype=np.int32), (-1, 26, 2))
        self._scores26 = np.reshape(np.array(scores26, dtype=np.float32), (-1, 26))

    @property
    def points26(self):
        if not hasattr(self, '_points26'):
            self._detectPoseWithRTMPose()
        return self._points26

    @property
    def scores26(self):
        if not hasattr(self, '_scores26'):
            self._detectPoseWithRTMPose()
        return self._scores26

    @property
    def points17(self):
        if not hasattr(self, '_points17'):
            self.backend = 'ultralytics.yolo11n-pose'
            self._detectBoxes(self.bgr)
        return self._points17

    @property
    def scores17(self):
        if not hasattr(self, '_scores17'):
            self.backend = 'ultralytics.yolo11n-pose'
            self._detectBoxes(self.bgr)
        return self._scores17

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

    @property
    def visual_instances(self):
        if not hasattr(self, '_visual_instances'):
            self.backend = 'ultralytics.yolo11m-seg'
            self._detectBoxes(self.bgr)
            self._visual_instances = np.copy(self.bgr)
            for n in range(len(self._instances_mask)):
                mask = self._instances_mask[n, :, :]
                self._visual_instances[mask > 0] = self._visual_instances[mask > 0] / 255 * Colors.getColor(n, bgr=True)
        return self._visual_instances

    """
    """
    @property
    def portrait_parsing(self):
        if not hasattr(self, '_portrait_parsing'):
            module = self._getModule('portrait_parsing')
            h, w = self.shape
            self._portrait_parsing = np.zeros(shape=(self.number, h, w), dtype=np.uint8)
            for n in range(self.number):
                lft, top, rig, bot = self.box[n, :]
                self._portrait_parsing[n, top:bot, lft:rig] = module(self.bgr[top:bot, lft:rig, :])
        return self._portrait_parsing

    @property
    def visual_portrait_parsing(self):
        if not hasattr(self, '_visual_portrait_parsing'):
            module = self._getModule('portrait_parsing')
            h, w = self.shape
            self._visual_portrait_parsing = np.zeros(shape=(self.number, h, w, 3), dtype=np.uint8)
            for n in range(len(self.portrait_parsing)):
                self._visual_portrait_parsing[n, :, :, :] = module.colorize(self.portrait_parsing[n, :, :])
        return self._visual_portrait_parsing

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