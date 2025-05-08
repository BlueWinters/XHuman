
import copy
import logging
import os
import math
import numpy as np
import cv2
import json
import skimage
from .cache import *
from .. import XManager


class LibFunction:
    """
    """
    @staticmethod
    def visual(bgr, landmarks, radians):
        face_landmark, head_pose = XManager.getModules(['face_landmark', 'head_pose'])
        for n, (points, radian) in enumerate(zip(landmarks, radians)):
            bgr = face_landmark.visual(bgr, points)
            bgr = head_pose.visual(bgr, radian, points)
        return bgr

    @staticmethod
    def getResources():
        return []

    """
    """
    def __init__(self, *args, **kwargs):
        pass

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    """
    """
    def initialize(self, *args, **kwargs):
        pass

    """
    """
    def __call__(self, function, *args, **kwargs):
        function = getattr(self, function)
        return function(*args, **kwargs)

    """
    """
    @staticmethod
    def formatResults(bgr, scores, boxes, points, landmarks, radians, targets):
        def _formatResult(target):
            if target == 'source':
                return scores, boxes, points, landmarks, radians
            if target == 'json':
                data = list()
                for s, b, p, l, r in zip(scores, boxes, points, landmarks, radians):
                    data.append(dict(scores=s.tolist(), boxes=b.tolist(),
                        points=p.tolist(), landmarks=l.tolist(), randian=r.tolist()))
                return json.dumps(data, indent=4)
            if target == 'visual':
                return LibFunction.visual(np.copy(bgr), landmarks, radians)
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    @staticmethod
    def detect(bgr, targets):
        face_detector, face_landmark, head_pose = XManager.getModules([
            'face_detection', 'face_landmark', 'head_pose'])
        scores, boxes, points = face_detector(bgr)
        landmarks = face_landmark(bgr, boxes=boxes)
        radians = head_pose(bgr, landmarks=landmarks)
        return LibFunction.formatResults(bgr, scores, boxes, points, landmarks, radians, targets)

    """
    """
    @staticmethod
    def packageAsCache(source):
        return XPortrait.packageAsCache(source)

    """
    """
    @staticmethod
    def formatF(transform_type, bgr, points_source, points_target, **kwargs):
        transform = skimage.transform.estimate_transform(transform_type, points_source, points_target)
        order = int(kwargs.pop('order', 1))
        mode = kwargs.pop('mode', 'constant')
        value = kwargs.pop('value', (255, 255, 255))
        output_shape = kwargs['output_shape'] if 'output_shape' in kwargs else None  # h,w
        warped_f = skimage.transform.warp(
            bgr.astype(np.float32), transform.inverse, order=order, mode=mode, cval=value, output_shape=output_shape)
        warped = np.round(warped_f).astype(np.uint8)
        return warped, transform

    @staticmethod
    def transformB(bgr_src, bgr_warped, transform, mask=None, **kwargs):
        h, w, c = bgr_src.shape
        warped2_f = skimage.transform.warp(
            bgr_warped.astype(np.float32), transform, order=1, mode='constant', cval=255, output_shape=(h, w))
        if not isinstance(mask, np.ndarray):
            k = kwargs.pop('kernel', 3)  # blur kernel size
            mask = np.zeros(bgr_warped.shape[:2], dtype=np.uint8)
            mask[k:-k, k:-k] = 255
            mask = cv2.GaussianBlur(mask, (k, k), sigmaX=k, sigmaY=k)
        assert mask.shape == bgr_warped.shape[:2], mask.shape
        assert mask.dtype == np.uint8, mask.dtype
        mask_f = skimage.transform.warp(mask.astype(np.float32), transform, order=1, mode='constant', cval=0, output_shape=(h, w))
        mask_f = mask_f.astype(np.float32)[:, :, None] / 255.
        fusion = warped2_f * mask_f + bgr_src.astype(np.float32) * (1 - mask_f)
        return np.round(fusion).astype(np.uint8)

    """
    """
    @staticmethod
    def applyTransformToImages(transform, image_list):
        assert isinstance(image_list, list), type(image_list)
        result_list = []
        for each in image_list:
            assert isinstance(each, dict), type(each)
            assert hasattr(transform, '__call__'), transform
            result_list.append(transform(**each))
        return result_list

    @staticmethod
    def rotateFaceF(source, landmark=None, **kwargs):
        def _calculateRadian(vector):
            return float(math.atan2(float(vector[1]), float(vector[0])))

        def _getFaceLandmark(cache: XPortrait):
            if landmark is not None and isinstance(landmark, np.ndarray):
                assert landmark.shape == (68, 2), landmark.shape
                return landmark
            return cache.landmark[0]

        cache = LibFunction.packageAsCache(source)
        h, w = cache.shape
        hh = int(kwargs['h']) if 'h' in kwargs else h
        ww = int(kwargs['w']) if 'w' in kwargs else w
        landmark = _getFaceLandmark(cache)
        radian1 = _calculateRadian(landmark[42, :] - landmark[39, :])
        radian2 = _calculateRadian(landmark[45, :] - landmark[36, :])
        radian = (radian1 + radian2) / 2
        degree = radian / math.pi * 180
        center = landmark[27, :].tolist()
        transform = skimage.transform.EuclideanTransform(rotation=radian)

        method = kwargs.pop('method', 'skimage')
        if method == 'skimage':
            order = int(kwargs.pop('order', 1))
            mode = kwargs.pop('mode', 'constant')
            value = kwargs.pop('value', 255)
            param = dict(order=order, mode=mode, cval=value, output_shape=(hh, ww))
            warped_f = skimage.transform.warp(cache.bgr.astype(np.float32), transform.inverse, **param)
            warped = np.round(warped_f).astype(np.uint8)

            def transform_function(**kwargs):
                param_copy = copy.deepcopy(param)
                param_copy.update(kwargs)
                param_copy['inverse_map'] = transform.inverse,
                return skimage.transform.warp(**param_copy)
            # apply transform using: LibFunction.applyTransformToImages(transform_image_call, other_images_list])
            return warped, transform, transform_function
        else:
            matrix = cv2.getRotationMatrix2D(center, degree, scale=1)
            warped = cv2.warpAffine(cache.bgr, matrix, (ww, hh))
            return warped

    @staticmethod
    def alignFacePositionF(source, target, method='skimage', **kwargs):
        def _getOneFaceLandmark(cache, landmark):
            if landmark is not None:
                assert isinstance(landmark, np.ndarray), type(landmark)
                assert landmark.shape == (68, 2), landmark.shape
                return landmark
            return cache.landmark[0]

        def _calculateKeyPoints(landmarks):
            assert len(landmarks) == 68, landmarks.shape
            point1 = np.mean(landmarks[36:42, :], axis=0)
            point2 = np.mean(landmarks[42:48, :], axis=0)
            point4 = landmarks[48, :]
            point5 = landmarks[54, :]
            points = np.stack([point1, point2, point4, point5], axis=0)
            return points.astype(np.float32)

        def _getAlignPoints(cache: XPortrait, landmark):
            landmark = _getOneFaceLandmark(cache, landmark)
            points = _calculateKeyPoints(landmark)
            return points, cache.shape

        cache_source = LibFunction.packageAsCache(source)
        cache_target = LibFunction.packageAsCache(target)
        source_points, source_shape = _getAlignPoints(cache_source, kwargs.pop('source_landmark', None))
        target_points, target_shape = _getAlignPoints(cache_target, kwargs.pop('target_landmark', None))

        if method == 'skimage':
            transform = skimage.transform.AffineTransform()
            if transform.estimate(source_points, target_points) is False:
                raise ValueError('value error in estimation')
            tar_h, tar_w = target_shape
            order = int(kwargs.pop('order', 1))
            mode = kwargs.pop('mode', 'constant')
            value = kwargs.pop('value', 255)
            param = dict(order=order, mode=mode, cval=value, output_shape=(tar_h, tar_w))
            warped_f = skimage.transform.warp(cache_source.bgr.astype(np.float32), transform.inverse, **param)
            warped = np.round(warped_f).astype(np.uint8)

            def transform_function(**inner_kwargs):
                param_copy = copy.deepcopy(param)
                param_copy.update(inner_kwargs)
                param_copy['inverse_map'] = transform.inverse,
                return skimage.transform.warp(**param_copy)
            # apply transform using: LibFunction.applyTransformToImages(transform_image_call, other_images_list])
            return warped, transform, transform_function
        else:
            matrix = cv2.estimateAffinePartial2D(source_points, target_points, method=cv2.LMEDS)[0]
            # warp and crop faces
            flags = int(kwargs.pop('flags', cv2.INTER_LINEAR))
            mode = kwargs.pop('mode', cv2.BORDER_CONSTANT)
            value = kwargs.pop('value', (255, 255, 255))
            param = dict(dsize=target_shape[::-1], flags=flags, borderMode=mode, borderValue=value)
            warped = cv2.warpAffine(cache_source.bgr, matrix, **param)
            return warped, matrix

    @staticmethod
    def alignFacePositionB(bgr_src, bgr_warped, transform):
        return LibFunction.transformB(bgr_src, bgr_warped, transform)

    """
    """
    @staticmethod
    def warpFaceShape(source_bgr, target_bgr, **kwargs):
        def getMatchIndex(source_xcache, target_xcache):
            name_list = kwargs.pop('shape', ['profile', 'eyes'])
            index_list = []
            if 'profile' in name_list:
                index_list.append(np.arange(0, 17, dtype=np.int32))
            if 'eyes' in name_list:
                index_list.append(np.arange(36, 43, dtype=np.int32))
            assert len(index_list) > 0, name_list
            index = np.concatenate(index_list)
            source_shape = np.reshape(source_xcache.landmark[0][index, :], (1, len(index), 2))
            target_shape = np.reshape(target_xcache.landmark[0][index, :], (1, len(index), 2))
            if 'image' in name_list:
                h, w, _ = source_bgr.shape
                box = np.reshape(np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32), (1, 4, 2))
                source_shape = np.concatenate([source_shape, box], axis=1)
                target_shape = np.concatenate([target_shape, box], axis=1)
            matches = [cv2.DMatch(i, i, 0) for i in range(1, source_shape.shape[1] + 1)]
            return matches, source_shape, target_shape

        # TODO: auto-align face position
        # source_bgr, matrix = LibFunction.alignFacePositionF(source_bgr, target_bgr, method='opencv')
        # tps estimation and warp
        xcache_source = LibFunction.packageAsCache(source_bgr)
        xcache_target = LibFunction.packageAsCache(target_bgr)
        method = kwargs.pop('method', 'skimage')
        if method == 'skimage':
            transform = skimage.transform.ThinPlateSplineTransform()
            tar_h, tar_w = xcache_target.shape
            order = int(kwargs.pop('order', 1))
            mode = kwargs.pop('mode', 'constant')
            value = kwargs.pop('value', 255)
            param = dict(order=order, mode=mode, cval=value, output_shape=(tar_h, tar_w))
            warped_f = skimage.transform.warp(xcache_source.bgr.astype(np.float32), transform.inverse, **param)
            warped = np.round(warped_f).astype(np.uint8)

            def transform_function(**kwargs):
                param_copy = copy.deepcopy(param)
                param_copy.update(kwargs)
                param_copy['inverse_map'] = transform
                return skimage.transform.warp(**param_copy)
            # apply transform using: LibFunction.applyTransformToImages(transform_image_call, other_images_list])
            return warped, transform, transform_function
        else:
            matches, source_shape, target_shape = getMatchIndex(xcache_source, xcache_target)
            tps = cv2.createThinPlateSplineShapeTransformer()
            tps.estimateTransformation(target_shape, source_shape, matches)
            flags = int(kwargs.pop('flags', cv2.INTER_LINEAR))
            mode = kwargs.pop('borderMode', cv2.BORDER_CONSTANT)
            value = kwargs.pop('borderValue', (255,255,255))
            param = dict(flags=flags, borderMode=mode, borderValue=value)
            warped = tps.warpImage(source_bgr, **param)
            return warped, tps

    """
    """
    @staticmethod
    def getProfileCurve(xcache: XPortrait, value=1):
        def transform(array):
            origin = np.mean(array, axis=0)
            origin[1] = np.min(array[:, 1]) - 1
            vector = array - origin
            norm = np.linalg.norm(vector, axis=1)
            degree = np.arccos(vector[:, 0] / norm) / np.pi * 180
            return np.stack([degree, norm], axis=1), origin

        def interpolate(array):
            from scipy import interpolate
            assert len(array) > 0
            array, origin = transform(np.array(array, dtype=np.float32))
            beg, end = np.min(array[:, 0]), np.max(array[:, 0])
            xx = np.linspace(beg, end, int(abs(end - beg)) * 8, dtype=np.float32)
            yy = interpolate.interp1d(array[:, 0], array[:, 1])(xx)
            xxx = np.cos(xx * np.pi / 180) * yy + origin[0]
            yyy = np.sin(xx * np.pi / 180) * yy + origin[1]
            return xxx, yyy

        points_profile = xcache.landmark[0][0:17, :]
        xx, yy = interpolate(points_profile)
        xx, yy = xx.round().astype(np.int32), yy.round().astype(np.int32)
        canvas = np.zeros(shape=xcache.shape, dtype=np.uint8)
        canvas[yy, xx, ...] = value
        return canvas
