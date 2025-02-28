
import logging
import os
import cv2
import numpy as np
import json
import pickle
import skimage
from .xcache import XCache



class XPortrait(XCache):
    """
    """
    @staticmethod
    def benchmark_property():
        suffix = 'bang_lulu'
        bgr = cv2.imread('benchmark/asset/xportrait/xportrait_source_{}.png'.format(suffix))
        xcache = XPortrait(bgr)  # , url='http://192.168.130.17:8089/api/'
        # print(xcache.sex)
        # xcache.load('cache/data/{}/xcache.pkl'.format(suffix))
        print(xcache.score, xcache.box, xcache.points, xcache.landmark, xcache.radian)
        print(xcache.identity_embedding)
        cv2.imwrite('benchmark/asset/xportrait/{}/xcache_parsing.png'.format(suffix), xcache.parsing)
        cv2.imwrite('benchmark/asset/xportrait/{}/xcache_alpha.png'.format(suffix), xcache.alpha)
        cv2.imwrite('benchmark/asset/xportrait/{}/xcache_alpha_hair.png'.format(suffix), xcache.alpha_hair)
        cv2.imwrite('benchmark/asset/xportrait/{}/xcache_alpha_skin.png'.format(suffix), xcache.alpha_skin)
        cv2.imwrite('benchmark/asset/xportrait/{}/xcache_alpha_cloth.png'.format(suffix), xcache.alpha_cloth)
        cv2.imwrite('benchmark/asset/xportrait/{}/xcache_fine_alpha_hair.png'.format(suffix), xcache.fine_alpha_hair)
        cv2.imwrite('benchmark/asset/xportrait/{}/xcache_fine_alpha_brow_left.png'.format(suffix), xcache.fine_alpha_brow_left)
        cv2.imwrite('benchmark/asset/xportrait/{}/xcache_fine_alpha_brow_right.png'.format(suffix), xcache.fine_alpha_brow_right)
        cv2.imwrite('benchmark/asset/xportrait/{}/xcache_fine_alpha_body.png'.format(suffix), xcache.fine_alpha_body)
        cv2.imwrite('benchmark/asset/xportrait/{}/xcache_fine_alpha_head.png'.format(suffix), xcache.fine_alpha_head)
        cv2.imwrite('benchmark/asset/xportrait/{}/xcache_fine_alpha_neck.png'.format(suffix), xcache.fine_alpha_neck)
        cv2.imwrite('benchmark/asset/xportrait/{}/xcache_fine_alpha_face.png'.format(suffix), xcache.fine_alpha_face)
        xcache.save('benchmark/asset/xportrait/{}/xcache.pkl'.format(suffix))

    @staticmethod
    def packageAsCache(source, **kwargs):
        assert isinstance(source, (str, np.ndarray, XPortrait)), type(source)
        if isinstance(source, str):
            if source.endswith('pkl'):
                return XPortrait.load(source, verbose=False)
            if source.endswith('png') or source.endswith('jpg') or source.endswith('bmp'):
                return XPortrait(bgr=cv2.imread(source), **kwargs)
        if isinstance(source, np.ndarray):
            assert len(source.shape) == 2 or (len(source.shape) == 3 and source.shape[2] == 3), source.shape
            return XPortrait(bgr=source, **kwargs)
        if isinstance(source, XPortrait):
            return source
        raise NotImplementedError(source)

    """
    global config
    """
    PropertyReadOnly = True

    """
    """
    def __init__(self, bgr, **kwargs):
        super(XPortrait, self).__init__(bgr=bgr)
        self.url = kwargs.pop('url', '127.0.0.0')
        # detection: 'SDK' or 'Insightface'
        detect_handle = dict(SDK=self._detectWithSDK, InsightFace=self._detectWithInsightface)
        self._detect_handle = detect_handle[kwargs.pop('detect_handle', 'SDK')]
        # strategy: 'area', 'score', 'pose
        self.strategy = kwargs.pop('strategy', 'area')
        assert self.strategy in ['area', 'score', 'pose'], self.strategy
        # rotations: [0, 90, 180, 270]
        self.rotations = kwargs.pop('rotations', [0])
        for rot in self.rotations:
            assert rot in [0, 90, 180, 270], rot
        # assert flag
        self.asserting = kwargs.pop('asserting', False)

    def local(self):
        return bool(self.url == '127.0.0.0')

    def _getModule(self, module):
        if self.local() is True:
            from ... import XManager
            return XManager.getModules(module)
        else:
            from serving.xruntime import XRuntime
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
                    logging.warning('class {} has no attribute {}'.format(XPortrait.__name__, name))
                    continue
                outer_name = name
                inner_name = '_{}'.format(name)
                name_pair_list.append((inner_name, outer_name))
        if 'save_all' in kwargs and kwargs['save_all'] is True and len(name_pair_list) == 0:
            for name, value in vars(XPortrait).items():
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
        xcache = XPortrait(bgr=data['_bgr'])
        property_list = []
        for name in data:
            if name == '_bgr': continue
            # name = name[1:] if isinstance(vars(XCachePortrait)[name], property) else name
            # if isinstance(vars(XCachePortrait)[name], property) and verbose is True:
            #     logging.warning('loading non-default member {} from {}'.format(name, path))
            setattr(xcache, name, data[name])
            property_list.append(name)
        if verbose is True:
            logging.warning('load from: {}, property list: {}'.format(path, property_list))
        return xcache

    def tolist(self):
        # items: 'detect', 'attribute', 'parsing', 'alpha', 'fine_alpha', 'insightface'
        data = list()
        for s, b, p, l, r in zip(self.score, self.box, self.points, self.landmark, self.radian):
            data.append(dict(score=s.tolist(), box=b.tolist(), points=p.tolist(), landmark=l.tolist(), randian=r.tolist()))
        return data  # to json: json.dumps(data, indent=4)

    """
    base attribute for face(s)
    """
    @staticmethod
    def _resort(strategy, scores, boxes, points, landmarks, radians, angles):
        def _return(index):
            return scores[index], boxes[index], points[index], landmarks[index], radians[index], angles[index]

        if strategy == 'area':
            # big --> small
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            return _return(np.argsort(area)[::-1])
        if strategy == 'score':
            # high --> low
            return _return(np.argsort(scores)[::-1])
        if strategy == 'pose':
            # small --> huge
            radians_abs_sum = np.sum(np.abs(radians), axis=1)
            return _return(np.argsort(radians_abs_sum))
        if strategy == 'lft2rig':
            # from left to right
            center_x = (boxes[:, 0] + boxes[:, 2]) // 2
            return _return(np.argsort(center_x))
        if strategy == 'rig2lft':
            # from right to left
            center_x = (boxes[:, 0] + boxes[:, 2]) // 2
            return _return(np.argsort(center_x)[::-1])
        if strategy == 'top2dwn':
            # from top to down
            center_y = (boxes[:, 1] + boxes[:, 3]) // 2
            return _return(np.argsort(center_y))
        if strategy == 'dwn2top':
            # from down to top
            center_y = (boxes[:, 1] + boxes[:, 3]) // 2
            return _return(np.argsort(center_y)[::-1])
        raise NotImplementedError('no such sorting strategy: {}'.format(strategy))

    @staticmethod
    def doNMS(scores, boxes, nms_threshold=0.4):
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_threshold)[0]
            order = order[inds + 1]
        return keep

    def _detectWithSDK(self, bgr):
        from .xexception import XPortraitExceptionAssert
        # module = self._getModule('function')
        # scores, boxes, points, landmarks, radians = module('detect', bgr, 'source')
        # scores, boxes, points, angles = self._getModule('face_detection')(bgr, image_angles=self.rotations)
        # landmarks = self._getModule('face_landmark')(bgr, image_angles=angles, boxes=boxes)
        # radians = self._getModule('head_pose')(bgr, landmarks=landmarks)
        # scores, boxes, points, landmarks, radians, angles = self._resort(
        #     self.strategy, scores, boxes, points, landmarks, radians, angles)
        # if self.asserting is True:
        #     XPortraitExceptionAssert.assertNoFace(len(scores))
        # self._number = len(scores)
        # self._score = np.reshape(scores, (-1,))
        # self._box = np.reshape(boxes, (-1, 4,))
        # self._points = np.reshape(points, (-1, 5, 2))
        # self._landmark = np.reshape(landmarks, (-1, 68, 2))
        # self._radian = np.reshape(radians, (-1, 3))
        # self._angles = np.reshape(angles, (-1,))
        scores, boxes, points, angles = self._getModule('face_detection')(bgr, image_angles=[0])
        landmarks = self._getModule('face_landmark')(bgr, image_angles=angles, boxes=boxes)
        radians = self._getModule('head_pose')(bgr, landmarks=landmarks)
        data = dict(_score=scores, _box=boxes, _points=points, _landmark=landmarks, _radian=radians)
        return len(scores), data

    def _detectWithInsightface(self, bgr):
        module = self._getModule('insightface')
        targets = 'source' if self.local() else 'json'
        data = module(bgr, targets=targets)
        data = data if isinstance(data, list) else json.loads(data[0])  # online or debug
        if len(data) > 0:
            scores = np.array([one['score'] for one in data])
            boxes = np.stack([one['bbox'] for one in data], axis=0)
            points = np.stack([one['kps'] for one in data], axis=0)
            landmarks2d68 = np.stack([one['landmark_2d_106'][module.Index_106To68, :] for one in data], axis=0)
            landmarks3d68 = np.stack([one['landmark_3d_68'] for one in data], axis=0)
            landmarks2d106 = np.stack([one['landmark_2d_106'] for one in data], axis=0)
            radians = np.stack([one['pose'] for one in data], axis=0)[:, np.array([1, 2, 0], dtype=np.int32)] / 180 * np.pi
            age = np.stack([one['age'] for one in data], axis=0)
            sex = [one['sex'] for one in data]
            # clip the value
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, self.shape[1])
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, self.shape[0])
            points[:, :, 0] = np.clip(points[:, :, 0], 0, self.shape[1])
            points[:, :, 1] = np.clip(points[:, :, 1], 0, self.shape[0])
            landmarks2d68[:, :, 0] = np.clip(landmarks2d68[:, :, 0], 0, self.shape[1])
            landmarks2d68[:, :, 1] = np.clip(landmarks2d68[:, :, 1], 0, self.shape[0])
        else:
            scores = np.zeros(shape=(0,), dtype=np.float32)
            boxes = np.zeros(shape=(0, 4), dtype=np.float32)
            points = np.zeros(shape=(0, 4, 2), dtype=np.float32)
            landmarks2d68 = np.zeros(shape=(0, 68, 2), dtype=np.float32)
            landmarks3d68 = np.zeros(shape=(0, 68, 3), dtype=np.float32)
            landmarks2d106 = np.zeros(shape=(0, 106, 2), dtype=np.float32)
            radians = np.zeros(shape=(0, 3), dtype=np.float32)
            age = np.zeros(shape=(0,), dtype=np.float32)
            sex = []
        data_dict = dict(
            _score=scores, _box=boxes, _points=points, _landmark=landmarks2d68, _radian=radians, _age=age, _sex=sex,
            _landmarks3d68=landmarks3d68, _landmarks2d106=landmarks2d106)
        return len(data), data_dict

    def inferenceWithRotation(self, bgr, image_angle=0):
        from ...geometry import GeoFunction
        if image_angle == 0:
            count, data = self._detect_handle(bgr)
            data['_box'] = np.reshape(data['_box'], (-1, 4))
            data['_points'] = np.reshape(data['_points'], (-1, 5, 2))
            data['_landmark'] = np.reshape(data['_landmark'], (-1, 68, 2))
            data['_radian'] = np.reshape(data['_radian'], (-1, 3))
            data['_angles'] = np.zeros(shape=count, dtype=np.int32)
            return count, data
        if image_angle in GeoFunction.CVRotationDict:
            rot = cv2.rotate(bgr, GeoFunction.CVRotationDict[image_angle])
            count, data = self._detect_handle(rot)
            h, w, c = rot.shape
            angle_back = GeoFunction.rotateBack(image_angle)
            data['_box'] = GeoFunction.rotateBoxes(np.reshape(data['_box'], (-1, 4)), angle_back, h, w)
            data['_points'] = GeoFunction.rotatePoints(np.reshape(data['_points'], (-1, 5, 2)), angle_back, h, w)
            data['_landmark'] = GeoFunction.rotatePoints(np.reshape(data['_landmark'], (-1, 68, 2)), angle_back, h, w)
            data['_radian'] = np.reshape(data['_radian'], (-1, 3))
            data['_angles'] = np.ones(shape=count, dtype=np.int32) * image_angle
            return count, data
        raise ValueError('angle {} not in [0,90,180,270]'.format(image_angle))

    def _detect(self, bgr):
        data_list = []
        for value in self.rotations:
            assert isinstance(value, (int, float)), value
            count, data = self.inferenceWithRotation(bgr, value)
            if count > 0:
                data_list.append(data)
        # NMS
        if len(data_list) > 0:
            data_dict = dict()
            data_keys = list(data_list[0].keys())
            for key in data_keys:
                data_dict[key] = np.concatenate([data[key] for data in data_list], axis=0)
            keep = self.doNMS(data_dict['_score'], data_dict['_box'], 0.3)
            self._number = len(keep)
            for key in data_keys:
                setattr(self, key, data_dict[key][keep])
        else:
            self._number = 0
            self._score = np.zeros(shape=(0,), dtype=np.float32)
            self._box = np.zeros(shape=(0, 4), dtype=np.int32)
            self._points = np.zeros(shape=(0, 10), dtype=np.int32)
            self._landmark = np.zeros(shape=(0, 68, 2), dtype=np.int32)
            self._radian = np.zeros(shape=(0, ), dtype=np.float32)
            self._angles = np.zeros(shape=(0,), dtype=np.int32)

    @property
    def number(self):
        if not hasattr(self, '_number'):
            self._detect(self.bgr)
        return self._number

    @property
    def score(self):
        if not hasattr(self, '_score'):
            self._detect(self.bgr)
        return self._score

    @property
    def box(self):
        if not hasattr(self, '_box'):
            self._detect(self.bgr)
        return self._box

    @property
    def points(self):
        if not hasattr(self, '_points'):
            self._detect(self.bgr)
        return self._points

    @property
    def landmark(self):
        if not hasattr(self, '_landmark'):
            self._detect(self.bgr)
        return self._landmark

    @property
    def radian(self):
        if not hasattr(self, '_radian'):
            self._detect(self.bgr)
        return self._radian

    @property
    def angles(self):
        if not hasattr(self, '_angles'):
            self._detect(self.bgr)
        return self._angles

    @property
    def visual_boxes(self):
        if not hasattr(self, '_visual_boxes'):
            self._visual_boxes = self._getModule('face_detection').visual_targets_cv2(
                np.copy(self.bgr), self.score, np.reshape(self.box, (-1, 4)), np.reshape(self.points, (-1, 10)), options=(False, True))
        return self._visual_boxes

    @property
    def visual_landmarks(self):
        if not hasattr(self, '_visual_landmarks'):
            visual = np.copy(self.bgr)
            for n in range(self.number):
                visual = self._getModule('face_landmark').visual(visual, self.landmark[n, ...])
            self._visual_landmarks = visual
        return self._visual_landmarks

    @property
    def visual_headpose(self):
        if not hasattr(self, '_visual_headpose'):
            visual = np.copy(self.bgr)
            for n in range(self.number):
                visual = self._getModule('head_pose').visual(visual, self.radian[n, :], self.landmark[n, ...])
            self._visual_headpose = visual
        return self._visual_headpose

    @property
    def visual_base(self):
        if not hasattr(self, '_visual_base'):
            self._visual_base = self._getModule('function').visual(np.copy(self.bgr), self.landmark, self.radian)
        return self._visual_base

    """
    face identity embedding
    """
    def _analysis(self):
        module = self._getModule('face_attribute')
        sex, age = module(self.bgr, landmarks=self.landmark)
        to_text = lambda sex: 'M' if int(sex) == 0 else 'F'
        self._sex = [to_text(sex[n]) for n in range(len(sex))]
        self._age = age

    @property
    def sex(self):
        if not hasattr(self, '_sex'):
            self._analysis()
        return self._sex

    @property
    def age(self):
        if not hasattr(self, '_age'):
            self._analysis()
        return self._age

    @property
    def glass(self):
        if not hasattr(self, '_glass'):
            # 'no-glass', 'thin-glass', 'thick-glass', 'sun-glass'
            module = self._getModule('compliance_testing')
            self._glass = module(self.bgr, landmarks=self.landmark, targets='string')
        return self._glass

    @property
    def identity_embedding(self):
        if not hasattr(self, '_identity_embedding'):
            identity_embedding = []
            identity_normed_embedding = []
            for n in range(self.number):
                x0, y0, x1, y1 = self.box[n]
                h, w = (y1 - y0), (x1 - x0)
                r = 0.5
                ih, iw = self.shape
                # slightly enlarge the box
                x0 = int(round(max(x0 - r * w, 0)))
                x1 = int(round(min(x1 + r * w, iw)))
                y0 = int(round(max(y0 - r * h, 0)))
                y1 = int(round(min(y1 + r * h, ih)))
                module = self._getModule('insightface')
                targets = 'source' if self.local() else 'json'
                data = module(self.bgr[y0:y1, x0:x1, ...], targets=targets)
                data = data[0] if isinstance(data, list) else json.loads(data[0])[0]  # online or debug
                identity_embedding.append(data['embedding'])
                identity_normed_embedding.append(data['embedding'] / np.linalg.norm(data['embedding']))
            self._identity_embedding = np.reshape(np.array(identity_embedding, dtype=np.float32), (-1, 512))
            self._identity_normed_embedding = np.reshape(np.array(identity_normed_embedding, dtype=np.float32), (-1, 512))
        return self._identity_embedding

    @property
    def identity_normed_embedding(self):
        if not hasattr(self, '_identity_normed_embedding'):
            return self._identity_normed_embedding
        return self._identity_normed_embedding

    """
    extensive attribute for image processing
    """
    @property
    def parsing(self):
        if not hasattr(self, '_parsing'):
            module = self._getModule('portrait_parsing')
            parsing = module(self.bgr)
            self._parsing = parsing if isinstance(parsing, np.ndarray) else parsing[0]
        return self._parsing

    @property
    def alpha(self):
        if not hasattr(self, '_alpha'):
            module = self._getModule('human_matting')
            alpha = module(self.bgr, targets='alpha')
            self._alpha = alpha if isinstance(alpha, np.ndarray) else alpha[0]
        return self._alpha

    @property
    def foreground(self):
        if not hasattr(self, '_foreground'):
            module = self._getModule('human_matting')
            foreground = module(self.bgr, aplpha=self.alpha, targets='foreground')
            self._foreground = foreground if isinstance(foreground, np.ndarray) else foreground[0]
        return self._foreground

    @property
    def composite(self):
        if not hasattr(self, '_composite'):
            module = self._getModule('human_matting')
            composite = module(self.bgr, aplpha=self.alpha, targets='composite')
            self._composite = composite if isinstance(composite, np.ndarray) else composite[0]
        return self._composite

    def _finetuneMatting(self, k: int = 11):
        module = self._getModule('human_fine_matting')
        hair, skin, cloth = module(self.bgr, alpha=self.alpha)
        self._alpha_hair = np.copy(hair)
        self._alpha_skin = np.copy(skin)
        self._alpha_cloth = np.copy(cloth)
        # refine some bug in alpha with parsing
        # mask_hair = np.where((self.parsing == 13), 1, 0).astype(np.uint8)
        # dilate = cv2.dilate(mask_hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        # self._alpha_hair[dilate == 0] = 0

    @property
    def alpha_hair(self):
        if not hasattr(self, '_alpha_hair'):
            self._finetuneMatting()
        return self._alpha_hair

    @property
    def alpha_skin(self):
        if not hasattr(self, '_alpha_skin'):
            self._finetuneMatting()
        return self._alpha_skin

    @property
    def alpha_cloth(self):
        if not hasattr(self, '_alpha_cloth'):
            self._finetuneMatting()
        return self._alpha_cloth

    """
    the result of fine-matting is not good, just refine with parsing
    """

    @staticmethod
    def _dilateRegion(mask, k: int):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilate = cv2.dilate(mask, kernel)
        addition = dilate - mask
        return dilate, addition

    def _splitHairMask(self, k=13):
        # split hair to hair(only), left brow, right brow
        mask_skin = np.where(self.parsing == 1, 1, 0).astype(np.uint8)
        mask_lft_brow = np.where(self.parsing == 2, 1, 0).astype(np.uint8)
        mask_rig_brow = np.where(self.parsing == 3, 1, 0).astype(np.uint8)
        mask_brow = np.clip(mask_lft_brow + mask_rig_brow, 0, 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_brow_dilate = cv2.dilate(mask_brow, kernel)
        mask_brow_dilate[(mask_skin != 1) & (mask_brow != 1)] = 0
        mask_lft_brow_dilate = cv2.dilate(mask_lft_brow, kernel)
        mask_rig_brow_dilate = cv2.dilate(mask_rig_brow, kernel)
        mask_rig_brow_dilate[mask_lft_brow_dilate == 1] = 0  # for overlap case
        self._fine_alpha_hair = np.copy(self.alpha_hair)
        self._fine_alpha_hair[mask_brow_dilate == 1] = 0
        fine_alpha_brow = self.alpha_hair - self._fine_alpha_hair
        self._fine_alpha_brow_left = np.copy(fine_alpha_brow)
        self._fine_alpha_brow_right = np.copy(fine_alpha_brow)
        self._fine_alpha_brow_left[mask_lft_brow_dilate != 1] = 0
        self._fine_alpha_brow_right[mask_rig_brow_dilate != 1] = 0

    def _splitWholeMask(self, k: int = 7):
        # split to head, body
        mask_body = np.where((self.parsing > 15), 1, 0).astype(np.uint8)
        mask_body_dilate = cv2.dilate(mask_body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        mask_body_summary = (((mask_body_dilate == 1) & (self.parsing == 0)) | mask_body).astype(np.uint8)
        self._fine_alpha_body = np.zeros_like(self.alpha)
        self._fine_alpha_body[mask_body_summary == 1] = self.alpha[mask_body_summary == 1]
        self._fine_alpha_head = self.alpha - self._fine_alpha_body

    def _splitHeadMask(self, k: int = 7):
        # split to neck, face
        mask_neck = np.where(self.parsing == 12, 1, 0).astype(np.uint8)
        mask_neck_dilate = cv2.dilate(mask_neck, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        mask_neck_summary = (((mask_neck_dilate == 1) & (self.parsing == 0)) | mask_neck).astype(np.uint8)
        self._fine_alpha_neck = np.zeros_like(self.alpha)
        self._fine_alpha_neck[mask_neck_summary == 1] = self.fine_alpha_head[mask_neck_summary == 1]
        self._fine_alpha_face = self.fine_alpha_head - self._fine_alpha_neck

    @property
    def fine_alpha_hair(self):
        if not hasattr(self, '_fine_alpha_hair'):
            self._splitHairMask()
        return self._fine_alpha_hair

    @property
    def fine_alpha_brow_left(self):
        if not hasattr(self, '_fine_alpha_brow_left'):
            self._splitHairMask()
        return self._fine_alpha_brow_left

    @property
    def fine_alpha_brow_right(self):
        if not hasattr(self, '_fine_alpha_brow_right'):
            self._splitHairMask()
        return self._fine_alpha_brow_right

    @property
    def fine_alpha_head(self):
        if not hasattr(self, '_fine_alpha_head'):
            self._splitWholeMask()
        return self._fine_alpha_head

    @property
    def fine_alpha_body(self):
        if not hasattr(self, '_fine_alpha_body'):
            self._splitWholeMask()
        return self._fine_alpha_body

    @property
    def fine_alpha_face(self):
        if not hasattr(self, '_fine_alpha_face'):
            self._splitHeadMask()
        return self._fine_alpha_face

    @property
    def fine_alpha_neck(self):
        if not hasattr(self, '_fine_alpha_neck'):
            self._splitHeadMask()
        return self._fine_alpha_neck


class XPortraitR(XPortrait):
    """
    """
    @staticmethod
    def benchmark_create():
        def getKeyPoints(landmark):
            assert len(landmark) == 68
            point1 = np.mean(landmark[36:42, :], axis=0)
            point2 = np.mean(landmark[42:48, :], axis=0)
            point4 = landmark[48, :]
            point5 = landmark[54, :]
            points = np.stack([point1, point2, point4, point5], axis=0)
            return points.astype(np.float32)

        suffix = 'auto'
        bgr = cv2.imread('cache/data/xcache_auto_source.png')
        loc = cv2.imread('cache/data/xcache_auto_location.png')
        source_cache = XPortrait(bgr, url='http://192.168.130.17:8089/api/')
        points_source = getKeyPoints(source_cache.landmark[0])
        points_target = getKeyPoints(XPortrait(loc).landmark[0])
        xcache = XPortraitR(source_cache, [points_source, points_target])
        cv2.imwrite('cache/data/{}/xcache_parsing.png'.format(suffix), xcache.parsing)
        cv2.imwrite('cache/data/{}/xcache_alpha.png'.format(suffix), xcache.alpha)
        cv2.imwrite('cache/data/{}/xcache_alpha_hair.png'.format(suffix), xcache.alpha_hair)
        cv2.imwrite('cache/data/{}/xcache_alpha_skin.png'.format(suffix), xcache.alpha_skin)
        cv2.imwrite('cache/data/{}/xcache_alpha_cloth.png'.format(suffix), xcache.alpha_cloth)
        cv2.imwrite('cache/data/{}/xcache_fine_alpha_hair.png'.format(suffix), xcache.fine_alpha_hair)
        cv2.imwrite('cache/data/{}/xcache_fine_alpha_brow_left.png'.format(suffix), xcache.fine_alpha_brow_left)
        cv2.imwrite('cache/data/{}/xcache_fine_alpha_brow_right.png'.format(suffix), xcache.fine_alpha_brow_right)
        cv2.imwrite('cache/data/{}/xcache_fine_alpha_body.png'.format(suffix), xcache.fine_alpha_body)
        cv2.imwrite('cache/data/{}/xcache_fine_alpha_head.png'.format(suffix), xcache.fine_alpha_head)
        cv2.imwrite('cache/data/{}/xcache_fine_alpha_neck.png'.format(suffix), xcache.fine_alpha_neck)
        cv2.imwrite('cache/data/{}/xcache_fine_alpha_face.png'.format(suffix), xcache.fine_alpha_face)

    """
    """
    def __init__(self, source, transform, **kwargs):
        self.source: XPortrait = self._assign(source)  # array or xcache
        self.transform = self._parse(transform)
        output_shape = kwargs.pop('output_shape', self.source.shape)  # h,w
        super(XPortraitR, self).__init__(bgr=self.transformImage(self.source.bgr, 1, *output_shape))

    def _assign(self, source):
        if isinstance(source, XPortrait):
            return source
        if isinstance(source, np.ndarray):
            return XPortrait(bgr=source)
        raise TypeError('invalid input source type: {}'.format(type(source)))

    def _parse(self, transform):
        # points shape should be N,2
        if isinstance(transform, skimage.transform._geometric._GeometricTransform):
            assert hasattr(transform, 'params')
            return transform
        if isinstance(transform, (tuple, list)):
            assert len(transform) == 2 or len(transform) == 3
            source_points, target_points, transform_type = transform if len(transform) == 3 \
                else (*transform, 'similarity')
            return skimage.transform.estimate_transform(transform_type, source_points, target_points)
        if isinstance(transform, dict):
            transform_type = transform['transform_type'] if 'transform_type' in transform else 'similarity'
            source_points, target_points = transform['source_points'], transform['target_points']
            return skimage.transform.estimate_transform(transform_type, source_points, target_points)
        raise ValueError('invalid input transform: {}'.format(transform))

    """
    transform
    """
    def transformImage(self, bgr, method: int, height=None, width=None, value=0):
        # nearest(0), linear(1)
        assert 0 <= method <= 5, 'invalid interpolation method {}'.format(method)
        if height is None or width is None:
            height, width = self.shape
        bgr_f = bgr.astype(np.float32) / 255.
        warped = skimage.transform.warp(bgr_f, self.transform.inverse,
                                        order=method, mode='constant', cval=value, output_shape=(height, width))
        return np.round(warped * 255).astype(np.uint8)

    def transformCoordinates(self, points):
        # shape: A,B,C,...,2
        assert points.shape[-1] == 2, 'invalid coordinate shape: {}'.format(points.shape)
        return np.reshape(self.transform(np.reshape(points, (-1, 2))), (*points.shape[:-1], 2))

    def backwardImage(self, bgr, method: int, height=None, width=None, value=0):
        # nearest(0), linear(1)
        assert 0 <= method <= 5, 'invalid interpolation method {}'.format(method)
        if height is None or width is None:
            height, width = self.shape
        bgr_f = bgr.astype(np.float32) / 255.
        warped = skimage.transform.warp(bgr_f, self.transform,
                                        order=method, mode='constant', cval=value, output_shape=(height, width))
        return np.round(warped * 255).astype(np.uint8)

    """
    base attribute for face(s)
    """
    def _detect(self, bgr):
        self._number = len(self.source.score)
        self._score = np.copy(self.source.score)
        self._box = np.reshape(self.transformCoordinates(np.reshape(self.source.box, (-1, 2, 2))), (-1, 4))
        self._points = self.transformCoordinates(self.source.points)
        self._landmark = self.transformCoordinates(self.source.landmark)
        self._radian = np.copy(self.source.radian)

    """
    face identity embedding
    """
    def _analysis(self):
        self._sex = np.copy(self.source.sex)
        self._age = np.copy(self.source.age)

    @property
    def glass(self):
        if not hasattr(self, '_glass'):
            self._glass = np.copy(self.source.glass)
        return self._glass

    @property
    def identity_embedding(self):
        if not hasattr(self, '_identity_embedding'):
            self._identity_embedding = np.copy(self.source.identity_embedding)
        return self._identity_embedding

    @property
    def identity_normed_embedding(self):
        if not hasattr(self, '_identity_normed_embedding'):
            self._identity_normed_embedding = np.copy(self.source.identity_normed_embedding)
        return self._identity_normed_embedding

    """
    extensive attribute for image processing
    """
    @property
    def parsing(self):
        if not hasattr(self, '_parsing'):
            self._parsing = self.transformImage(self.source.parsing, 0, value=0)
        return self._parsing

    @property
    def alpha(self):
        if not hasattr(self, '_alpha'):
            self._alpha = self.transformImage(self.source.alpha, 1, value=0)
        return self._alpha

    @property
    def foreground(self):
        if not hasattr(self, '_foreground'):
            self._foreground = self.transformImage(self.source.foreground, 1, value=0)
        return self._foreground

    @property
    def composite(self):
        if not hasattr(self, '_composite'):
            self._composite = self.transformImage(self.source.composite, 1, value=0)
        return self._composite

    def _finetuneMatting(self, k: int = 11):
        self._alpha_hair = self.transformImage(self.source.alpha_hair, 1, value=0)
        self._alpha_skin = self.transformImage(self.source.alpha_skin, 1, value=0)
        self._alpha_cloth = self.transformImage(self.source.alpha_cloth, 1, value=0)

    """
    the result of fine-matting is not good, just refine with parsing
    """
    def _splitHairMask(self, k: int = 13):
        self._fine_alpha_hair = self.transformImage(self.source.fine_alpha_hair, 1, value=0)
        self._fine_alpha_brow_left = self.transformImage(self.source.fine_alpha_brow_left, 1, value=0)
        self._fine_alpha_brow_right = self.transformImage(self.source.fine_alpha_brow_right, 1, value=0)

    def _splitWholeMask(self, k: int = 7):
        self._fine_alpha_body = self.transformImage(self.source.fine_alpha_body, 1, value=0)
        self._fine_alpha_head = self.transformImage(self.source.fine_alpha_head, 1, value=0)

    def _splitHeadMask(self, k: int = 7):
        self._fine_alpha_neck = self.transformImage(self.source.fine_alpha_neck, 1, value=0)
        self._fine_alpha_face = self.transformImage(self.source.fine_alpha_face, 1, value=0)