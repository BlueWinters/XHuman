
import logging
import os
import numpy as np
import cv2
import json
from .. import XManager

try:
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import insightface
    if not insightface.__version__ >= '0.7':
        logging.error('insightface.__version__({}) '
            'should be >= 0.7'.format(insightface.__version__))
except ImportError:
    logging.error('no such module insightface, try to: pip install insightface')


class LibInsightFaceWrapper:
    """
    """
    @staticmethod
    def getResources():
        file_list = {
            'buffalo_l': ['1k3d68.onnx', '2d106det.onnx', 'det_10g.onnx', 'genderage.onnx', 'w600k_r50.onnx'],
        }

        return [
            *['{}/models/buffalo_l/{}'.format(LibInsightFaceWrapper.EngineConfig['folder'], each) for each in file_list['buffalo_l']],
            '{}/inswapper_128.onnx'.format(LibInsightFaceWrapper.EngineConfig['folder']),
        ]

    """
    """
    EngineConfig = {
        'name': 'buffalo_l',
        'device': 'cuda:0',
        'folder': 'thirdparty/insightface',
        'extension': {'swapper': 'inswapper_128.onnx'},
    }

    """
    """
    def __init__(self, *args, **kwargs):
        self.config = self.EngineConfig
        self.h = 640
        self.w = 640

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    """
    """
    def _getContext(self):
        assert 'device' in self.config
        content = self.config['device'].split(':')
        assert content[0] == 'cuda' or content[0] == 'cpu'
        return int(content[1]) if content[0] == 'cuda' else -1

    def initialize(self, *args, **kwargs):
        root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
        path = '{}/{}'.format(root, self.config['folder'])
        assert os.path.exists(path), path
        if hasattr(self, 'config') and hasattr(self, '_application'):
            if self.config['path'] == path: return
            # remove the previous device and module
            logging.warning('remove the previous application')
            if hasattr(self, '_application'): setattr(self, '_application', None)
            if hasattr(self, 'swapper'): setattr(self, 'swapper', None)
        self.config['path'] = path
        self._application = insightface.app.FaceAnalysis(
            name=self.config['name'], root=self.config['path'], download=False)
        self._application.prepare(ctx_id=self._getContext(), det_size=(self.h, self.w))

    """
    get module
    """
    @property
    def application(self):
        if hasattr(self, '_application') is False:
            self.initialize()
        return self._application

    def _getExtendModule(self, name:str):
        if hasattr(self, name) == False:
            assert name in self.config['extension']
            path = '{}/{}'.format(self.config['path'], self.config['extension'][name])
            module = insightface.model_zoo.get_model(
                path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], download=False)
            setattr(self, name, module)
        return getattr(self, name)

    """
    """
    def _assertImage(self, image):
        assert len(image.shape) == 3
        assert image.shape[2] == 3

    """
    pipeline for pair inputs
    """
    def _swapFace(self, image, source_face, target_face, super_resolution):
        swapper = self._getExtendModule('swapper')
        image = swapper.get(image, target_face, source_face, paste_back=True)
        if super_resolution == True:
            face_restoration = XManager.getModules('face_restoration')
            lft, top, rig, bot = np.round(target_face['bbox']).astype(np.int32).tolist()
            scale_h = (bot - top) / 128.
            scale_w = (rig - lft) / 128.
            scale = float((scale_h + scale_w) / 2.)
            image = face_restoration(bgr=image, upscale=1,
                targets='source') if scale > 1 else image
        return image

    def _calculateSimilarity(self, source_face, target_face):
        source_embedding = source_face['embedding']
        target_embedding = target_face['embedding']
        cosine_similarity = source_embedding.dot(target_embedding) / \
                  (np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding))
        return cosine_similarity

    def _extractArgsPair(self, *args, **kwargs):
        if len(args) != 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        process = kwargs.pop(
            'process', ['swap_face+sr', 'calculate_similarity'])
        process = [process] if isinstance(process, str) else process
        for each in process: assert \
            each == 'swap_face' or \
            each == 'swap_face+sr' or \
            each == 'calculate_similarity'
        targets = kwargs.pop('targets', 'source')
        return targets, process

    def _inferenceWithPairInput(self, source_image:np.ndarray, target_image:np.ndarray, process:list):
        self._assertImage(source_image)
        self._assertImage(target_image)
        source_face_list = self.application.get(source_image)
        target_face_list = self.application.get(target_image)
        if len(source_face_list) != 1:
            raise Exception('source image has {} face != 1'.format(len(source_face_list)))
        if len(target_face_list) == 0:
            raise Exception('target image has no face'.format(len(target_face_list)))
        results = dict()
        source_face = source_face_list[0]
        if 'swap_face' in process or 'swap_face+sr' in process:
            super_resolution = bool('swap_face+sr' in process)
            target_image_copy = np.copy(target_image)
            for target_face in target_face_list:
                target_image_copy = self._swapFace(target_image_copy, source_face, target_face, super_resolution)
            results['swap_face'] = target_image_copy
        if 'calculate_similarity' in process:
            similarity_list = list()
            for target_face in target_face_list:
                similarity = self._calculateSimilarity(source_face, target_face)
                similarity_list.append(dict(value=similarity, bbox=target_face['bbox']))
            results['similarity'] = sorted(similarity_list, key=lambda sim: sim['value'])
        return results

    def _returnResultFromPair(self, output, targets):
        def _formatResult(target):
            if target == 'source': return output
            if target == 'json':
                assert 'similarity' in output
                for one in output['similarity']:
                    one['bbox'] = one['bbox'].tolist()
                    one['value'] = float(one['value'])
                output['similarity'] = json.dumps(output['similarity'], indent=4)
                return output
            if target == 'image':
                assert 'swap_face' in output
                return output['swap_face']
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def callWithPairInput(self, source_image: np.ndarray, target_image: np.ndarray, *args, **kwargs):
        target, process = self._extractArgsPair(*args, **kwargs)
        output = self._inferenceWithPairInput(source_image, target_image, process)
        return self._returnResultFromPair(output, target)

    """
    call for single input
    """
    def _extractArgsSingle(self, *args, **kwargs):
        if len(args) != 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        target = kwargs.pop('targets', 'source')
        return target

    def _packageAsDict(self, face, array2list:bool=False):
        data = dict()
        for key, value in face.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist() if array2list == True else value
        data['sex'] = 'M' if face['gender'] == 1 else 'F'
        data['age'] = int(face.age)
        return data

    def _returnResultFromSingle(self, output, targets):
        def _formatResult(target):
            if target == 'source':
                return [self._packageAsDict(face, False) for face in output]
            if target == 'json':
                data = list()
                for n, face in enumerate(output):
                    data.append(self._packageAsDict(face, True))
                return json.dumps(data, indent=4)
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def callWithSingleInput(self, bgr:np.ndarray, *args, **kwargs):
        self._assertImage(bgr)
        target = self._extractArgsSingle(*args, **kwargs)
        source_face_list = self.application.get(bgr)
        return self._returnResultFromSingle(source_face_list, target)

    """
    """
    def __call__(self, *args, **kwargs):
        assert len(args) == 1 or len(args) == 2
        return self.callWithSingleInput(*args, **kwargs) if len(args) == 1 \
            else self.callWithPairInput(*args, **kwargs)
