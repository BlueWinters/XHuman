
import logging
import numpy as np
import cv2
import os
import copy
import torch
import torch.nn.functional as F
from .. import XManager


class LibPortraitParsing:
    @staticmethod
    def get_color_map(N=256, normalized=False):
        def bitget(byt_eval, idx):
            return (byt_eval & (1 << idx)) != 0

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3
            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        return cmap

    @classmethod
    def colorize(cls, seg):
        color_map = cls.get_color_map(N=26)
        return color_map[seg, :]

    """
    """
    ParsingSeqName = [
        'background',
        'skin',
        'right-brow', 'left-brow',
        'right-eye', 'left-eye',
        'nose',
        'mouth', 'upper-lip', 'lower-lip',
        'left-ear', 'right-ear',
        'neck', 'hair', 'eye-glass', 'ear-ring', 'hat', 'clothes', 'necklace',
        'body', 'left-arm', 'right-arm',
        'left-leg', 'right-leg', 'bag', 'adding',
    ]
    ParsingIndexDict = {name: int(n) for n, name in enumerate(ParsingSeqName)}

    @staticmethod
    def getResources():
        raise NotImplementedError

    """
    """
    def __init__(self, *args, **kwargs):
        self.max_size = 512+256
        self.root = None

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        if self.root is None:
            self.root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter

    """
    """
    def inference(self, bgr):
        h, w = bgr.shape[0:2]
        batch_bgr, h_offset, w_offset, h2, w2 = self._format(bgr, 384, 255, 255, 255)
        parsing_soft = self.forward(batch_bgr)
        parsing_final = self.post(parsing_soft, h2, w2, h_offset, w_offset, h, w)
        return parsing_final

    @staticmethod
    def _format(image, size, b, g, r):
        height, width = np.shape(image)[0], np.shape(image)[1]
        ratio = height / width
        if height > width:
            h2 = size
            w2 = int(h2 / ratio)
            image = cv2.resize(image, (w2, h2), interpolation=cv2.INTER_LINEAR)
            pad1 = int((size - w2) / 2)
            pad2 = size - w2 - pad1
            bb = np.pad(image[:, :, 0], ((0, 0), (pad1, pad2)), mode='constant', constant_values=(b, b))
            gg = np.pad(image[:, :, 1], ((0, 0), (pad1, pad2)), mode='constant', constant_values=(g, g))
            rr = np.pad(image[:, :, 2], ((0, 0), (pad1, pad2)), mode='constant', constant_values=(r, r))
            bb = np.expand_dims(bb, axis=2)
            gg = np.expand_dims(gg, axis=2)
            rr = np.expand_dims(rr, axis=2)
            image = np.concatenate((bb, gg, rr), axis=2)
            h_offset = 0
            w_offset = pad1
        else:
            w2 = size
            h2 = int(w2 * ratio)
            image = cv2.resize(image, (w2, h2), interpolation=cv2.INTER_LINEAR)
            pad1 = int((size - h2) / 2)
            pad2 = size - h2 - pad1
            bb = np.pad(image[:, :, 0], ((pad1, pad2), (0, 0)), mode='constant', constant_values=(b, b))
            gg = np.pad(image[:, :, 1], ((pad1, pad2), (0, 0)), mode='constant', constant_values=(g, g))
            rr = np.pad(image[:, :, 2], ((pad1, pad2), (0, 0)), mode='constant', constant_values=(r, r))
            bb = np.expand_dims(bb, axis=2)
            gg = np.expand_dims(gg, axis=2)
            rr = np.expand_dims(rr, axis=2)
            image = np.concatenate((bb, gg, rr), axis=2)
            h_offset = pad1
            w_offset = 0

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = image[np.newaxis, ...] / 255
        return image, h_offset, w_offset, h2, w2

    def forward(self, batch_bgr):
        raise NotImplementedError

    def post(self, parsing_soft_array, h2, w2, h_offset, w_offset, h, w):
        assert isinstance(parsing_soft_array, np.ndarray)
        parsing_soft_src = parsing_soft_array[0, :, h_offset:h2 + h_offset, w_offset:w2 + w_offset]
        seg_pr = np.argmax(parsing_soft_src, axis=0)  # C,H,W --> H,W
        return cv2.resize(seg_pr, (w, h), interpolation=cv2.INTER_NEAREST)

    """
    """
    def _extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        return targets

    def _returnResult(self, output, targets):
        def _formatResult(target):
            if target == 'source':
                return output
            if target == 'visual':
                return self.colorize(output)
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, bgr, *args, **kwargs):
        target = self._extractArgs(*args, **kwargs)
        output = self.inference(bgr)
        return self._returnResult(output, target)


class LibPortraitParsingTorchScript(LibPortraitParsing):
    """
    """
    EngineConfigTorchScript = {
        'type': 'torch',
        'device': 'cuda:0',
        'parameters': 'base/portrait_parsing.ts'
    }

    def __init__(self, *args, **kwargs):
        super(LibPortraitParsingTorchScript, self).__init__()
        self.session_ts = XManager.createEngine(self.EngineConfigTorchScript)

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        if self.root is None:
            self.root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
        self.session_ts.initialize(*args, **kwargs)

    def forward(self, batch_bgr):
        batch_input = batch_bgr
        parsing_soft = self.session_ts.inference(batch_input, detach=False)
        return parsing_soft

    def post(self, parsing_soft_tensor, h2, w2, h_offset, w_offset, h, w):
        parsing_soft_src = parsing_soft_tensor[:, :, h_offset:h2 + h_offset, w_offset:w2 + w_offset]
        parsing_soft_rsz = F.interpolate(parsing_soft_src, (h, w), mode='bilinear', align_corners=True)
        seg_pr_tensor = torch.argmax(parsing_soft_rsz, dim=1, keepdim=False)
        seg_pr = self.session_ts.detach(seg_pr_tensor[0, :, :]).astype(np.uint8)
        return seg_pr


class LibPortraitParsingOnnx(LibPortraitParsing):
    """
    """
    EngineConfigOnnx = {
        'checkpoint': 'base/portrait_parsing.onnx',
        'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
    }

    def __init__(self, *args, **kwargs):
        super(LibPortraitParsingOnnx, self).__init__()
        self.engine = XManager.createEngine(self.EngineConfigOnnx)
        self.root = None

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def forward(self, batch_bgr):
        raise NotImplementedError


class LibPortraitParsingTensorRT(LibPortraitParsing):
    """
    """
    @staticmethod
    def formatPath(path_onnx, mode):
        return '{}-{}.engine'.format(os.path.splitext(path_onnx)[0], mode)

    @staticmethod
    def exportAsTensorRT(root, name, mode, **kwargs):
        from ..utils.tensorrt import build_engine
        assert mode in ['fp32', 'fp16'], 'mode should be in ["fp32", "fp16"]'
        path_onnx = '{}/{}'.format(root, LibPortraitParsingOnnx.EngineConfigOnnx['checkpoint'])
        path_trt = LibPortraitParsingTensorRT.formatPath(path_onnx, mode)
        build_engine(path_onnx, path_trt, 1, mode, None)

    def __init__(self, trt_engine, *args, **kwargs):
        super(LibPortraitParsingTensorRT, self).__init__()
        self.trt_engine = trt_engine
        self.trt_session = XManager.createEngine(self.getConfig())
        self.root = None

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def getConfig(self, device='cuda:0'):
        path_onnx = copy.deepcopy(LibPortraitParsingOnnx.EngineConfigOnnx['checkpoint'])
        path_engine = '{}-{}.engine'.format(path_onnx[:-5], self.trt_engine)
        return dict(type='tensorrt', device='cuda:0', parameters=path_engine)

    def initialize(self, *args, **kwargs):
        if self.root is None:
            self.root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
        self.trt_session.initialize(*args, **kwargs)

    def forward(self, batch_image):
        return self.trt_session.inference(batch_image)[0]


class LibPortraitParsingWrapper:
    """
    """
    Engine = ['ts', 'fp16', 'onnx']

    @staticmethod
    def getResources():
        return [
            LibPortraitParsingOnnx.EngineConfigOnnx['checkpoint'],
            LibPortraitParsingTorchScript.EngineConfigTorchScript['parameters'],
            # LibPortraitParsingTensorRT.EngineConfigTensorRT['parameters'],
        ]

    """
    """
    def __init__(self, *args, **kwargs):
        self.dict = dict()
        self.root = None

    def __getitem__(self, item):
        assert isinstance(item, str), item
        assert item in self.Engine, item
        if item not in self.dict:
            if item == 'ts':
                module = LibPortraitParsingTorchScript()
                module.initialize(root=self.root)
                self.dict[item] = module
            if item == 'onnx':
                module = LibPortraitParsingOnnx()
                module.initialize(root=self.root)
                self.dict[item] = module
            if item == 'fp16' or item == 'int8':
                module = LibPortraitParsingTensorRT(item)
                module.initialize(root=self.root)
                self.dict[item] = module
        return self.dict[item]

    def initialize(self, *args, **kwargs):
        if self.root is None:
            self.root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
