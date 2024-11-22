
import logging
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from .. import XManager



class LibPortraitParsing:
    @staticmethod
    def get_color_map(N=256, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

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
    ParsingIndexDict = {name:int(n) for n, name in enumerate(ParsingSeqName)}

    @staticmethod
    def getResources():
        return [
            LibPortraitParsing.EngineConfig['parameters'],
        ]

    """
    """
    EngineConfig = {
        'type': 'torch',
        'device': 'cuda:0',
        'parameters': 'base/portrait_parsing.ts'
    }

    """
    """
    def __init__(self, *args, **kwargs):
        self.engine = XManager.createEngine(self.EngineConfig)
        self.max_size = 512+256

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        self.engine.initialize(*args, **kwargs)

    """
    """
    def inference(self, bgr):
        h, w = bgr.shape[0:2]
        batch_bgr, h_offset, w_offset, h2, w2 = self._format(bgr, 384, 255, 255, 255)
        parsing_soft = self._forward(batch_bgr)
        parsing_final = self._post(parsing_soft, h2, w2, h_offset, w_offset, h, w)
        return parsing_final

    def _format(self, image, size, b, g, r):
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

    def _forward(self, batch_bgr):
        batch_input = batch_bgr
        parsing_soft = self.engine.inference(batch_input, detach=False)
        return parsing_soft

    def _post(self, parsing_soft_tensor, h2, w2, h_offset, w_offset, h, w):
        parsing_soft_src = parsing_soft_tensor[:, :, h_offset:h2 + h_offset, w_offset:w2 + w_offset]
        parsing_soft_rsz = F.interpolate(parsing_soft_src, (h, w), mode='bilinear', align_corners=True)
        seg_pr_tensor = torch.argmax(parsing_soft_rsz, dim=1, keepdim=False)
        seg_pr = self.engine.detach(seg_pr_tensor[0,:,:]).astype(np.uint8)
        return seg_pr

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
            if target == 'source': return output
            if target == 'visual': return self.colorize(output)
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