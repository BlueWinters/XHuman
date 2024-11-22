
import logging
import numpy as np
import cv2
from ... import XManager



class LibHumanFineMatting:
    """
    """
    @staticmethod
    def getResources():
        return [
            LibHumanFineMatting.EngineConfig['parameters'],
        ]

    """
    """
    EngineConfig = {
        'type': 'torch',
        'device': 'cuda:0',
        'parameters': 'base/human_fine_matting.ts'
    }

    """
    """
    def __init__(self, *args, **kwargs):
        self.engine = XManager.createEngine(self.EngineConfig)
        self.max_size = 512+256

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    """
    """
    def initialize(self, *args, **kwargs):
        self.engine.initialize(*args, **kwargs)

    def _assert(self, bgr):
        assert len(bgr.shape) == 3

    def _format(self, image, alpha):
        height, width = image.shape[0:2]
        height2 = int(np.ceil(height / 16) * 16)
        width2  = int(np.ceil(width  / 16) * 16)

        pad_h1 = int(np.round((height2 - height) / 2))
        pad_w1 = int(np.round((width2  - width)  / 2))

        image_pad = np.ones([height2, width2, 3]) * 255
        alpha_pad = np.zeros([height2, width2])

        image_pad[pad_h1:pad_h1 + height, pad_w1:pad_w1 + width, :] = image
        alpha_pad[pad_h1:pad_h1 + height, pad_w1:pad_w1 + width] = alpha

        image_pad = np.transpose(image_pad, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)/255
        alpha_pad = alpha_pad[np.newaxis, np.newaxis, :, :].astype(np.float32)/255

        h_offset = pad_h1
        w_offset = pad_w1

        return image_pad, alpha_pad, h_offset, w_offset

    def _forward(self, batch_bgr, batch_alp):
        out = self.engine.inference(batch_bgr, batch_alp)
        hair  = np.clip(out[0][0, 0, :, :], 0, 1).astype(np.float32)
        skin  = np.clip(out[1][0, 0, :, :], 0, 1).astype(np.float32)
        cloth = np.clip(out[2][0, 0, :, :], 0, 1).astype(np.float32)
        return hair, skin, cloth

    def _post(self, hair, skin, cloth, h_offset, w_offset, h, w):
        hair  =  hair[h_offset:h + h_offset, w_offset:w + w_offset]
        skin  =  skin[h_offset:h + h_offset, w_offset:w + w_offset]
        cloth = cloth[h_offset:h + h_offset, w_offset:w + w_offset]
        hair  = np.clip(np.round(hair * 255), 0, 255).astype(np.uint8)
        skin  = np.clip(np.round(skin * 255), 0, 255).astype(np.uint8)
        cloth = np.clip(np.round(cloth * 255), 0, 255).astype(np.uint8)
        return hair, skin, cloth

    def _estimateAlpha(self, bgr, alpha):
        if isinstance(alpha, np.ndarray):
            if bgr.shape[:2] == alpha.shape and alpha.dtype == np.uint8:
                return alpha
        human_matting = XManager.getModules('human_matting_v2')
        alpha = human_matting(bgr, targets='alpha')
        return alpha

    def inference(self, bgr, alpha):
        self._assert(bgr)
        h, w = bgr.shape[0:2]
        batch_bgr, batch_alp, h_offset, w_offset = self._format(bgr, alpha)
        hair, skin, cloth = self._forward(batch_bgr, batch_alp)
        return self._post(hair, skin, cloth, h_offset, w_offset, h, w)

    """
    """
    def _extractArgs(self, *args, **kwargs):
        if len(args) > 1:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        alpha = kwargs.pop('alpha', None)
        return targets, alpha

    def _returnResult(self, output, alpha, targets):
        def _formatResult(target):
            if target == 'source':
                return output
            if target == 'all':
                hair, skin, cloth = output
                return hair, skin, cloth, alpha
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, bgr, *args, **kwargs):
        targets, alpha = self._extractArgs(*args, **kwargs)
        alpha = self._estimateAlpha(bgr, alpha)
        output = self.inference(bgr, alpha)
        return self._returnResult(output, alpha, targets)
