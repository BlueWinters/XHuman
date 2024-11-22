
import logging
import cv2
import numpy as np
from .utils import finer
from .libhumanmatting_base import LibHumanMatting_Wrapper
from ... import XManager



class LibHumanMatting_DevKit(LibHumanMatting_Wrapper):
    """
    """
    @staticmethod
    def getResources():
        return [
            LibHumanMatting_DevKit.EngineConfig['parameters'],
        ]

    EngineConfig = {
        'type': 'torch',
        'device': 'cuda:0',
        'parameters': 'base/human_matting_devkit.ts'
    }

    """
    """
    def __init__(self, *args, **kwargs):
        super(LibHumanMatting_DevKit, self).__init__(*args, **kwargs)
        self.engine = XManager.createEngine(self.EngineConfig)

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    """
    """
    def initialize(self, *args, **kwargs):
        self.engine.initialize(*args, **kwargs)

    def _format(self, image, size, b, g, r):
        image_big = image.copy()
        height, width = np.shape(image)[0], np.shape(image)[1]
        ratio = height / width
        if height > width:
            h2 = size
            w2 = int(h2 / ratio)
            image = cv2.resize(image, (w2, h2), interpolation=cv2.INTER_LINEAR)
            image_sml = image.copy()
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
            image_sml = image.copy()
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

        image = np.transpose(image, (2, 0, 1)).astype(np.float64)
        image = image[np.newaxis, ...] / 255
        image_sml = image_sml.astype(np.float64) / 255
        image_big = image_big.astype(np.float64) / 255
        return image, image_sml, image_big, h_offset, w_offset, h2, w2

    def _forward(self, batch_bgr):
        batch_input = batch_bgr
        out = self.engine.inference(batch_input)
        alpha = np.clip(out['alpha'], 0, 1)
        alpha = alpha[0, ...].astype(np.float32)
        return alpha

    def _post(self, alpha, bgr_sml, bgr_big, h2, w2, h_offset, w_offset):
        alpha_sml = alpha[0, h_offset:h2 + h_offset, w_offset:w2 + w_offset]
        alpha_big = finer(bgr_sml, alpha_sml, bgr_big)
        alpha_big = np.clip(np.round(alpha_big * 255), 0, 255).astype(np.uint8)
        return alpha_big

    def inference(self, source):
        batch_bgr, bgr_sml, bgr_big, h_offset, w_offset, h2, w2 = \
            self._format(source, 512, 255, 255, 255)
        alpha = self._forward(batch_bgr)
        return self._post(alpha, bgr_sml, bgr_big, h2, w2, h_offset, w_offset)
