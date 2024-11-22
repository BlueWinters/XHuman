
import logging
import os
import cv2
import numpy as np
from ...base.extension import interp_cython
from ... import XManager



class LibFaceflaw:
    """
    """
    Config = {
        'type': 'torch',
        'device': 'cuda:0',
        'parameters': 'xface/advanced/portrait_beauty_faceflaw.pt',
    }

    def __init__(self, *args, **kwargs):
        self.engine = XManager.createEngine(self.Config)
        self.max_size = 1024

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        self.engine.initialize(*args, **kwargs)

    def _assert(self, bgr):
        assert len(bgr.shape) == 3

    def anti_aliasing_sampler(self, matrix, dst_h, dst_w):
        # isTensor = isinstance(matrix, torch.Tensor)
        # if isTensor:
        #     matrix = matrix.float().detach().cpu().numpy()[0].transpose(1, 2, 0)
        #     device, dtype = matrix.device, matrix.dtype
        matrix = np.ascontiguousarray((matrix))
        h, w, c = matrix.shape
        out = np.zeros(shape=(dst_h, dst_w, c), dtype=matrix.dtype)
        if matrix.dtype == 'float32':
            interp_cython.bicubic_float(matrix, w, h, c, dst_w, dst_h, out)
        else:
            interp_cython.bicubic_byte(matrix, w, h, c, dst_w, dst_h, out)
        # if isTensor:
        #     out = torch.from_numpy(out)[None,]
        #     out = out.movedim(-1, 1)
        #     out = out.to(device, dtype=dtype)
        return out

    def _format(self, bgr, landmarks):
        height, width, _ = bgr.shape
        x = np.min(landmarks[0][:, 0])
        y = np.min(landmarks[0][:, 1])
        x1 = np.max(landmarks[0][:, 0])
        y1 = np.max(landmarks[0][:, 1])
        w, h = x1 - x, y1 - y

        size = min(w, h)
        center_x, center_y = int(x + w // 2), int(y + h // 2)
        top = center_y - int(1.097 * size)
        bottom = center_y + int(0.565 * size)
        left = center_x - int(0.687 * size)
        right = center_x + int(0.687 * size)

        pad_t = abs(top) if top < 0 else 0
        pad_b = bottom - height + 1 if bottom > height - 1 else 0
        pad_l = abs(left) if left < 0 else 0
        pad_r = right - width + 1 if right > width - 1 else 0

        top = max(0, top)
        bottom = min(height - 1, bottom)
        left = max(0, left)
        right = min(width - 1, right)

        roi = bgr[top:bottom, left:right]
        roi = cv2.copyMakeBorder(roi, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        roi_h, roi_w = roi.shape[:2]
        if roi_h > roi_w and roi_h > self.max_size:
            ratio = roi_h / roi_w
            h2 = self.max_size
            w2 = int(h2 / ratio)
            roi = self.anti_aliasing_sampler(roi, h2, w2)
        elif roi_w > roi_h and roi_w > self.max_size:
            ratio = roi_w / roi_h
            w2 = self.max_size
            h2 = int(w2 / ratio)
            roi = self.anti_aliasing_sampler(roi, h2, w2)
        else:
            pass
        roi = np.transpose(roi, (2, 0, 1)).astype(np.float32)
        roi = roi[np.newaxis, ...] / 255
        return roi, roi_h, roi_w, [pad_t, pad_b, pad_l, pad_r], [top, bottom, left, right],

    def _forward(self, batch_bgr):
        batch_input = batch_bgr
        parsing_soft = self.engine.inference(batch_input)
        return parsing_soft[0, ...].astype(np.float32)

    def _post(self, parsing_soft, roi_h, roi_w, pad, coords, h, w):
        seg = np.zeros(shape=(h, w), dtype=np.uint8)
        parsing_soft = np.transpose(parsing_soft, (1, 2, 0))
        parsing_soft = cv2.resize(parsing_soft, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
        parse = np.argmax(parsing_soft, 2)
        parse = np.uint8(parse)
        parse = parse[pad[0]:parse.shape[0]-pad[1], pad[2]:parse.shape[1]-pad[3]]
        seg[coords[0]:coords[1], coords[2]:coords[3]] = parse
        return seg

    def inference(self, bgr, landmarks):
        self._assert(bgr)
        h, w = bgr.shape[0:2]
        batch_bgr, roi_h, roi_w, pad, coords = self._format(bgr, landmarks)
        parsing_soft = self._forward(batch_bgr)
        return self._post(parsing_soft, roi_h, roi_w, pad, coords, h, w)