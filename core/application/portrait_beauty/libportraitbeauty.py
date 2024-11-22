
import logging
import os
import cv2
import numpy as np
from .face_utils import warp_and_crop_face, smooth_mask, BoundingBox
from .face_flaw import LibFaceflaw
from .operator_tools import alpha_multiply, elementwise_multiply
from ...base.cache import XPortrait
from ...base.cache import XPortraitExceptionAssert
from ... import XManager



class LibPortraitBeauty:
    """
    """
    @staticmethod
    def getResources():
        return [
            LibPortraitBeauty.EngineConfig['single']['parameters'],
            LibPortraitBeauty.EngineConfig['extend']['parameters'],
            LibPortraitBeauty.EngineConfig['hires']['parameters'],
        ]

    """
    """
    EngineConfig = {
        'single': {
            'type': 'torch',
            'device': 'cuda:0',
            'parameters': 'application/portrait_beauty_512X512_w_o_finetune_0.3mk+0.7mk.ts',
        },
        'extend': {
            'type': 'torch',
            'device': 'cuda:0',
            'parameters': 'application/portrait_beauty_512X512_w_finetune_multiscale_0.3mk+0.7mk.ts',
        },
        'hires': {
            'type': 'torch',
            'device': 'cuda:0',
            'parameters': 'application/portrait_beauty_face_enhancement_hires.ts',  # face_enhancement_hires.pt
        }
    }

    """
    """
    def __init__(self, *args, **kwargs):
        self.engine = XManager.createEngine(self.EngineConfig['single'])
        self.engine_extend = XManager.createEngine(self.EngineConfig['extend'])
        self.engine_hires = XManager.createEngine(self.EngineConfig['hires'])
        self.face_flaw = LibFaceflaw()
        self.max_size = 512

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        self.engine.initialize(*args, **kwargs)
        self.engine_extend.initialize(*args, **kwargs)
        self.engine_hires.initialize(*args, **kwargs)
        self.face_flaw.initialize(*args, **kwargs)

    """
    """
    def _getPrior(self, bgr, landmarks, parsing, matting, pre_light):
        if pre_light is not None:
            bgr = XManager.getModules('portrait_light')(
                bgr, landmark=landmarks, alpha=matting, crop_ratio=pre_light)
            matting = parsing = None  # reset to None
        if isinstance(landmarks, np.ndarray) is False:
            landmarks = XManager.getModules('face_landmark')(bgr)
        if isinstance(parsing, np.ndarray) is False:
            parsing = XManager.getModules('face_parsing')(bgr)
        if isinstance(matting, np.ndarray) is False:
            matting = XManager.getModules('human_matting')(bgr, targets='alpha')
        if len(landmarks.shape) == 2:
            landmarks = landmarks[None, :, :]
        return bgr, landmarks, parsing, matting

    def _assert(self, bgr):
        assert len(bgr.shape) == 3, 'bgr shape is {}'.format(str(bgr.shape))

    def _get_largest_connected_component(self, mask):
        # 连通组件标记
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        # 找到最大连通域的索引（不包括背景）
        index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        # 创建一个图像，其中只包含最大连通域
        largest_connected_component = np.zeros_like(mask)
        largest_connected_component[labels == index] = 255
        return largest_connected_component

    def _format(self, image, mask, size, b, g, r):
        hair_mask = ((mask <= 14) & (mask > 0)).astype(np.uint8)
        hair_mask = self._get_largest_connected_component(hair_mask)
        hair_mask += (mask == 19).astype(np.uint8)

        height, width = np.shape(image)[0], np.shape(image)[1]
        x, y, w, h = cv2.boundingRect(hair_mask)
        expand_pad = size // 5
        if w > h:
            w = min(w + expand_pad, width)
            x = max(0, x - expand_pad // 2)
            y = max((y - (w - h) // 2), 0)
            x1 = min(x + w, width)
            y1 = min(y + w, height)
        else:
            h = min(h + expand_pad, height)
            y = max(0, y - expand_pad // 2)
            x = max((x - (h - w) // 2), 0)
            x1 = min(x + h, width)
            y1 = min(y + h, height)
        image = image[y:y1, x:x1]

        height, width = np.shape(image)[0], np.shape(image)[1]
        ratio = height / width
        if height > width:
            flags = cv2.INTER_AREA if height > size else cv2.INTER_LANCZOS4
            h2 = size
            w2 = int(h2 / ratio)
            image = cv2.resize(image, (w2, h2), interpolation=flags)
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
            flags = cv2.INTER_AREA if width > size else cv2.INTER_LANCZOS4
            w2 = size
            h2 = int(w2 * ratio)
            image = cv2.resize(image, (w2, h2), interpolation=flags)
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
        return image, BoundingBox(w_offset, h_offset, w2, h2), BoundingBox(x, y, x1 - x, y1 - y)

    def _forward(self, batch_bgr, mode='base'):
        batch_input = cv2.cvtColor(batch_bgr, cv2.COLOR_BGR2RGB)
        batch_input = np.transpose(batch_input, (2, 0, 1)).astype(np.float32)
        batch_input = batch_input[np.newaxis, ...] / 255.
        batch_input = (batch_input - 0.5) / 0.5
        if mode == 'base':
            output = self.engine.inference(batch_input)[0, ...].transpose(1, 2, 0)
        elif mode == 'extend':
            output = self.engine_extend.inference(batch_input)[0, ...].transpose(1, 2, 0)
        elif mode == 'hires':
            output = self.engine_hires.inference(batch_input)[0, ...].transpose(1, 2, 0)
        else:
            raise ValueError(f"The mode of {mode} is not supported! ")
        output = np.clip((output * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    def _post(self, bgr, coarse_result, box_pad, box_crop):
        output = np.zeros_like(bgr)
        coarse_result = coarse_result[box_pad.y:box_pad.y + box_pad.h, box_pad.x:box_pad.x + box_pad.w, :]
        flags = cv2.INTER_AREA if max(box_pad.w, box_pad.h) > max(box_crop.w, box_crop.h) else cv2.INTER_LANCZOS4
        coarse_result = cv2.resize(coarse_result, (box_crop.w, box_crop.h), interpolation=flags)
        output[box_crop.y:box_crop.y + box_crop.h, box_crop.x:box_crop.x + box_crop.w] = coarse_result
        return output

    def _upsampleBackground(self, bgr, mask, matting):
        image = bgr.copy()
        height, width, _ = image.shape
        face_mask = ((mask <= 11) & (mask > 0) | (mask == 14)).astype(np.uint8)
        face_mask = self._get_largest_connected_component(face_mask)

        x, y, w, h = cv2.boundingRect(face_mask)
        center_y = y + h // 2
        std_L = min(w, h)

        bottom = center_y + int(2.015 * std_L)
        if bottom >= height - 1:
            top = max(center_y - int(1.074 * std_L), 0)
            x1, y1, w1, h1 = cv2.boundingRect(matting)
            left = max(x1 - std_L // 10, 0)
            right = max(x1 + w1 + std_L // 10, width)
            bottom = min(bottom, height)
            roi = image[top:bottom, left:right]
            # roi, _ = self.upsampler.enhance(roi, outscale=1)
            # roi = XManager.getModules('face_restoration_v1')(roi, upscale=1)
            image[top:bottom, left:right] = roi
        return image

    def _inpaintingFlaw(self, bgr, landmark, parsing, flaw_index):
        if flaw_index is not None:
            organ_mask = ((parsing >= 2) & (parsing <= 9)).astype(np.float32)
            mask = self.face_flaw.inference(bgr, landmark)
            if isinstance(flaw_index, int):
                flaw_mask = (mask == flaw_index).astype(np.uint8) * 255  # 1 --> 斑痘  2 -- > 痣
            else:
                flaw_mask = (mask > 0).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            flaw_mask = cv2.dilate(flaw_mask, kernel)
            organ_mask = cv2.dilate(organ_mask, kernel)
            flaw_mask = np.clip(flaw_mask.astype(np.float32) * (1 - organ_mask), 0, 255).astype(np.uint8)
            out = cv2.inpaint(bgr, flaw_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
            return out
        else:
            return bgr

    def _merge(self, bgr, fine_result, coarse_result, mask, matte, combine_mask, parallel):
        exclude_mask = ((mask == 0) | (mask >= 15) & (mask != 19)).astype(np.float32)
        hair_mask = (mask == 13).astype(np.float32) + ((mask == 4) | (mask == 5)).astype(np.float32) # 头发和眼睛
        skin_mask = ((mask == 12) | (mask == 19)).astype(np.float32)
        exclude_mask = smooth_mask(exclude_mask, 0, 4)
        hair_mask = smooth_mask(hair_mask, 0, 4)
        skin_mask = smooth_mask(skin_mask, 0, 10)
        combine_mask = smooth_mask(combine_mask, 0, 50)
        matte = matte[:,:,None].astype(np.float32) / 255.
        out = elementwise_multiply(fine_result, hair_mask, combine_mask, coarse_result, parallel)
        out = alpha_multiply(out, skin_mask, coarse_result, parallel) # 替换脖子区域
        out = alpha_multiply(out, exclude_mask, bgr, parallel)
        out = alpha_multiply(bgr, matte, out, parallel)  # 保持背景不变
        out = np.clip(out, 0, 255).astype(np.uint8) # 是否和原图融合
        return out

    def _restore(self, bgr, landmarks, parallel, mode='base'):
        # the mask for pasting restored faces back
        mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        self.max_size = 1024 if mode == 'hires' else 512

        height, width = bgr.shape[:2]
        combine_mask = np.zeros(bgr.shape, dtype=np.float32)
        for i, landmark in enumerate(landmarks):
            point1 = np.mean(landmark[36:42, :], axis=0)
            point2 = np.mean(landmark[42:48, :], axis=0)
            point3 = landmark[33, :]
            point4 = landmark[48, :]  # 48
            point5 = landmark[54, :]  # 54
            facial5points = np.stack([point1, point2, point3, point4, point5], axis=0)
            facial5points = np.transpose(facial5points, (1, 0))
            of, tfm_inv = warp_and_crop_face(bgr, facial5points, crop_size=(self.max_size, self.max_size))
            # enhance the face
            ef = self._forward(of, mode=mode)
            tmp_mask = mask
            tmp_mask = cv2.resize(tmp_mask, ef.shape[:2])
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)
            tmp_mask = np.clip(tmp_mask, 0, 1)
            out_img = alpha_multiply(bgr.astype(np.float32), tmp_mask[:,:,None], tmp_img, parallel)
            combine_mask += tmp_mask[:,:,None]
        out_img = np.clip(out_img, 0, 255).astype(np.uint8)
        return out_img, combine_mask

    def inference(self, bgr, landmarks, parsing, alpha, pre_light, flaw_index, parallel):
        bgr, landmarks, parsing, alpha = self._getPrior(bgr, landmarks, parsing, alpha, pre_light)
        fine_result, combine_mask = self._restore(bgr, landmarks, parallel, mode='base')
        batch_bgr, box_pad, box_crop = self._format(bgr, parsing, self.max_size, 255, 255, 255)
        coarse_result = self._forward(batch_bgr, mode='extend')
        coarse_result = self._post(bgr, coarse_result, box_pad, box_crop)
        fine_result, _ = self._restore(fine_result, landmarks, parallel, mode='hires')
        fine_result = self._inpaintingFlaw(fine_result, landmarks, parsing, flaw_index)
        # bgr = self._upsampleBackground(bgr, mask, matting)
        output = self._merge(bgr, fine_result, coarse_result, parsing, alpha, combine_mask, parallel)
        return output

    """
    """
    def _extractArgs(self, *args, **kwargs):
        if len(args) > 1:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        inference_kwargs = dict()
        bgr, xcache = (args[0], args[0]) if len(args) == 1 \
            else (kwargs.pop('bgr', None), kwargs.pop('xcache', None))
        if xcache is not None and isinstance(xcache, XPortrait):
            inference_kwargs['bgr'] = xcache.bgr
            inference_kwargs['landmarks'] = xcache.landmark
            inference_kwargs['parsing'] = xcache.parsing
            inference_kwargs['alpha'] = xcache.alpha
        if bgr is not None and isinstance(bgr, np.ndarray):
            inference_kwargs['bgr'] = bgr
            inference_kwargs['landmarks'] = kwargs.pop('landmarks', None)
            inference_kwargs['parsing'] = kwargs.pop('parsing', None)
            inference_kwargs['alpha'] = kwargs.pop('alpha', None)
        inference_kwargs['pre_light'] = pre_light = kwargs.pop('pre_light', None)
        inference_kwargs['flaw_index'] = kwargs.pop('flaw_index', [1, 2])  # 1 --> 只去除斑痘  2 -- > 只去除痣
        inference_kwargs['parallel'] = bool(kwargs.pop('parallel', True))
        assert 'bgr' in inference_kwargs, inference_kwargs.keys()
        assert pre_light is None or (pre_light > 0), pre_light
        XPortraitExceptionAssert.assertKwArgs('flaw_index', inference_kwargs['flaw_index'], [None, 1, 2, [1, 2]])
        return targets, inference_kwargs

    def _returnResult(self, output, targets):
        def _formatResult(target):
            if target == 'source':
                return output
            if target == 'gradio':
                return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, *args, **kwargs):
        targets, inference_kwargs = self._extractArgs(*args, **kwargs)
        output = self.inference(**inference_kwargs)
        return self._returnResult(output, targets)

