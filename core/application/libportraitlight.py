
import logging
import cv2
import numpy as np
from ..base.cache import XPortrait
from ..base.extension import interp_cython
from .. import XManager


class LibPortraitLight:
    """
    """
    @staticmethod
    def getResources():
        return [
            LibPortraitLight.EngineConfig['parameters'],
        ]

    """
    """
    EngineConfig = {
        'type': 'torch',
        'device': 'cuda:0',
        'parameters': 'application/portrait_light.ts'
    }

    """
    """
    def __init__(self, *args, **kwargs):
        self.engine = XManager.createEngine(self.EngineConfig)
        self.max_size = 384
        self.tempRGB = [1., 1., 1.]
        self.eyedist_norm = 90
        self.left_eye = 36
        self.right_eye = 45
        self.edge_weight = 0.

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    """
    """
    def initialize(self, *args, **kwargs):
        self.engine.initialize(*args, **kwargs)

    def _assert(self, bgr):
        assert len(bgr.shape) == 3

    def _boxfilter(self, I, rad):
        kernel = 2 * rad + 1
        N = cv2.blur(I, (kernel, kernel))
        return N

    def _guideFilter(self, I, P, rads, eps):
        N = np.ones(np.shape(I))
        meanI = self._boxfilter(I.astype(np.float32), rads)
        meanP = self._boxfilter(P.astype(np.float32), rads)
        corrIP = self._boxfilter(I * P, rads)
        corrI = self._boxfilter(I * I, rads)
        varI = corrI - meanI * meanI
        covIP = corrIP - meanI * meanP
        a = covIP / (varI + eps)
        b = meanP - a * meanI
        meanA = self._boxfilter(a, rads) / N
        meanB = self._boxfilter(b, rads) / N
        res = meanA * I + meanB
        return res

    def _getPrior(self, bgr, alpha, landmark):
        xcache = XPortrait(bgr)
        if alpha is None:
            alpha = np.copy(xcache.alpha)
        if landmark is None:
            landmark = np.copy(xcache.landmark)
        if len(landmark.shape) == 3:
            landmark = landmark[0, :, :]
        return alpha, landmark

    def _cropForward(self, bgr, alpha, landmark, ratio=0.8):
        H, W, C = bgr.shape
        if ratio is not None:
            lft, rig = np.min(landmark[:, 0]), np.max(landmark[:, 0])
            top, bot = np.min(landmark[:, 1]), np.max(landmark[:, 1])
            w = rig - lft + 1
            h = bot - top + 1
            lft = int(round(max(lft - w * ratio, 0)))
            rig = int(round(min(rig + w * ratio, W-1)))
            top = int(round(max(top - h * ratio, 0)))
            bot = int(round(min(bot + h * ratio, H-1)))
            crop_bgr = bgr[top:bot+1, lft:rig+1, ...]
            crop_alpha = alpha[top:bot+1, lft:rig+1, ...]
            crop_landmark = np.copy(landmark)
            crop_landmark[:, 0] -= lft
            crop_landmark[:, 1] -= top
            return crop_bgr, crop_alpha, crop_landmark, (lft, rig, top, bot)
        return bgr, alpha, landmark, (0, W-1, 0, H-1)

    def _format(self, image, matte, lmks):
        height, width = np.shape(image)[0], np.shape(image)[1]
        offX = lmks[self.left_eye, 0] - lmks[self.right_eye, 0]
        offY = lmks[self.left_eye, 1] - lmks[self.right_eye, 1]
        eyedist = np.sqrt(offX**2 + offY**2)
        padOkH = int(height * self.eyedist_norm / eyedist + 0.5)
        padOkW = int(width * self.eyedist_norm / eyedist + 0.5)

        pStd_NoPad = np.zeros(shape=(padOkH, padOkW, 3), dtype=np.uint8)
        diff_re_resize = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        interp_cython.bicubic_byte(np.ascontiguousarray(image), width, height, 3, padOkW, padOkH, pStd_NoPad)
        interp_cython.bicubic_byte(np.ascontiguousarray(pStd_NoPad), padOkW, padOkH, 3, width, height, diff_re_resize)
        pStd_NoPad = pStd_NoPad.astype(np.float32) / 255.
        diff_re_resize = diff_re_resize.astype(np.float32) / 255.
        if matte is not None:
            mat_resize = cv2.resize(matte, (padOkW, padOkH), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255.
            pStd_NoPad = pStd_NoPad * mat_resize[:,:,np.newaxis] + \
                         (1 - mat_resize[:,:,np.newaxis]) * np.array(self.tempRGB).reshape(1, 1, 3)

        padH = int(max(np.ceil(padOkH / 32.) * 32, self.max_size))
        padW = int(max(np.ceil(padOkW / 32.) * 32, self.max_size))
        padleft = (padW - padOkW) // 2
        padtop = (padH - padOkH) // 2
        padright = padW - padOkW - padleft
        padbottom = padH - padOkH - padtop

        pStd_Pad = cv2.copyMakeBorder(pStd_NoPad, padtop, padbottom, padleft, padright,
            cv2.BORDER_CONSTANT, value=self.tempRGB)
        batch_input = pStd_Pad[:,:,::-1]
        batch_input = np.transpose(batch_input, (2, 0, 1))
        batch_input = batch_input[np.newaxis, ...] * 2 - 1

        image = image.astype(np.float32) / 255.
        diff =  image - diff_re_resize
        filter = self._guideFilter(image, image, 3, 0.01)
        edge = image - filter
        return batch_input, pStd_Pad, padleft, padtop, padOkH, padOkW, diff, edge

    def _forward(self, batch_bgr):
        batch_input = batch_bgr
        output = self.engine.inference(batch_input)
        output = (output + 1) / 2
        return output[0, ...].astype(np.float32)

    def _post(self, source, output, h1, w1, padleft, padtop, h, w, ratio):
        output = np.transpose(output, (1, 2, 0))
        output = output[:, :, ::-1]
        output = output * ratio + source.astype(np.float32) * (1 - ratio)
        output = output[padtop:padtop + h1, padleft:padleft + w1, :]
        out = np.zeros(shape=(h, w, 3), dtype=np.float32)
        interp_cython.bicubic_float(np.ascontiguousarray(
            output.copy()), output.shape[1], output.shape[0], 3, w, h, out)
        return np.round(out * 255).astype(np.uint8)

    def _cropBackward(self, source_bgr, source_alpha, output, crop_padding, paste_pack):
        lft, rig, top, bot = crop_padding
        alpha_mask = np.zeros_like(source_alpha)
        alpha_mask[top:bot + 1, lft:rig + 1] = source_alpha[top:bot + 1, lft:rig + 1]
        alpha_float = alpha_mask[:, :, np.newaxis].astype(np.float32) / 255
        canvas = np.zeros_like(source_bgr)
        canvas[top:bot + 1, lft:rig + 1, :] = output
        output_float = canvas.astype(np.float32)
        source_float = source_bgr.astype(np.float32) if paste_pack is True \
            else np.ones_like(source_bgr) * 255
        fusion = alpha_float * output_float + (1 - alpha_float) * source_float
        return np.round(fusion).astype(np.uint8)

    def inference(self, bgr, alpha, landmark, crop_ratio, paste_pack, fusion_ratio):
        alpha, landmark = self._getPrior(bgr, alpha, landmark)
        crop_bgr, crop_alpha, crop_landmark, crop_padding = self._cropForward(bgr, alpha, landmark, crop_ratio)
        batch_input, format_source, pad_lft, pad_top, h1, w1, diff, edge = \
            self._format(crop_bgr, crop_alpha, crop_landmark)
        output = self._forward(batch_input)
        output = self._post(format_source, output, h1, w1, pad_lft, pad_top, *crop_bgr.shape[:2], fusion_ratio)
        # fusion = np.round((output + diff + self.edge_weight * edge) * 255).astype(np.uint8)
        # output = np.round(output * 255).astype(np.uint8)
        output = self._cropBackward(bgr, alpha, output, crop_padding, paste_pack)
        return output

    """
    """
    def _extractArgs(self, *args, **kwargs):
        if len(args) > 1:
            logging.warning('{} useless parameters in {}'.format(
                len(args)-1, self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        inference_kwargs = dict()
        inference_kwargs['crop_ratio'] = crop_ratio = kwargs.pop('crop_ratio', None)
        inference_kwargs['paste_pack'] = bool(kwargs.pop('paste_pack', False))
        inference_kwargs['fusion_ratio'] = float(kwargs.pop('fusion_ratio', 1))
        assert crop_ratio is None or (crop_ratio > 0), crop_ratio
        bgr, xcache = (args[0], args[0]) if len(args) == 1 \
            else (kwargs.pop('bgr', None), kwargs.pop('xcache', None))
        if xcache is not None and isinstance(xcache, XPortrait):
            inference_kwargs['bgr'] = xcache.bgr
            inference_kwargs['alpha'] = xcache.alpha
            inference_kwargs['landmark'] = xcache.landmark
        if bgr is not None and isinstance(bgr, np.ndarray):
            inference_kwargs['bgr'] = bgr
            inference_kwargs['alpha'] = kwargs.pop('alpha', None)
            inference_kwargs['landmark'] = kwargs.pop('landmark', None)
        assert 'bgr' in inference_kwargs, (len(args), type(bgr), type(xcache))
        return targets, inference_kwargs

    def _returnResult(self, output, targets):
        def _formatResult(target):
            if target == 'source': return output
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
