
import logging
import numpy as np
import cv2
from .. import XManager



class LibFaceRestorationGFPGAN:
    """
    """
    @staticmethod
    def getEngineConfig(version):
        if version == '1.4':
            return {
                'type': 'torch',
                'device': 'cuda:0',
                'parameters': 'thirdparty/gfpgan.v1.4.gpu.ts'
            }

        if version == '1.5':
            return {
                'type': 'torch',
                'device': 'cuda:0',
                'parameters': 'thirdparty/gfpgan.v1.5.gpu.ts'
            }

        raise NotImplementedError('no such version: version={}'.format(version))

    @staticmethod
    def getResources():
        return [
            LibFaceRestorationGFPGAN.getEngineConfig(version='1.4')['parameters'],
        ]

    """
    """
    def __init__(self, version, *args, **kwargs):
        self.engine = XManager.createEngine(self.getEngineConfig(version))
        # base
        self.mean = np.array((0.5, 0.5, 0.5), dtype=np.float32)
        self.std = np.array((0.5, 0.5, 0.5), dtype=np.float32)
        self.face_size = (512, 512)
        # reference: template_5.png
        self.template = np.array(
            [[192.98138, 239.94708],
             [318.90277, 240.1936],
             [256.63416, 314.01935],
             [201.26117, 371.41043],
             [313.08905, 371.15118]], dtype=np.float32)

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        # specify gpu or cpu checkpoint version
        additional = dict()
        if 'device' in kwargs and kwargs['device'] == 'cpu':
            additional['parameters'] = '{}.cpu.pt'.format(self.EngineConfig['parameters'][:-7])
        self.engine.initialize(*args, **kwargs, **additional)

    """
    """
    def _resizeSource(self, bgr, upscale:float):
        h, w, c = bgr.shape
        upscale = max(upscale, 1)
        dst_h, dst_w = int(round(h * upscale)), int(round(w * upscale))
        resized = cv2.resize(bgr, (dst_w, dst_h))
        return resized

    def _getAlignPoints(self, landmarks):
        point1 = np.mean(landmarks[36:42, :], axis=0)
        point2 = np.mean(landmarks[42:48, :], axis=0)
        point3 = landmarks[33, :]
        point4 = landmarks[48, :]
        point5 = landmarks[54, :]
        points = np.stack([point1, point2, point3, point4, point5], axis=0)
        return points.astype(np.float32)

    def _warpFace(self, image, landmarks):
        assert len(landmarks) == 68
        points = self._getAlignPoints(landmarks)
        face_template = self.template * (np.array(self.face_size) / 512.0)
        affine_matrix = cv2.estimateAffinePartial2D(points, face_template, method=cv2.LMEDS)[0]
        # warp and crop faces
        warped = cv2.warpAffine(image, affine_matrix, tuple(self.face_size),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
        return warped, affine_matrix

    def _forward(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = (rgb.astype(np.float32) / 255. - self.mean) / self.std
        batch_array_rgb = np.expand_dims(np.transpose(rgb, (2, 0, 1)), axis=0)
        # batch_tensor_rgb = torch.from_numpy(batch_array_rgb).cuda()
        batch_output_rgb = self.engine.inference(batch_array_rgb)
        output = np.transpose(batch_output_rgb[0, ...], (1, 2, 0))
        output = np.clip(output, -1, 1)
        return ((output + 1) / 2 * 255).astype(np.uint8)[:, :, ::-1]

    def _convertParsing(self, parsing):
        mask = np.zeros(parsing.shape)
        index_colormap = [
            0, 255, 255, 255, 255,
            255, 255, 255, 255, 255,
            255, 255, 255, 0, 0,
            255, 0, 0, 0
        ]
        for n, color in enumerate(index_colormap):
            mask[parsing == n] = color
        return mask

    def _pasteBack(self, image, face, background, matrix, upscale_factor:float, use_parsing=False):
        # get inverse
        inverse = cv2.invertAffineTransform(matrix)
        inverse *= upscale_factor
        #
        h, w, _ = image.shape
        h_up, w_up = int(h * upscale_factor), int(w * upscale_factor)
        upsample_background = cv2.resize(background, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)

        extra_offset = 0.5 * upscale_factor if upscale_factor > 1 else 0
        inverse[:, 2] += extra_offset
        inv_restored = cv2.warpAffine(face, inverse, (w_up, h_up))

        if use_parsing == True:
            face_parsing = XManager.getModules('portrait_parsing')
            parsing = face_parsing(inv_restored, targets='source')
            # blur the mask
            mask = self._convertParsing(parsing)
            # cv2.imwrite(R'E:\deploy\algorithm-client\data\face_restoration\mask.png', mask)
            mask = cv2.GaussianBlur(mask, (101, 101), 11)
            mask = cv2.GaussianBlur(mask, (101, 101), 11)
            # remove the black borders
            threshold = 10
            mask[:threshold, :] = 0
            mask[-threshold:, :] = 0
            mask[:, :threshold] = 0
            mask[:, -threshold:] = 0
            mask = mask / 255.
            mask = cv2.resize(mask, inv_restored.shape[:2])
            mask = cv2.warpAffine(mask, inverse, (w_up, h_up), flags=3)
            inv_soft_mask = mask[:, :, None]
            pasted_face = inv_restored
        else:
            mask = np.ones(self.face_size, dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse, (w_up, h_up))
            # remove the black borders
            inv_mask_erosion = cv2.erode(
                inv_mask, np.ones((int(2 * upscale_factor), int(2 * upscale_factor)), np.uint8))
            pasted_face = inv_mask_erosion[:, :, None] * inv_restored
            total_face_area = np.sum(inv_mask_erosion)  # // 3
            # compute the fusion edge based on the area of face
            w_edge = int(total_face_area ** 0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
            inv_soft_mask = inv_soft_mask[:, :, None]

        upsample_background = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_background
        return upsample_background.astype(np.uint8)

    def _enhenceBackground(self, bgr, with_background):
        final = np.copy(bgr)
        if with_background == True:
            enhancer = XManager.getModules('practical_restoration_v1')
            final = enhancer(bgr, targets='source')
        return final

    """
    """
    def inferenceSingleFace(self, bgr, points):
        warped, matrix = self._warpFace(bgr, points)
        face_restoration = self._forward(warped)
        return face_restoration, matrix

    def inferenceMultiFace(self, bgr, points_list:list, enhence_background:bool):
        # restore faces
        data = []
        for n in range(len(points_list)):
            face, matrix = self.inferenceSingleFace(bgr, points_list[n])
            data.append((face, matrix))
        # for background
        final = self._enhenceBackground(bgr, enhence_background)
        # paste back
        for face, matrix in data:
            final = self._pasteBack(
                bgr, face, final, matrix, upscale_factor=1, use_parsing=False)
        return final, [face for face, matrix in data]

    def _extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        paste_back = kwargs.pop('paste_back', True)
        upscale = kwargs.pop('upscale', 1.)
        background = kwargs.pop('background', False)
        return targets, paste_back, upscale, background

    def _returnResult(self, output, targets):
        def _formatResult(target):
            if target == 'source':
                return output[0]
            if target == 'faces':
                return output[1]
            if target == 'gradio':
                final = cv2.cvtColor(output[0], cv2.COLOR_BGR2RGB)
                faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in output[1]]
                return final, *faces
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, bgr, *args, **kwargs):
        targets, paste_back, upscale, background = self._extractArgs(*args, **kwargs)
        face_landmark = XManager.getModules('face_landmark')
        bgr_rsz = self._resizeSource(bgr, upscale)
        points_list = face_landmark(bgr_rsz, targets='source')
        output = self.inferenceMultiFace(bgr_rsz, points_list, background)
        return self._returnResult(output, targets)


class LibFaceRestorationV14(LibFaceRestorationGFPGAN):
    def __init__(self, *args, **kwargs):
        super(LibFaceRestorationV14, self).__init__(version='1.4')


class LibFaceRestorationV15(LibFaceRestorationGFPGAN):
    def __init__(self, *args, **kwargs):
        super(LibFaceRestorationV15, self).__init__(version='1.5')