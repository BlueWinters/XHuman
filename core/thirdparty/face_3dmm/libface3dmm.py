
import logging
import cv2
import os
import numpy as np
import scipy
import ncnn
import torch
from ... import XManager


class LibFace3DMM:
    """
    """
    EngineConfig = {
        'type': 'torch',
        'device': 'cuda:0',
        'parameters': '3d/face_reconstruction.ts',
        'sim_mat': '3d/similarity_landmark3d_all.mat',
    }

    """
    """
    def __init__(self, *args, **kwargs):
        self.engine = XManager.createEngine(self.EngineConfig)
        self.net = None
        self.device = XManager.CommonDevice
        self.size = 224
        self.lm3d_std = None
        self.to_numpy = lambda tensor: tensor.detach().cpu().numpy()
        self.to_image = lambda array: np.clip(
            np.transpose(array[0, ...], axes=(1, 2, 0)), 0, 255).astype(np.uint8)

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def _getPath(self, root, name):
        assert name in self.EngineConfig
        return '{}/{}'.format(root, self.EngineConfig[name]) \
            if len(root) > 0 else self.EngineConfig[name]

    def initialize(self, *args, **kwargs):
        self.engine.initialize(*args, **kwargs)
        # if self.net is None:
        #     self.net = ncnn.Net()
        #     self.net.opt.use_vulkan_compute = True  # ÆôÓÃ Vulkan GPU ¼ÆËã
        #     self.net.opt.use_fp16_packed = True
        #     self.net.opt.use_fp16_storage = True
        #     self.net.opt.use_fp16_arithmetic = True
        #     self.net.set_vulkan_device(0)
        #     self.net.load_param("X:/checkpoints/deep-3d-reconstruction/pnnx/face_reconstruction.ncnn.param")
        #     self.net.load_model("X:/checkpoints/deep-3d-reconstruction/pnnx/face_reconstruction.ncnn.bin")
        root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
        # landmark-3d std
        sim_mat = '{}/{}'.format(root, self.EngineConfig['sim_mat']) \
            if len(root) > 0 else self.EngineConfig['sim_mat']
        self.lm3d_std = self._loadLandmark3d(sim_mat)

    def _loadLandmark3d(self, sim_mat):
        assert os.path.exists(sim_mat), sim_mat
        landmark3d = scipy.io.loadmat(sim_mat)['lm']
        return self._calculate5Points(landmark3d)

    """
    """
    @staticmethod
    def _calculate5Points(landmark):
        # calculate 5 facial landmarks using 68 landmarks
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        landmark = np.stack([
            landmark[lm_idx[0], :],
            np.mean(landmark[lm_idx[[1, 2]], :], 0),
            np.mean(landmark[lm_idx[[3, 4]], :], 0),
            landmark[lm_idx[5], :],
            landmark[lm_idx[6], :]], axis=0)
        landmark = landmark[[1, 2, 0, 3, 4], :]
        return landmark

    @staticmethod
    def _calculateParameters(xp, x):
        npts = xp.shape[1]
        A = np.zeros([2 * npts, 8])
        A[0:2 * npts - 1:2, 0:3] = x.transpose()
        A[0:2 * npts - 1:2, 3] = 1
        A[1:2 * npts:2, 4:7] = x.transpose()
        A[1:2 * npts:2, 7] = 1
        b = np.reshape(xp.transpose(), [2 * npts, 1])
        k, _, _, _ = np.linalg.lstsq(A, b)
        R1 = k[0:3]
        R2 = k[4:7]
        sTx = k[3]
        sTy = k[7]
        s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
        t = np.stack([sTx, sTy], axis=0)
        return t, s

    @staticmethod
    def _formatAndCrop(rgb, t, s, target_size=224.):
        h0, w0, _ = rgb.shape
        w = int(w0 * s)
        h = int(h0 * s)
        # resize
        img = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_CUBIC)
        # crop
        lft = int(w / 2 - target_size / 2 + float((t[0] - w0 / 2) * s))
        rig = lft + int(target_size)
        top = int(h / 2 - target_size / 2 + float((h0 / 2 - t[1]) * s))
        bot = top + int(target_size)
        # crop
        img_cropped = img[top:bot, lft:rig]
        return img_cropped, (h, w, lft, top, rig, bot)

    def _alignFace(self, bgr, points, rescale_factor=102.):
        assert len(points) == 5
        h0, w0, _ = bgr.shape
        t, s = self._calculateParameters(
            np.transpose(points, (1, 0)), np.transpose(self.lm3d_std, (1, 0)))
        # processing the image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_new, box = self._formatAndCrop(rgb, t, rescale_factor / s, target_size=self.size)
        return rgb_new, box

    def _formatInput(self, bgr, align_method='SDK'):
        h, w, c = bgr.shape
        if align_method == 'SDK':
            module = XManager.getModules('face_landmark')
            landmark = module(bgr)[0]
            landmark_copy = np.copy(landmark)
            landmark[:, -1] = h - 1 - landmark[:, -1]
            rgb, box = self._alignFace(bgr, self._calculate5Points(landmark))
            batch_rgb = np.transpose(rgb.astype(np.float32) / 255., axes=(2, 0, 1))[None, :, :, :]
            return batch_rgb, box, landmark_copy
        else:
            module = XManager.getModules('insightface')
            landmark = module(bgr, tasks='detection')[0]['kps']
            landmark_copy = np.copy(landmark)
            landmark[:, -1] = h - 1 - landmark[:, -1]
            rgb, box = self._alignFace(bgr, landmark)
            batch_rgb = np.transpose(rgb.astype(np.float32) / 255., axes=(2, 0, 1))[None, :, :, :]
            return batch_rgb, box, landmark_copy

    @staticmethod
    def splitParameters(coefficient):
        # size: 257
        id_coeffs = coefficient[:, :80]
        exp_coeffs = coefficient[:, 80: 144]
        tex_coeffs = coefficient[:, 144: 224]
        angles = coefficient[:, 224: 227]
        gammas = coefficient[:, 227: 254]
        translations = coefficient[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }

    """
    """
    def inference(self, bgr):
        inputs, crop_box, landmark = self._formatInput(bgr)
        output = self.engine.inference(inputs, detach=False)
        return output, (inputs, crop_box, landmark)
        # with self.net.create_extractor() as extractor:
        #     input_ncnn = np.ascontiguousarray(inputs[0, :, :, :], dtype=np.float32)
        #     extractor.input("in0", ncnn.Mat(input_ncnn))
        #     output_ncnn = extractor.extract("out0")[1].numpy().astype(np.float32)
        #     output_torch = torch.from_numpy(output_ncnn).float().unsqueeze(0).cuda(self.device)
        # return output_torch, (inputs, crop_box, landmark)

    """
    """
    def _extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        inference_kwargs = dict()
        return targets, inference_kwargs

    def _returnResult(self, format_info, output, targets):
        def _formatResult(target):
            # source output(coefficient)
            if target == 'source' or target == 'source_tensor':
                return output
            if target == 'source_numpy':
                return self.to_numpy(output)[0, :]
            if target == 'source_dict':
                data_dict = self.splitParameters(output)
                return {name: self.to_numpy(feat)[0, :].astype(np.float32) for name, feat in data_dict.items()}
            if target == 'format_info':
                return format_info  # format_input, crop_box, landmark
            raise Exception('no such return type "{}"'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, bgr, *args, **kwargs):
        targets, inference_kwargs = self._extractArgs(*args, **kwargs)
        output, format_input = self.inference(bgr)
        return self._returnResult(format_input, output, targets)
