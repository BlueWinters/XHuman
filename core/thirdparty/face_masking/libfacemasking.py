
import logging
import sys
import cv2
import numpy as np
import torch
from .bfm import ParametricFaceModel
from .mesh_render import MeshRenderer
from .mesh_render_numpy import MeshRendererNumpy
from ... import XManager


class LibFaceMasking:
    """
    """
    Parameters = {
        'bfm_mat': '3d/bfm_model_front.mat',
        # additional from v2
        'bfm_uvs2': '3d/template_mesh/bfm_uvs2.npy',
    }

    """
    """
    def __init__(self, *args, **kwargs):
        self.device = XManager.CommonDevice
        self.size = 224
        self.camera_d = 10.0
        self.focal = 1015.0
        self.center = 112.0
        self.fov = None
        self.z_near = 5.0
        self.z_far = 15.0
        self.param_face = None
        self.rasterize_size = None
        self.bfm_uv = None
        self.bfm_uv_np = None
        self.renderer = None
        self.renderer_np = None
        self.use_opengl = True if sys.platform == 'win32' else False
        self.to_numpy = lambda tensor: tensor.detach().cpu().numpy() \
            if isinstance(tensor, torch.Tensor) else tensor
        self.to_image = lambda array: np.clip(np.transpose(
            array[0, ...], axes=(1, 2, 0)), 0, 255).astype(np.uint8)

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def _getPath(self, root, name):
        assert name in self.Parameters
        return '{}/{}'.format(root, self.Parameters[name]) \
            if len(root) > 0 else self.Parameters[name]

    def initialize(self, *args, **kwargs):
        root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
        # bfm parameters
        self.param_face = ParametricFaceModel(
            bfm_mat=self._getPath(root, 'bfm_mat'),
            camera_distance=self.camera_d,
            focal=self.focal,
            center=self.center
        ).to(self.device)
        # bfm-uv: 35709,2
        self.bfm_uv_np = np.load(self._getPath(root, 'bfm_uvs2'))
        self.bfm_uv = torch.from_numpy(self.bfm_uv_np).to(self.device).float()
        # render
        self.fov = 2 * np.arctan(self.center / self.focal) * 180 / np.pi
        self.rasterize_size = int(2 * self.center)
        self.renderer = MeshRenderer(self.fov, self.z_near, self.z_far, self.rasterize_size).to(self.device)
        self.renderer_np = MeshRendererNumpy(self.fov, self.z_near, self.z_far, self.rasterize_size)

    """
    rendering
    """
    def _renderFaceFine(self, coefficient, texture):
        assert isinstance(coefficient, torch.Tensor)
        vertex, tex, color, normals, _ = self.param_face.compute_for_render(coefficient)
        # render with additional texture
        if texture is not None:
            if isinstance(texture, np.ndarray):
                # image, mask, depth = self.renderer_np.render_uv_texture(
                #     self.to_numpy(vertex), self.to_numpy(self.param_face.face_buf), self.bfm_uv_np, texture)
                image, mask, depth = self.renderer_np.render_uv_texture_cython(
                    self.to_numpy(vertex), self.to_numpy(self.param_face.face_buf), self.bfm_uv_np, texture)
                return image, mask, self._normalizeDepth(depth, mask), None
            else:
                mask_tensor, depth_tensor, face_tensor, uv_tensor = self.renderer.render_uv_texture(
                    vertex.detach(), self.param_face.face_buf, self.bfm_uv.clone(), texture, reverse_uv=True)
                face_np, mask_np, depth_np, uv_np = self._convertToImage(face_tensor, mask_tensor, depth_tensor, uv_tensor)
                return face_np, mask_np, depth_np, uv_np
        else:
            # render with source texture(mean texture from 3dmm)
            mask_th, depth_th, face_th = self.renderer.render_feature_texture(
                vertex, self.param_face.face_buf, feat=tex)
            face_np, mask_np, depth_np, _ = self._convertToImage(face_th, mask_th, depth_th, None)
            return face_np, mask_np, depth_np, None

    def _renderFaceShape(self, coefficient, with_np=False):
        assert isinstance(coefficient, torch.Tensor)
        vertex, tex, color, normals, _ = self.param_face.compute_for_render(coefficient)
        gray_shading = self.param_face.compute_for_render_shape(torch.ones_like(tex) * 0.78, normals)
        if with_np:
            mask_np, depth_np, face_np = self.renderer_np.render_feature_texture(
                self.to_numpy(vertex), self.to_numpy(self.param_face.face_buf), feat=self.to_numpy(gray_shading))
            depth_np = self._normalizeDepth(depth_np, mask_np)
            return face_np, mask_np, depth_np, None
        else:
            mask_th, depth_th, face_th = self.renderer.render_feature_texture(
                vertex, self.param_face.face_buf, feat=gray_shading)
            face_np, mask_np, depth_np, _ = self._convertToImage(face_th, mask_th, depth_th, None)
            return face_np, mask_np, depth_np, None

    @staticmethod
    def _normalizeDepth(depth:np.ndarray, mask:np.ndarray):
        d_min, d_max = np.min(depth[depth > 0]), np.max(depth[depth > 0])
        depth = np.clip((depth - d_min) / (d_max - d_min) * 255, 0., 255.).astype(np.uint8)
        depth = np.where(mask > 0, 255 - depth, 0)
        return depth

    def _convertToImage(self, face, mask, depth, uv, normalize=True):
        face_np = self.to_image(self.to_numpy(face * 255)) if face is not None else face
        mask_np = self.to_image(self.to_numpy(mask * 255))[:, :, 0] if mask is not None else mask
        depth_np = self.to_numpy(depth)[0, 0, :, :] if depth is not None else depth
        uv_np = self.to_image(self.to_numpy(uv * 255)) if uv is not None else uv  # B,2,H,W --> H,W,2
        if normalize is True and depth is not None:
            depth_np = self._normalizeDepth(depth_np, mask_np)
        return face_np, mask_np, depth_np, uv_np

    @staticmethod
    def _pasteBack(bgr, box, face, mask, depth):
        h, w, c = bgr.shape
        hh, ww, lft, top, rig, bot = box
        rh = float(hh) / h
        rw = float(ww) / w
        nh = int(round(face.shape[0] / rh))
        nw = int(round(face.shape[1] / rw))
        mask = cv2.erode(mask, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        face_new = cv2.resize(face, (nw, nh), interpolation=cv2.INTER_CUBIC)
        mask_new = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_LINEAR)
        depth_new = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # mask_new = cv2.erode(mask_new, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        lp = max(int(round(lft / rw)), 0)
        rp_src = w - nw - lp
        rp = max(rp_src, 0)
        tp = max(int(round(top / rh)), 0)
        bp_src = h - nh - tp
        bp = max(bp_src, 0)
        face_new = np.pad(face_new, ((tp, bp), (lp, rp), (0, 0)), constant_values=0)
        mask_new = np.pad(mask_new, ((tp, bp), (lp, rp)), constant_values=0)
        depth_new = np.pad(depth_new, ((tp, bp), (lp, rp)), constant_values=0)
        if rp_src < 0:
            mask_new, face_new = mask_new[:, :rp_src], face_new[:, :rp_src]
        if bp_src < 0:
            mask_new, face_new = mask_new[:bp_src, :], face_new[:bp_src, :]
        alpha = mask_new.astype(np.float32)[:, :, None] / 255.
        if face_new.shape[2] == 4:
            k = 7
            face_bgr = face_new[:, :, :3]
            face_alpha = face_new[:, :, 3:4]
            face_alpha = np.where(face_alpha == 255, face_alpha, 0).astype(np.uint8)
            alpha = cv2.GaussianBlur(face_alpha, (k, k), sigmaX=k // 2, sigmaY=k // 2)
            alpha = alpha.astype(np.float32)[:, :, None] / 255.
            composite = face_bgr * alpha + (1 - alpha) * bgr
        else:
            assert face_new.shape[2] == 3, face_new.shape
            composite = face_new * alpha + (1 - alpha) * bgr
        return np.round(composite).astype(np.uint8), face_new, mask_new, depth_new

    """
    """
    @staticmethod
    def estimateParameters(bgr):
        module = XManager.getModules('face_3dmm')
        output, format_info = module(bgr, targets=['source', 'format_info'])
        return output, format_info

    """
    """
    def _extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        texture = kwargs.pop('texture', None)
        inference_kwargs = dict(texture=texture)
        return targets, inference_kwargs

    def _returnResult(self, source_input, format_info, output, targets, texture):
        def _formatResult(target):
            format_input, crop_box, landmark = format_info
            # render source with additional texture
            if target == 'source':
                face, mask, depth, uv = self._renderFaceFine(output, texture)
                return self._pasteBack(source_input, crop_box, face, mask, depth)
            if target == 'shape':
                face, mask, depth, uv = self._renderFaceShape(output)
                return self._pasteBack(source_input, crop_box, face, mask, depth)
            raise Exception('no such return type "{}"'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, bgr, *args, **kwargs):
        targets, inference_kwargs = self._extractArgs(*args, **kwargs)
        output, format_info = self.estimateParameters(bgr)
        return self._returnResult(bgr, format_info, output, targets, **inference_kwargs)
