
import logging
import os
import cv2
import numpy as np
import random
import skimage
import pickle
import typing
import tqdm
# import numba
from ...geometry import Rectangle
from ...base import XPortrait, XPortraitHelper
from ...utils.context import XContextTimer
from ...utils.video import XVideoReader, XVideoWriter
from ... import XManager


class LibMasking_Blur:
    """
    """
    @staticmethod
    def benchmark():
        pass

    @staticmethod
    def prepare():
        bgr = cv2.imread('benchmark/asset/prepare/input.png')
        box = LibMasking_Blur.toCache(bgr).box
        logging.warning('prepare finish...')

    """
    """
    def __init__(self, *args, **kwargs):
        pass

    def __del__(self):
        # logging.warning('delete module {}'.format(self.__class__.__name__))
        pass

    def initialize(self, *args, **kwargs):
        pass

    """
    """
    @staticmethod
    def formatSizeWithPaddingForward(bgr, dst_h, dst_w, padding_value=255):
        # base on long side, padding to target size
        src_h, src_w, _ = bgr.shape
        src_ratio = float(src_h / src_w)
        dst_ratio = float(dst_h / dst_w)
        if src_ratio > dst_ratio:
            rsz_h, rsz_w = dst_h, int(round(float(src_w / src_h) * dst_h))
            resized = cv2.resize(bgr, (rsz_w, rsz_h))
            lp = (dst_w - rsz_w) // 2
            rp = dst_w - rsz_w - lp
            resized = np.pad(resized, ((0, 0), (lp, rp), (0, 0)), constant_values=padding_value, mode='constant')
            padding = (0, 0, lp, rp)
        else:
            rsz_h, rsz_w = int(round(float(src_h / src_w) * dst_w)), dst_w
            resized = cv2.resize(bgr, (rsz_w, rsz_h))
            tp = (dst_h - rsz_h) // 2
            bp = dst_h - rsz_h - tp
            resized = np.pad(resized, ((tp, bp), (0, 0), (0, 0)), constant_values=255, mode='constant')
            padding = (tp, bp, 0, 0)
        return resized, padding

    @staticmethod
    def formatSizeWithPaddingBackward(bgr_src, bgr_cur, padding):
        tp, bp, lp, rp = padding
        bp = bgr_cur.shape[0] - bp  # 0 --> h
        rp = bgr_cur.shape[1] - rp  # 0 --> w
        src_h, src_w, _ = bgr_src.shape
        resized = cv2.resize(bgr_cur[tp:bp, lp:rp, :], (src_w, src_h))
        return resized

    """
    """
    @staticmethod
    def doWithGaussianBlur(bgr, kernel=15, blur_size=256):
        format_bgr, padding = LibMasking_Blur.formatSizeWithPaddingForward(bgr, blur_size, blur_size)
        kernel = kernel if kernel % 2 == 1 else (kernel + 1)  # should be odd
        blured_bgr = cv2.GaussianBlur(format_bgr, (kernel, kernel), sigmaX=kernel // 2, sigmaY=kernel // 2)
        reformat_blur_bgr = LibMasking_Blur.formatSizeWithPaddingBackward(bgr, blured_bgr, padding)
        return reformat_blur_bgr

    @staticmethod
    def doWithBlurMotion(bgr, kernel=15, blur_size=256):
        format_bgr, padding = LibMasking_Blur.formatSizeWithPaddingForward(bgr, blur_size, blur_size)
        kernel = kernel if kernel % 2 == 1 else (kernel + 1)  # should be odd
        c = int(kernel / 2)
        x, y = 16, 16
        blur_kernel = np.zeros((kernel, kernel), dtype=np.uint8)
        blur_kernel = cv2.line(blur_kernel, (c + x, c + y), (c, c), (1,), 1)
        blur_kernel = blur_kernel / int(np.count_nonzero(blur_kernel))
        blured_bgr = cv2.filter2D(format_bgr, ddepth=-1, kernel=blur_kernel)
        reformat_blur_bgr = LibMasking_Blur.formatSizeWithPaddingBackward(bgr, blured_bgr, padding)
        return reformat_blur_bgr

    @staticmethod
    # @numba.jit(nopython=True, nogil=True, parallel=True)
    def maskingWithBlurWaterJit(bgr, bgr_copy, int_x, int_y, x_new, y_new):
        h, w, c = bgr.shape
        # for ii in numba.prange(h):
        #     for jj in numba.prange(w):
        for ii in range(h):
            for jj in range(w):
                new_xx = int_x[ii, jj]
                new_yy = int_y[ii, jj]
                if x_new[ii, jj] < 0 or x_new[ii, jj] > w - 1:
                    continue
                if y_new[ii, jj] < 0 or y_new[ii, jj] > h - 1:
                    continue
                bgr_copy[ii, jj, :] = bgr[new_yy, new_xx, :]
        return bgr_copy

    @staticmethod
    def getBlurWaterParameters(bgr, A=2.0, B=8.0):
        h, w, c = bgr.shape
        # A = 2.0  # rotation degree
        # B = 8.0  # each water length
        center_x = (w - 1) / 2.0
        center_y = (h - 1) / 2.0
        xx = np.arange(w)
        yy = np.arange(h)
        x_mask = np.repeat(xx[None, :], h, 0)
        y_mask = np.repeat(yy[:, None], w, 1)
        xx_dif = x_mask - center_x
        yy_dif = center_y - y_mask
        theta = np.arctan2(yy_dif, xx_dif)
        r = np.sqrt(xx_dif * xx_dif + yy_dif * yy_dif)
        r1 = r + A * w * 0.01 * np.sin(B * 0.1 * r)
        x_new = r1 * np.cos(theta) + center_x
        y_new = center_y - r1 * np.sin(theta)
        int_x = np.floor(x_new)
        int_x = int_x.astype(int)
        int_y = np.floor(y_new)
        int_y = int_y.astype(int)
        return int_x, int_y, x_new, y_new

    @staticmethod
    def doWithBlurWater(bgr, A=2.0, B=8.0, map_coordinates=None):
        if map_coordinates is None:
            map_coordinates = LibMasking_Blur.getBlurWaterParameters(bgr, A, B)
        assert len(map_coordinates) == 4, len(map_coordinates)
        int_x, int_y, x_new, y_new = map_coordinates
        h, w, c = bgr.shape
        bgr_copy = np.copy(bgr)
        for ii in range(h):
            for jj in range(w):
                new_xx = int_x[ii, jj]
                new_yy = int_y[ii, jj]
                if x_new[ii, jj] < 0 or x_new[ii, jj] > w - 1:
                    continue
                if y_new[ii, jj] < 0 or y_new[ii, jj] > h - 1:
                    continue
                bgr_copy[ii, jj, :] = bgr[new_yy, new_xx, :]
        # LibMasking_Blur.maskingWithBlurWaterJit(bgr, bgr_copy, int_x, int_y, x_new, y_new)
        return bgr_copy, map_coordinates

    @staticmethod
    def doWithBlurDiffusion(bgr, k_neigh=11, pre_blur_kernel=3, post_blur_kernel=3):
        if pre_blur_kernel > 0:
            blur_bgr = LibMasking_Blur.doWithGaussianBlur(bgr, kernel=pre_blur_kernel)
        else:
            blur_bgr = np.copy(bgr)
        result_bgr = blur_bgr.copy()
        h, w, c = bgr.shape
        for hj in range(k_neigh, h - k_neigh, 1):
            for wi in range(k_neigh, w - k_neigh, 1):
                jj = int((random.random() - 0.5) * (k_neigh * 2 - 1))
                ii = int((random.random() - 0.5) * (k_neigh * 2 - 1))
                hh = (hj + jj) % h
                ww = (wi + ii) % w
                result_bgr[hj, wi, :] = blur_bgr[hh, ww, :]
        if post_blur_kernel > 0:
            result_bgr = LibMasking_Blur.doWithGaussianBlur(result_bgr, kernel=post_blur_kernel)
        return result_bgr

    """
    """
    @staticmethod
    def getMaskFromBox(h, w, box, ratio, value=255):
        mask = np.ones(shape=(h, w), dtype=np.uint8)
        lft, top, rig, bot = box
        if ratio > 0:
            hh = bot - top
            ww = rig - lft
            lft = int(max(0, lft - ww * ratio))
            top = int(max(0, top - hh * ratio))
            rig = int(min(w, rig + ww * ratio))
            bot = int(min(h, bot + hh * ratio))
        mask[top:bot, lft:rig] = value
        return mask

    @staticmethod
    def getFaceMaskByPoints(bgr, box, n=0, top_line='brow', value=255):
        lft, top, rig, bot = box
        cache = XPortrait(bgr[top:bot, lft:rig, :])
        return XPortraitHelper.getFaceRegion(cache, index=n, top_line=top_line, value=value)[n]

    @staticmethod
    def getHeadMask(cache, value=255):
        parsing = cache.parsing
        mask = np.where((0 < parsing) & (parsing < 15) & (parsing != 12), value, 0).astype(np.uint8)
        return mask

    @staticmethod
    def workOnSelected(source_bgr, blured_bgr, kernel=17, mask=None, **kwargs):
        if mask is None:
            mask = np.ones_like(source_bgr, dtype=np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (kernel, kernel), kernel // 2, kernel // 2)
        multi = mask.astype(np.float32)[:, :, None] / 255.
        fusion = source_bgr * (1 - multi) + blured_bgr * multi
        return np.round(fusion).astype(np.uint8)

    @staticmethod
    def inferenceWithBox(bgr, box, masking_option):
        parameters = masking_option.parameters
        blur_type = parameters['blur_type']
        focus_type = parameters['focus_type']
        if blur_type == 'blur_gaussian':
            blur_kernel = parameters['blur_kernel'] if 'blur_kernel' in parameters else 15
            return LibMasking_Blur.inferenceOnBox_WithBlurGaussian(bgr, box, blur_kernel, focus_type=focus_type)
        if blur_type == 'blur_motion':
            blur_kernel = parameters['blur_kernel'] if 'blur_kernel' in parameters else 15
            return LibMasking_Blur.inferenceOnBox_WithBlurMotion(bgr, box, blur_kernel, focus_type=focus_type)
        if blur_type == 'blur_water':
            rotation_degree = parameters['rotation_degree'] if 'rotation_degree' in parameters else 2
            water_length = parameters['water_length'] if 'water_length' in parameters else 8
            map_coordinates = parameters['map_coordinates'] if 'map_coordinates' in parameters else None
            blur_bgr, map_coordinates = LibMasking_Blur.inferenceOnBox_WithBlurWater(
                bgr, box, rotation_degree, water_length, focus_type=focus_type, map_coordinates=map_coordinates)
            parameters['map_coordinates'] = map_coordinates
            return blur_bgr
        if blur_type == 'blur_pencil':
            k_neigh = parameters['k_neigh'] if 'k_neigh' in parameters else 17
            pre_k = parameters['pre_k'] if 'pre_k' in parameters else 3
            post_k = parameters['post_k'] if 'post_k' in parameters else 3
            return LibMasking_Blur.inferenceOnBox_WithBlurDiffusion(bgr, box, k_neigh, pre_k, post_k, focus_type=focus_type)
        if blur_type == 'blur_diffuse':
            k_neigh = parameters['k_neigh'] if 'k_neigh' in parameters else 17
            return LibMasking_Blur.inferenceOnBox_WithBlurDiffusion(bgr, box, k_neigh, 0, 0, focus_type=focus_type)
        raise NotImplementedError(parameters)

    @staticmethod
    def inferenceOnBox_WithBlurGaussian(bgr, box, blur_kernel, focus_type):
        h, w = bgr.shape[:2]
        blured_bgr = LibMasking_Blur.doWithGaussianBlur(bgr, kernel=blur_kernel)
        if focus_type == 'face':
            # lft, top, rig, bot = box
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            lft, top, rig, bot = box = Rectangle(box).expand(0.2, 0.2).clip(0, 0, w, h).asInt()
            mask[top:bot, lft:rig] = LibMasking_Blur.getFaceMaskByPoints(bgr, box)
        elif focus_type == 'head':
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            lft, top, rig, bot = Rectangle(box).toSquare().expand(0.5, 0.5).clip(0, 0, w, h).asInt()
            mask[top:bot, lft:rig] = LibMasking_Blur.getHeadMask(XPortrait(bgr[top:bot, lft:rig, :]))
            mask[box[3]:, :] = 0
        else:
            mask = LibMasking_Blur.getMaskFromBox(h, w, box, ratio=0.)
        return LibMasking_Blur.workOnSelected(bgr, blured_bgr, mask=mask)

    @staticmethod
    def inferenceOnBox_WithBlurMotion(bgr, box, blur_kernel, focus_type):
        h, w = bgr.shape[:2]
        blured_bgr = LibMasking_Blur.doWithBlurMotion(bgr, kernel=blur_kernel)
        if focus_type == 'face':
            # lft, top, rig, bot = box
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            lft, top, rig, bot = box = Rectangle(box).expand(0.2, 0.2).clip(0, 0, w, h).asInt()
            mask[top:bot, lft:rig] = LibMasking_Blur.getFaceMaskByPoints(bgr, box)
        elif focus_type == 'head':
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            lft, top, rig, bot = Rectangle(box).toSquare().expand(0.5, 0.5).clip(0, 0, w, h).asInt()
            mask[top:bot, lft:rig] = LibMasking_Blur.getHeadMask(XPortrait(bgr[top:bot, lft:rig, :]))
            mask[box[3]:, :] = 0
        else:
            mask = LibMasking_Blur.getMaskFromBox(h, w, box, ratio=0.)
        return LibMasking_Blur.workOnSelected(bgr, blured_bgr, mask=mask)

    @staticmethod
    def inferenceOnBox_WithBlurWater(bgr, box, a, b, focus_type, map_coordinates=None):
        h, w = bgr.shape[:2]
        lft, top, rig, bot = box
        if focus_type == 'face':
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            lft, top, rig, bot = box = Rectangle(box).expand(0.2, 0.2).clip(0, 0, w, h).asInt()
            mask[top:bot, lft:rig] = LibMasking_Blur.getFaceMaskByPoints(bgr, box)
        elif focus_type == 'head':
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            lft, top, rig, bot = Rectangle(box).toSquare().expand(0.5, 0.5).clip(0, 0, w, h).asInt()
            mask[top:bot, lft:rig] = LibMasking_Blur.getHeadMask(XPortrait(bgr[top:bot, lft:rig, :]))
            mask[box[3]:, :] = 0
        else:
            mask = LibMasking_Blur.getMaskFromBox(h, w, box, ratio=0.)
        part = bgr[top:bot, lft:rig, :]
        resized = cv2.resize(part, (256, 256))
        blured_bgr, map_coordinates = LibMasking_Blur.doWithBlurWater(resized, a, b, map_coordinates)
        blured_bgr = cv2.resize(blured_bgr, part.shape[:2][::-1])
        copy_bgr = np.copy(bgr)
        copy_bgr[top:bot, lft:rig, :] = blured_bgr
        return LibMasking_Blur.workOnSelected(bgr, copy_bgr, mask=mask), map_coordinates

    @staticmethod
    def inferenceOnBox_WithBlurDiffusion(bgr, box, k_neigh, pre_k, post_k, focus_type):
        h, w = bgr.shape[:2]
        blured_bgr = LibMasking_Blur.doWithBlurDiffusion(bgr, k_neigh, pre_k, post_k)
        if focus_type == 'face':
            # lft, top, rig, bot = box
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            lft, top, rig, bot = box = Rectangle(box).expand(0.2, 0.2).clip(0, 0, w, h).asInt()
            mask[top:bot, lft:rig] = LibMasking_Blur.getFaceMaskByPoints(bgr, box)
        elif focus_type == 'head':
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            lft, top, rig, bot = Rectangle(box).toSquare().expand(0.5, 0.5).clip(0, 0, w, h).asInt()
            mask[top:bot, lft:rig] = LibMasking_Blur.getHeadMask(XPortrait(bgr[top:bot, lft:rig, :]))
            mask[box[3]:, :] = 0
        else:
            mask = LibMasking_Blur.getMaskFromBox(h, w, box, ratio=0.)
        return LibMasking_Blur.workOnSelected(bgr, blured_bgr, mask=mask)

    """
    """
    @staticmethod
    def toCache(source):
        assert isinstance(source, (np.ndarray, str, XPortrait))
        if isinstance(source, str):
            source = cv2.imread(source)
        return XPortrait.packageAsCache(source, asserting=False)

    @staticmethod
    def blur_ImagesToImages(path_in, path_out):
        assert os.path.isdir(path_in), path_in
        assert os.path.isdir(path_out), path_out
        with XContextTimer(True) as context:
            name_list = sorted(os.listdir(path_in))[:5]
            with tqdm.tqdm(total=len(name_list)) as bar:
                for n, name in enumerate(name_list):
                    bgr = cv2.imread('{}/{}'.format(path_in, name))
                    cache = LibMasking_Blur.toCache(bgr)
                    h, w = cache.shape
                    canvas = np.copy(cache.bgr)
                    for i in range(cache.number):
                        # canvas = LibMasking_Blur.inferenceOnBoxWithBlurGaussian(canvas, cache.box[i, :], 13)
                        # canvas = LibMasking_Blur.inferenceOnBoxWithBlurMotion(canvas, cache.box[i, :], 13)
                        # canvas = LibMasking_Blur.inferenceOnBoxWithBlurWater(canvas, cache.box[i,:], 4, 4)
                        # canvas = LibMasking_Blur.inferenceBoxWithBlurPencil(canvas, cache.box[i, :], 17, 3, 3)
                        canvas = LibMasking_Blur.inferenceOnBox_WithBlurDiffusion(canvas, cache.box[i, :], 17, 0, 0, 'head')
                    cv2.imwrite('{}/{}'.format(path_out, name), canvas)
                    bar.update(1)
            logging.warning('blur finish...')

    @staticmethod
    def blur_VideoToVideo(path_in, path_out, max_num=None):
        assert os.path.exists(path_in), path_in
        reader = XVideoReader(path_in)
        writer = XVideoWriter(reader.desc(True))
        writer.open(path_out)
        with XContextTimer(True):
            num_frames = len(reader)
            max_num = min(max_num, num_frames) if isinstance(max_num, int) else num_frames
            with tqdm.tqdm(total=max_num) as bar:
                for n, bgr in zip(range(max_num), reader):
                    cache = LibMasking_Blur.toCache(bgr)
                    canvas = np.copy(cache.bgr)
                    for i in range(cache.number):
                        # canvas = LibMasking_Blur.inferenceOnBoxWithBlurGaussian(canvas, cache.box[i, :], 13)
                        # canvas = LibMasking_Blur.inferenceOnBoxWithBlurMotion(canvas, cache.box[i, :], 13)
                        # canvas = LibMasking_Blur.inferenceOnBoxWithBlurWater(canvas, cache.box[i,:], 4, 4)
                        # canvas = LibMasking_Blur.inferenceBoxWithBlurPencil(canvas, cache.box[i, :], 17, 3, 3)
                        canvas = LibMasking_Blur.inferenceOnBox_WithBlurDiffusion(canvas, cache.box[i, :], 17, 0, 0, 'head')
                    writer.write(canvas)
                    bar.update(1)
            logging.warning('blur finish...')


