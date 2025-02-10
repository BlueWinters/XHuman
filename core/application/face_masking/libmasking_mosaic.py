
import logging
import os
import cv2
import numpy as np
import tqdm
from skimage import segmentation
from .boundingbox import Rectangle, BoundingBox
from .helper.image_helper import MaskingHelper
from ...base.cache import XPortrait, XPortraitHelper
from ...thirdparty.cache import XBody
from ...utils.context import XContextTimer
from ...utils.video import XVideoReader, XVideoWriter
from ... import XManager


class LibMasking_Mosaic:
    """
    """
    @staticmethod
    def benchmark():
        pass

    @staticmethod
    def prepare():
        bgr = cv2.imread('benchmark/asset/prepare/input.png')
        box = LibMasking_Mosaic.toCache(bgr).box
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
    def resizeDownAndUp(bgr, box, nh, nw):
        lft, top, rig, bot = box
        h = bot - top
        w = rig - lft
        hh = nw  # int((h / 20 - h / 10) / 4 * value + h / 10)
        ww = nh  # int((w / 20 - w / 10) / 4 * value + w / 10)
        crop = bgr[top:bot, lft:rig, :]
        sub = cv2.resize(crop, (ww, hh))
        up = cv2.resize(sub, (w, h), interpolation=cv2.INTER_NEAREST)
        bgr_copy = np.copy(bgr)
        bgr_copy[top:bot, lft:rig, :] = up
        return bgr_copy

    @staticmethod
    def inferenceBoxWithMosaicSquare(bgr, box, num_pixels, focus_type):
        if focus_type == 'head' or focus_type == 'face':
            cache = XPortrait(bgr)
            h, w = cache.shape
            # lft, top, rig, bot = box
            lft, top, rig, bot = Rectangle(box).toSquare().expand(0.8, 0.8).clip(0, 0, w, h).asInt()
            hh = bot - top
            ww = rig - lft
            nh = int(float(h / hh) * num_pixels)
            nw = int(float(w / ww) * num_pixels)
            num_pixels = int((nh + nw) / 2)
            mosaic_bgr = LibMasking_Mosaic.resizeDownAndUp(bgr, [0, 0, w, h], num_pixels, num_pixels)
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            if focus_type == 'face':
                mask[top:bot, lft:rig] = LibMasking_Mosaic.getFaceMaskByPoints(XPortrait(bgr[top:bot, lft:rig, :]))
            if focus_type == 'head':
                mask[top:bot, lft:rig] = LibMasking_Mosaic.getHeadMask(bgr, box_src=box, box_tar=(lft, top, rig, bot))
                mask[box[3]:, :] = 0
            return LibMasking_Mosaic.workOnSelected(cache.bgr, mosaic_bgr, mask=mask)
        else:
            return LibMasking_Mosaic.resizeDownAndUp(bgr, box, num_pixels, num_pixels)

    """
    """
    @staticmethod
    def doSuperPixel(bgr, **kwargs):
        kernel = 11
        bgr = cv2.GaussianBlur(bgr, (kernel, kernel), sigmaX=kernel // 2, sigmaY=kernel // 2)
        n_div = kwargs.pop('n_div', 32)
        mesh_size = kwargs.pop('mesh_size', 32)
        n_div = n_div if n_div > 0 else int((bgr.shape[0] * bgr.shape[1]) / (mesh_size * mesh_size))
        h, w, c = bgr.shape
        n_segments = n_div * int(n_div * max(h, w) / min(h, w))
        compactness = kwargs.pop('compactness', 1)
        mask = kwargs['mask'] if 'mask' in kwargs else None
        segments = segmentation.slic(bgr, n_segments=n_segments, slic_zero=True, compactness=compactness, start_label=1, mask=mask)
        # segments = cv2.resize(segments, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        # segments = cv2.resize(segments, (w // 1, h // 1), interpolation=cv2.INTER_NEAREST)
        return segments, n_segments

    @staticmethod
    def visualAsMean(label_field, bgr, bg_label=0, bg_color=(255, 255, 255)):
        out = np.zeros(label_field.shape + (3,), dtype=bgr.dtype)
        labels = np.unique(label_field)
        if (labels == bg_label).any():
            labels = labels[labels != bg_label]
            mask = (label_field == bg_label).nonzero()
            out[mask] = bg_color
        for label in labels:
            mask = (label_field == label).nonzero()
            color = bgr[mask].mean(axis=0).astype(np.uint8)
            out[mask] = color
        return out

    @staticmethod
    def doPostprocess(bgr, seg, vis_boundary):
        bgr_copy = np.copy(bgr)
        bgr_copy = LibMasking_Mosaic.visualAsMean(seg, bgr_copy, 0)
        if vis_boundary is True:
            # bgr_copy = segmentation.mark_boundaries(bgr_copy, seg, color=(1, 1, 1), mode='outer')
            # bgr_copy = np.round(bgr_copy * 255).astype(np.uint8)
            # TODO: debug
            boundary = segmentation.find_boundaries(seg, mode='thick').astype(np.uint8) * 128
            boundary = cv2.resize(boundary, bgr.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            multi = boundary.astype(np.float32)[:, :, None] / 255.
            fusion = bgr_copy.astype(np.float32) * (1-multi) + np.ones_like(bgr_copy) * 255 * multi
            bgr_copy = np.round(fusion).astype(np.uint8)
        return bgr_copy

    @staticmethod
    def doWithMosaicPolygon(bgr, vis_boundary, **kwargs):
        super_pixels = kwargs.pop('super_pixels', None)
        super_pixels = super_pixels if super_pixels is not None \
            else LibMasking_Mosaic.doSuperPixel(bgr, **kwargs)[0]
        new_bgr = LibMasking_Mosaic.doPostprocess(bgr, super_pixels, vis_boundary)
        return new_bgr, super_pixels

    @staticmethod
    def workOnSelected(source_bgr, blured_bgr, kernel=0, mask=None, **kwargs):
        if mask is None:
            mask = np.ones_like(source_bgr, dtype=np.uint8) * 255
        if kernel > 0:
            mask = cv2.GaussianBlur(mask, (kernel, kernel), kernel // 2, kernel // 2)
        multi = mask.astype(np.float32)[:, :, None] / 255.
        fusion = source_bgr * (1 - multi) + blured_bgr * multi
        return np.round(fusion).astype(np.uint8)

    @staticmethod
    def getFaceMaskByPoints(cache, n=0, top_line='brow', value=255):
        return XPortraitHelper.getFaceRegion(cache, index=n, top_line=top_line, value=value)[n]

    @staticmethod
    def getHeadMask(bgr, box_src, box_tar, value=255):
        return MaskingHelper.getHeadMaskByParsing(bgr, box_src, box_tar, value)

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
    def inferenceBoxWithMosaicPolygon(bgr, box, n_div, vis_boundary, focus_type, super_pixels=None):
        if focus_type == 'head' or focus_type == 'face':
            h, w, c = bgr.shape
            lft, top, rig, bot = Rectangle(box).toSquare().expand(0.8, 0.8).clip(0, 0, w, h).asInt()
            part = bgr[top:bot, lft:rig]
            size = float(sum(part.shape[:2])) / 2  # (h+w)/2
            # n_div = int(size / 256 * n_div)
            part_resized = cv2.resize(part, (384, 384))
            mosaic_bgr, super_pixels = LibMasking_Mosaic.doWithMosaicPolygon(
                part_resized, vis_boundary, n_div=n_div, super_pixels=super_pixels)
            bgr_copy = np.copy(bgr)
            bgr_copy[top:bot, lft:rig] = cv2.resize(mosaic_bgr, part.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            if focus_type == 'face':
                mask[top:bot, lft:rig] = LibMasking_Mosaic.getFaceMaskByPoints(XPortrait(part))
            if focus_type == 'head':
                mask[top:bot, lft:rig] = LibMasking_Mosaic.getHeadMask(bgr, box_src=box, box_tar=(lft, top, rig, bot))
                mask[box[3]:, :] = 0
            mesh_size = int(size / 256 * 7 / 4)
            mesh_size = int(np.clip(mesh_size, 3, 13))
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mesh_size, mesh_size)))
            bgr_copy = LibMasking_Mosaic.workOnSelected(bgr, bgr_copy, mask=mask)
            return bgr_copy, super_pixels
        else:
            lft, top, rig, bot = box
            part = np.copy(bgr[top:bot, lft:rig])
            # size = float(sum(part.shape[:2])) / 2  # (h+w)/2
            # n_div = int(size / 256 * n_div)
            mosaic_bgr, super_pixels = LibMasking_Mosaic.doWithMosaicPolygon(part, vis_boundary, n_div=n_div, super_pixels=super_pixels)
            bgr_copy = np.copy(bgr)
            bgr_copy[top:bot, lft:rig] = mosaic_bgr
            return bgr_copy, super_pixels

    @staticmethod
    def inferenceWithBox(bgr, box, masking_option):
        parameters = masking_option.parameters
        mosaic_type = parameters['mosaic_type']
        focus_type = parameters['focus_type']
        super_pixels = parameters['super_pixels'] if 'super_pixels' in parameters else None
        if mosaic_type == 'mosaic_pixel_square':
            num_pixel = parameters['num_pixel'] if 'num_pixel' in parameters else 48
            return LibMasking_Mosaic.inferenceBoxWithMosaicSquare(bgr, box, num_pixel, focus_type=focus_type)
        if mosaic_type == 'mosaic_pixel_polygon':
            n_div = parameters['n_div'] if 'n_div' in parameters else 32
            vis_boundary = parameters['vis_boundary'] if 'vis_boundary' in parameters else False
            mosaic_bgr, super_pixels = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(
                bgr, box, n_div, vis_boundary=vis_boundary, focus_type=focus_type, super_pixels=super_pixels)
            parameters['super_pixels'] = super_pixels
            return mosaic_bgr
        # specific config
        if mosaic_type == 'mosaic_pixel_polygon_small':
            # n_div = parameters['n_div'] if 'n_div' in parameters else 8
            mosaic_bgr, super_pixels = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(
                bgr, box, 24, vis_boundary=False, focus_type=focus_type, super_pixels=super_pixels)
            parameters['super_pixels'] = super_pixels
            return mosaic_bgr
        if mosaic_type == 'mosaic_pixel_polygon_small_line':
            # n_div = parameters['n_div'] if 'n_div' in parameters else 8
            mosaic_bgr, super_pixels = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(
                bgr, box, 16, vis_boundary=True, focus_type=focus_type, super_pixels=super_pixels)
            parameters['super_pixels'] = super_pixels
            return mosaic_bgr
        if mosaic_type == 'mosaic_pixel_polygon_big':
            # n_div = parameters['n_div'] if 'n_div' in parameters else 16
            mosaic_bgr, super_pixels = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(
                bgr, box, 24, vis_boundary=False, focus_type=focus_type, super_pixels=super_pixels)
            parameters['super_pixels'] = super_pixels
            return mosaic_bgr
        if mosaic_type == 'mosaic_pixel_polygon_big_line':
            # n_div = parameters['n_div'] if 'n_div' in parameters else 16
            mosaic_bgr, super_pixels = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(
                bgr, box, 24, vis_boundary=True, focus_type=focus_type, super_pixels=super_pixels)
            parameters['super_pixels'] = super_pixels
            return mosaic_bgr
        raise NotImplementedError(parameters)

    """
    """
    @staticmethod
    def toCache(source):
        assert isinstance(source, (np.ndarray, str, XPortrait))
        if isinstance(source, str):
            source = cv2.imread(source)
        return XPortrait.packageAsCache(source, asserting=False)

    @staticmethod
    def mosaic_ImagesToImages(path_in, path_out):
        assert os.path.isdir(path_in), path_in
        assert os.path.isdir(path_out), path_out
        with XContextTimer(True) as context:
            name_list = sorted(os.listdir(path_in))[:2]
            with tqdm.tqdm(total=len(name_list)) as bar:
                for n, name in enumerate(name_list):
                    bgr = cv2.imread('{}/{}'.format(path_in, name))
                    cache = LibMasking_Mosaic.toCache(bgr)
                    canvas = np.copy(cache.bgr)
                    for i in range(cache.number):
                        # canvas = LibMasking_Mosaic.inferenceBoxWithMosaicSquare(canvas, cache.box[i, :], 5)
                        canvas = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 32, True)
                        # canvas = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], False)
                    cv2.imwrite('{}/{}'.format(path_out, name), canvas)
                    bar.update(1)
            logging.warning('blur finish...')

    @staticmethod
    def mosaic_VideoToVideo(path_in, path_out, max_num=None):
        assert os.path.exists(path_in), path_in
        reader = XVideoReader(path_in)
        writer = XVideoWriter(reader.desc(True))
        writer.open(path_out)
        with XContextTimer(True) as context:
            num_frames = len(reader)
            max_num = min(max_num, num_frames) if isinstance(max_num, int) else num_frames
            with tqdm.tqdm(total=max_num) as bar:
                for n, bgr in zip(range(max_num), reader):
                    cache = LibMasking_Mosaic.toCache(bgr)
                    mosaic_bgr = LibMasking_Mosaic.inferenceBoxWithMosaicSquare(cache.bgr)
                    writer.write(mosaic_bgr)
                    bar.update(1)
            logging.warning('blur finish...')