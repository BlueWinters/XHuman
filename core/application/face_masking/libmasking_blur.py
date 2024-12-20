
import logging
import os
import cv2
import numpy as np
import random
import skimage
import pickle
import typing
import tqdm
from ...base import XPortrait
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

    @staticmethod
    def doWithGaussianBlur(bgr, kernel=15, blur_size=256):
        # if box is not None:
        #     lft, top, rig, bot = box
        #     size = max(bot - top, rig - lft)
        #     kernel_max = int(size // 10)  # 5
        #     kernel_min = int(size // 20)  # 1
        #     kernel = int(kernel_min + (kernel_max - kernel_min) / 4 * value)
        format_bgr, padding = LibMasking_Blur.formatSizeWithPaddingForward(bgr, blur_size, blur_size)
        kernel = kernel if kernel % 2 == 1 else (kernel + 1)  # should be odd
        blured_bgr = cv2.GaussianBlur(format_bgr, (kernel, kernel), kernel // 2, kernel // 2)
        reformat_blur_bgr = LibMasking_Blur.formatSizeWithPaddingBackward(bgr, blured_bgr, padding)
        return reformat_blur_bgr

    @staticmethod
    def toCache(source):
        assert isinstance(source, (np.ndarray, str, XPortrait))
        if isinstance(source, str):
            source = cv2.imread(source)
        return XPortrait.packageAsCache(source, asserting=False)

    @staticmethod
    def getMask(cache):
        mask = np.ones(shape=cache.shape, dtype=np.uint8)
        for n in range(len(cache.box)):
            lft, top, rig, bot = cache.box[n, :]
            mask[top:bot, lft:rig] = 255
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
    def inference(reader_iterator, writer_interator, **kwargs):
        with tqdm.tqdm(total=len(reader_iterator)) as bar:
            for n, source in enumerate(reader_iterator):
                cache = LibMasking_Blur.toCache(source)
                blured_bgr = LibMasking_Blur.doWithGaussianBlur(cache.bgr)
                canvas = LibMasking_Blur.workOnSelected(cache.bgr, blured_bgr, mask=LibMasking_Blur.getMask(cache))
                writer_interator(canvas)
                bar.update(1)

    @staticmethod
    def inferenceOnBox(bgr, box, blur_kernel):
        mask = np.ones(shape=bgr.shape[:2], dtype=np.uint8)
        lft, top, rig, bot = box
        mask[top:bot, lft:rig] = 255
        blured_bgr = LibMasking_Blur.doWithGaussianBlur(bgr, kernel=blur_kernel)
        return LibMasking_Blur.workOnSelected(bgr, blured_bgr, mask=mask)

    """
    """
    @staticmethod
    def blur_ImagesToImages(path_in, path_out):
        assert os.path.isdir(path_in), path_in
        assert os.path.isdir(path_out), path_out
        with XContextTimer(True) as context:
            name_list = sorted(os.listdir(path_in))[:750]
            with tqdm.tqdm(total=len(name_list)) as bar:
                for n, name in enumerate(name_list):
                    bgr = cv2.imread('{}/{}'.format(path_in, name))
                    cache = LibMasking_Blur.toCache(bgr)
                    blured_bgr = LibMasking_Blur.doWithGaussianBlur(cache.bgr)
                    canvas = LibMasking_Blur.workOnSelected(cache.bgr, blured_bgr, mask=LibMasking_Blur.getMask(cache))
                    cv2.imwrite('{}/{}'.format(path_out, name), canvas)
                    bar.update(1)
            logging.warning('blur finish...')

    @staticmethod
    def blur_VideoToVideo(path_in, path_out, max_num=None):
        assert os.path.exists(path_in), path_in
        reader = XVideoReader(path_in)
        writer = XVideoWriter(reader.desc(True))
        writer.open(path_out)
        with XContextTimer(True) as context:
            num_frames = len(reader)
            max_num = min(max_num, num_frames) if isinstance(max_num, int) else num_frames
            with tqdm.tqdm(total=max_num) as bar:
                for n, bgr in zip(range(max_num), reader):
                    cache = LibMasking_Blur.toCache(bgr)
                    blured_bgr = LibMasking_Blur.doWithGaussianBlur(cache.bgr)
                    canvas = LibMasking_Blur.workOnSelected(cache.bgr, blured_bgr, mask=LibMasking_Blur.getMask(cache))
                    writer.write(canvas)
                    bar.update(1)
            logging.warning('blur finish...')