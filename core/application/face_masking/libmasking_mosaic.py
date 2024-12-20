
import logging
import os
import cv2
import numpy as np
import tqdm
from ...base import XPortrait
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

    @staticmethod
    def resizeDownAndUp(bgr, box, num_pixels):
        lft, top, rig, bot = box
        h = bot - top
        w = rig - lft
        hh = num_pixels  # int((h / 20 - h / 10) / 4 * value + h / 10)
        ww = num_pixels  # int((w / 20 - w / 10) / 4 * value + w / 10)
        crop = bgr[top:bot, lft:rig, :]
        sub = cv2.resize(crop, (ww, hh))
        up = cv2.resize(sub, (w, h), interpolation=cv2.INTER_NEAREST)
        bgr[top:bot, lft:rig, :] = up
        return bgr

    @staticmethod
    def doWithMosaic(bgr, value=5, boxes=None):
        if boxes is None:
            cache = XPortrait(bgr, asserting=False)
            boxes = np.copy(cache.box)
        assert boxes.shape[1] == 4, boxes.shape
        canvas = np.copy(bgr)
        for n in range(len(boxes)):
            canvas = LibMasking_Mosaic.resizeDownAndUp(canvas, boxes[n, :], value)
        return canvas

    @staticmethod
    def toCache(source):
        assert isinstance(source, (np.ndarray, str, XPortrait))
        if isinstance(source, str):
            source = cv2.imread(source)
        return XPortrait.packageAsCache(source, asserting=False)

    @staticmethod
    def inference(reader_iterator, writer_interator, **kwargs):
        with tqdm.tqdm(total=len(reader_iterator)) as bar:
            for n, source in enumerate(reader_iterator):
                cache = LibMasking_Mosaic.toCache(source)
                mosaic_bgr = LibMasking_Mosaic.doWithMosaic(cache.bgr)
                writer_interator(mosaic_bgr)
                bar.update(1)

    @staticmethod
    def inferenceOnBox(bgr, box, num_pixel=5):
        return LibMasking_Mosaic.resizeDownAndUp(bgr, box, num_pixel)

    """
    """
    @staticmethod
    def mosaic_ImagesToImages(path_in, path_out):
        assert os.path.isdir(path_in), path_in
        assert os.path.isdir(path_out), path_out
        with XContextTimer(True) as context:
            name_list = sorted(os.listdir(path_in))[:750]
            with tqdm.tqdm(total=len(name_list)) as bar:
                for n, name in enumerate(name_list):
                    bgr = cv2.imread('{}/{}'.format(path_in, name))
                    cache = LibMasking_Mosaic.toCache(bgr)
                    mosaic_bgr = LibMasking_Mosaic.doWithMosaic(cache.bgr)
                    cv2.imwrite('{}/{}'.format(path_out, name), mosaic_bgr)
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
                    mosaic_bgr = LibMasking_Mosaic.doWithMosaic(cache.bgr)
                    writer.write(mosaic_bgr)
                    bar.update(1)
            logging.warning('blur finish...')