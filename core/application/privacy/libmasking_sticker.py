
import logging
import os
import cv2
import numpy as np
import tqdm
from ...base import XPortrait
from ...utils.context import XContextTimer
from ...utils.video import XVideoReader, XVideoWriter
from ...utils.resource import Resource
from ... import XManager


class LibMasking_Sticker:
    """
    """
    @staticmethod
    def benchmark():
        pass

    @staticmethod
    def prepare():
        bgr = cv2.imread('benchmark/asset/prepare/input.png')
        box = LibMasking_Sticker.toCache(bgr).box
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
    def doWithPasteSticker(bgr, sticker=None, boxes=None):
        if boxes is None:
            cache = XPortrait(bgr, asserting=False)
            boxes = np.copy(cache.box)
        if sticker is None:
            path = '{}/cartoon/00.png'.format(os.path.split(__file__)[0])
            sticker = Resource.loadImage(path)
        assert boxes.shape[1] == 4, boxes.shape
        canvas = np.copy(bgr)
        for n in range(len(boxes)):
            lft, top, rig, bot = cache.box[n]
            h = bot - top
            w = rig - lft
            canvas[top:bot, lft:rig, :] = cv2.resize(sticker, (w, h))
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
                cache = LibMasking_Sticker.toCache(source)
                mosaic_bgr = LibMasking_Sticker.doWithPasteSticker(cache.bgr)
                writer_interator(mosaic_bgr)
                bar.update(1)

    @staticmethod
    def inferenceWithBox(bgr, box, sticker):
        lft, top, rig, bot = box
        h = bot - top
        w = rig - lft
        bgr[top:bot, lft:rig, :] = cv2.resize(sticker, (w, h))
        return bgr

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
                    cache = LibMasking_Sticker.toCache(bgr)
                    mosaic_bgr = LibMasking_Sticker.doWithPasteSticker(cache.bgr)
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
                    cache = LibMasking_Sticker.toCache(bgr)
                    mosaic_bgr = LibMasking_Sticker.doWithPasteSticker(cache.bgr)
                    writer.write(mosaic_bgr)
                    bar.update(1)
            logging.warning('blur finish...')