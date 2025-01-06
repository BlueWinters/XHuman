
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
        if isinstance(sticker, np.ndarray):
            assert sticker.shape[2] == 3 or sticker.shape[2] == 4, sticker.shape[2]
            lft, top, rig, bot = box
            h = bot - top
            w = rig - lft
            if sticker.shape[2] == 3:
                bgr_copy = np.copy(bgr)
                bgr_copy[top:bot, lft:rig, :] = cv2.resize(sticker, (w, h))
                return bgr_copy
            if sticker.shape[2] == 4:
                resized_sticker = cv2.resize(sticker, (w, h))
                sticker_bgr = resized_sticker[:, :, :3]
                sticker_mask = resized_sticker[:, :, 3:4]
                # mask = np.zeros_like(shape=bgr.shape)
                # bgr_copy = np.copy(bgr)
                # bgr_copy[top:bot, lft:rig, :] = sticker_bgr
                # mask[top:bot, lft:rig, :] = sticker_mask
                # multi = mask.astype(np.float32) / 255.
                # bgr = np.round(bgr * (1 - multi) + bgr_copy * multi).astype(np.uint8)
                part = bgr[top:bot, lft:rig, :]
                mask = sticker_mask.astype(np.float32) / 255.
                fusion = part * (1 - mask) + sticker_bgr * mask
                fusion_bgr = np.round(fusion).astype(np.uint8)
                bgr_copy = np.copy(bgr)
                bgr_copy[top:bot, lft:rig, :] = fusion_bgr
                return bgr_copy
        if isinstance(sticker, dict):
            sticker_image = sticker['bgr']
            if 'eyes_center' in sticker:
                ratio = 0.5
                h, w, c = bgr.shape
                lft, top, rig, bot = box
                hh = bot - top
                ww = rig - lft
                lft = int(max(0, lft - ww * ratio))
                top = int(max(0, top - hh * ratio))
                rig = int(min(w, rig + ww * ratio))
                bot = int(min(h, bot + hh * ratio))
                dst_h, dst_w = bot - top, rig - lft
                part = bgr[top:bot, lft:rig, :]
                cache = XPortrait(part)
                points_sticker = sticker['eyes_center']
                points_source = cache.points[0, :2]
                matrix = cv2.estimateAffinePartial2D(points_sticker, points_source, method=cv2.LMEDS)[0]
                param = dict(dsize=(dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
                sticker_warped = cv2.warpAffine(sticker_image, matrix, **param)
                sticker_warped_bgr, sticker_warped_alpha = sticker_warped[:, :, :3], sticker_warped[:, :, 3:4]
                mask = sticker_warped_alpha.astype(np.float32) / 255.
                fusion = part * (1 - mask) + sticker_warped_bgr * mask
                fusion_bgr = np.round(fusion).astype(np.uint8)
                bgr_copy = np.copy(bgr)
                bgr_copy[top:bot, lft:rig, :] = fusion_bgr
                return bgr_copy
            if 'box' in sticker:
                sticker_box = sticker['box']
                lft, top, rig, bot = sticker_box
                h = bot - top
                w = rig - lft
                if sticker_image.shape[2] == 3:
                    bgr_copy = np.copy(bgr)
                    bgr_copy[top:bot, lft:rig, :] = cv2.resize(sticker_image, (w, h))
                    return bgr_copy
                if sticker_image.shape[2] == 4:
                    resized_sticker = cv2.resize(sticker_image, (w, h))
                    sticker_bgr = resized_sticker[:, :, :3]
                    sticker_mask = resized_sticker[:, :, 3:4]
                    part = bgr[top:bot, lft:rig, :]
                    mask = sticker_mask.astype(np.float32) / 255.
                    fusion = part * (1 - mask) + sticker_bgr * mask
                    fusion_bgr = np.round(fusion).astype(np.uint8)
                    bgr_copy = np.copy(bgr)
                    bgr_copy[top:bot, lft:rig, :] = fusion_bgr
                    return bgr_copy

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