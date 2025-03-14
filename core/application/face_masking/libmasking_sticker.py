
import logging
import os
import cv2
import skimage
import numpy as np
import tqdm
from .boundingbox import BoundingBox
from ...base import XPortrait, XPortraitHelper
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
    def getIndexByBox(cache, box):
        assert isinstance(cache, XPortrait), type(cache)
        if cache.number > 0:
            box_src = np.reshape(np.array(box, dtype=np.int32), (1, 4))
            box_cur = np.reshape(np.array(cache.box, dtype=np.int32), (-1, 4))
            iou = BoundingBox.computeIOU(boxes1=box_src, boxes2=box_cur)  # 1,N
            return int(np.argmax(iou[0, :]))
        else:
            raise ValueError

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
    def inferenceWithBox(bgr, canvas, box, masking_option):
        sticker = masking_option.parameters
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
                bgr_copy = np.copy(canvas)
                bgr_copy[top:bot, lft:rig, :] = fusion_bgr
                return bgr_copy
        if isinstance(sticker, dict):
            sticker_image = sticker['bgr']
            if 'eyes_center' in sticker:
                ratio = 0.2
                h, w, c = bgr.shape
                lft, top, rig, bot = box
                hh = bot - top
                ww = rig - lft
                lft = int(max(0, lft - ww * ratio))
                top = int(max(0, top - hh * ratio))
                rig = int(min(w, rig + ww * ratio))
                bot = int(min(h, bot + hh * ratio))
                part = bgr[top:bot, lft:rig, :]
                cache = XPortrait(part, rotations=[0, 90, 180, 270])
                points_sticker = np.array(sticker['eyes_center'], dtype=np.float32)
                try:
                    n = LibMasking_Sticker.getIndexByBox(cache, box)  # default is 0
                    points_source = XPortraitHelper.getCenterOfEachEyes(cache)[n]  # cache.points[n, :2]
                    points_source[:, 0] += lft
                    points_source[:, 1] += top
                    matrix = cv2.estimateAffinePartial2D(points_sticker, points_source, method=cv2.LMEDS)[0]
                    param = dict(dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
                    sticker_warped = cv2.warpAffine(sticker_image, matrix, **param)
                    sticker_warped_bgr, sticker_warped_alpha = sticker_warped[:, :, :3], sticker_warped[:, :, 3:4]
                    mask = sticker_warped_alpha.astype(np.float32) / 255.
                    fusion = canvas * (1 - mask) + sticker_warped_bgr * mask
                    fusion_bgr = np.round(fusion).astype(np.uint8)
                    return fusion_bgr
                except:
                    return canvas
            if 'eyes_center_fix' in sticker:
                ratio = 0.2
                h, w, c = bgr.shape
                lft, top, rig, bot = box
                hh = bot - top
                ww = rig - lft
                lft = int(max(0, lft - ww * ratio))
                top = int(max(0, top - hh * ratio))
                rig = int(min(w, rig + ww * ratio))
                bot = int(min(h, bot + hh * ratio))
                part = bgr[top:bot, lft:rig, :]
                cache = XPortrait(part, rotations=[0, 90, 180, 270])
                points_sticker = np.array(sticker['eyes_center_fix'], dtype=np.float32)
                try:
                    n = LibMasking_Sticker.getIndexByBox(cache, box)  # default is 0
                    pts_lft = cache.points[n, 0:1].astype(np.float32)
                    pts_rig = cache.points[n, 1:2].astype(np.float32)
                    points_source = np.concatenate([pts_lft, pts_rig], axis=0).astype(np.float32)
                    points_source[:, 0] += lft
                    points_source[:, 1] += top
                    assert len(points_sticker) == len(points_source), (points_sticker.shape, points_source.shape)
                    param = dict(dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
                    transform = skimage.transform.SimilarityTransform()
                    transform.estimate(points_sticker, points_source)
                    sticker_warped = cv2.warpAffine(sticker_image, transform.params[:2, :], **param)
                    sticker_warped_bgr, sticker_warped_alpha = sticker_warped[:, :, :3], sticker_warped[:, :, 3:4]
                    mask = sticker_warped_alpha.astype(np.float32) / 255.
                    fusion = canvas * (1 - mask) + sticker_warped_bgr * mask
                    fusion_bgr = np.round(fusion).astype(np.uint8)
                    return fusion_bgr
                except:
                    return canvas
            if 'align' in sticker:
                ratio = 0.8
                h, w, c = bgr.shape
                lft, top, rig, bot = box
                hh = bot - top
                ww = rig - lft
                lft = int(max(0, lft - ww * ratio))
                top = int(max(0, top - hh * ratio))
                rig = int(min(w, rig + ww * ratio))
                bot = int(min(h, bot + hh * ratio))
                # dst_h, dst_w = bot - top, rig - lft
                part = bgr[top:bot, lft:rig, :3]
                cache = XPortrait(part)
                points_sticker = np.copy(XPortrait(sticker_image[:, :, :3]).points[0])
                try:
                    n = LibMasking_Sticker.getIndexByBox(cache, box)  # default is 0
                    points_source = np.copy(cache.points[n])
                    points_source[:, 0] += lft
                    points_source[:, 1] += top
                    matrix = cv2.estimateAffinePartial2D(points_sticker, points_source, method=cv2.LMEDS)[0]
                    param = dict(dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
                    sticker_warped = cv2.warpAffine(sticker_image, matrix, **param)
                    sticker_warped_bgr, sticker_warped_alpha = sticker_warped[:, :, :3], sticker_warped[:, :, 3:4]
                    mask = sticker_warped_alpha.astype(np.float32) / 255.
                    fusion = canvas * (1 - mask) + sticker_warped_bgr * mask
                    fusion_bgr = np.round(fusion).astype(np.uint8)
                    return fusion_bgr
                except:
                    return canvas
            if 'box' in sticker:
                H, W, C = bgr.shape
                box_src, box_fmt = sticker['box']
                box_remap = BoundingBox.remapBBox(box_src, box_fmt, box)
                lft, top, rig, bot = BoundingBox(np.array(box_remap, dtype=np.int32)).clip(0, 0, W, H).decouple()
                h = bot - top
                w = rig - lft
                st_x, st_y, st_w, st_h = cv2.boundingRect(sticker_image[:, :, 3])
                if st_w == 0 or st_h == 0:
                    return np.copy(bgr)  # bug
                sticker_image = sticker_image[st_y:st_y + st_h, st_x:st_x + st_w, ...]
                resized_sticker = cv2.resize(sticker_image, (w, h))
                sticker_bgr = resized_sticker[:, :, :3]
                sticker_mask = resized_sticker[:, :, 3:4]
                part = bgr[top:bot, lft:rig, :]
                mask = sticker_mask.astype(np.float32) / 255.
                fusion = part * (1 - mask) + sticker_bgr * mask
                fusion_bgr = np.round(fusion).astype(np.uint8)
                bgr_copy = np.copy(canvas)
                bgr_copy[top:bot, lft:rig, :] = fusion_bgr
                return bgr_copy
            if 'paste' in sticker:
                ltrb_ori, ltrb = sticker['paste']
                image_c = bgr[ltrb[1]:ltrb[3], ltrb[0]:ltrb[2], :]
                # image_ori_c = bgr[ltrb_ori[1]:ltrb_ori[3], ltrb_ori[0]:ltrb_ori[2], :]
                w_ori = ltrb[2] - ltrb[0]
                h_ori = ltrb[3] - ltrb[1]
                result_nd = sticker_image[:, :, :3]
                result_alpha = sticker_image[:, :, 3]
                result_nd_resize = cv2.resize(result_nd, (w_ori, h_ori))
                result_alpha_resize = cv2.resize(result_alpha, (w_ori, h_ori))
                result_alpha_resize = result_alpha_resize[:, :, np.newaxis] / 255.0
                image_C_cartoon = result_nd_resize * result_alpha_resize + (1 - result_alpha_resize) * image_c
                bgr_copy = np.copy(canvas)
                bgr_copy[ltrb[1]:ltrb[3], ltrb[0]:ltrb[2], :] = image_C_cartoon
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
