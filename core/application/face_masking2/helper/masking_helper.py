import logging

import cv2
import numpy as np
import skimage
from .cursor import AsynchronousCursor
from .angle_helper import AngleHelper
from ..scanning_image import *
from ..scanning_video import *
from ....base import XPortrait, XPortraitHelper
from ....geometry import Rectangle, GeoFunction
from .... import XManager


class MaskingHelper:
    @staticmethod
    def formatSizeWithPaddingForward(bgr, dst_h, dst_w, padding_value=255) -> (np.ndarray, tuple):
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
    def formatSizeWithPaddingBackward(bgr_src, bgr_cur, padding) -> np.ndarray:
        tp, bp, lp, rp = padding
        bp = bgr_cur.shape[0] - bp  # 0 --> h
        rp = bgr_cur.shape[1] - rp  # 0 --> w
        src_h, src_w, _ = bgr_src.shape
        resized = cv2.resize(bgr_cur[tp:bp, lp:rp, :], (src_w, src_h))
        return resized

    """
    """

    @staticmethod
    def getSingleConnectedRegion(mask):
        label_image = skimage.measure.label(mask.astype(np.uint8), connectivity=2, return_num=False)
        regions = skimage.measure.regionprops(label_image)
        if not regions:
            return np.zeros_like(mask, dtype=bool)
        max_region = max(regions, key=lambda r: r.area)
        max_mask = (label_image == max_region.label).astype(np.uint8)
        return max_mask

    @staticmethod
    def getPortraitMaskingWithInfoImagePlus(bgr, objects_list, option_dict, with_hair=True, expand=0.8):
        h, w, c = bgr.shape
        # segmentation
        module = XManager.getModules('ultralytics')['yolo11m-seg']
        result = module(bgr, classes=[0], verbose=False)[0]
        masks = list()
        if len(result) > 0 and result.masks is not None:
            result_masks = np.round(result.masks.cpu().numpy().data * 255).astype(np.uint8)  # note: C,H,W and [0,1]
            masks = [cv2.resize(result_masks[n, :, :], (w, h)) for n in range(len(result_masks))]
        mask_dict = dict()
        for n, info_object in enumerate(objects_list):
            if info_object.identity not in option_dict:
                continue
            if option_dict[info_object.identity].NameEN.startswith('sticker'):
                continue
            if isinstance(info_object, InfoImage_Plate):
                lft, top, rig, bot = info_object.box
                mask = np.zeros(shape=(h, w), dtype=np.uint8)
                mask[top:bot, lft:rig] = 255
                setattr(info_object, 'mask_info', dict(mask=mask, box=(lft, top, rig, bot)))
                continue
            if isinstance(info_object, InfoImage_Person):
                lft, top, rig, bot = Rectangle(info_object.box).expand(expand, expand).clip(0, 0, w, h).asInt()
                bgr_copy = np.copy(bgr)
                # 1.
                # for i in range(len(info_image)):
                #     if i != n:
                #         face_mask = XPortraitHelper.getFaceRegionByLandmark(h, w, info_image.info_person_list[i].landmark)
                #         bgr_copy[face_mask > 0] = 255
                # 2.
                if len(result) > 0 and result.masks is not None and len(masks) > 0:
                    mask_box = np.zeros(shape=(h, w), dtype=np.uint8)
                    l, t, r, b = info_object.box
                    mask_box[t:b, l:r] = 255
                    count_nonzero = []
                    for j in range(len(masks)):
                        mm = ((masks[j] > 0) & (mask_box > 0)).astype(np.uint8)
                        count_nonzero.append(np.count_nonzero(mm))
                    index = int(np.argmax(np.array(count_nonzero, dtype=np.int32)))
                    mask_cur = masks[index]
                    bgr_copy[mask_cur == 0] = 255
                # portrait parsing
                # ajna = np.mean(info_person.landmark[17:27, :], axis=0)
                # angle = 0 if ajna[1] < info_person.landmark[30, 1] else 180  # or info_person.angle
                # angle = info_person.angle
                angle = AngleHelper.getAngleRollByLandmark(info_object.landmark)
                logging.info('identity-{}, angle-{}'.format(info_object.identity, angle))
                part = bgr_copy[top:bot, lft:rig]
                part_rot = GeoFunction.rotateImage(part, angle)
                part_rot_parsing = XManager.getModules('portrait_parsing')(part_rot)
                part_parsing = GeoFunction.rotateImage(part_rot_parsing, GeoFunction.rotateBack(angle))
                if with_hair is False:
                    part_mask = np.where((0 < part_parsing) & (part_parsing < 15) & (part_parsing != 12) & (part_parsing != 13), 255, 0).astype(np.uint8)
                else:
                    part_mask = np.where((0 < part_parsing) & (part_parsing < 15) & (part_parsing != 12), 255, 0).astype(np.uint8)
                part_mask_single = MaskingHelper.getSingleConnectedRegion(part_mask) * 255
                mask = np.zeros(shape=(h, w), dtype=np.uint8)
                mask[top:bot, lft:rig] = part_mask_single
                mask_dict[info_object.identity] = mask
                setattr(info_object, 'mask_info', dict(mask=mask, box=(lft, top, rig, bot)))
                # cv2.imwrite(R'N:\archive\2025\0215-masking\error_image\01\parsing\{}-bgr_copy.png'.format(n), bgr_copy)
                # cv2.imwrite(R'N:\archive\2025\0215-masking\error_image\01\parsing\{}-parsing.png'.format(n), XManager.getModules('portrait_parsing').colorize(part_rot_parsing))
        return mask_dict

    @staticmethod
    def getPortraitMaskingWithInfoImage(bgr, info_image, option_dict, with_hair=True, expand=0.8):
        h, w, c = bgr.shape
        mask_dict = dict()
        for n, info_person in enumerate(info_image):
            assert isinstance(info_person, InfoImage_Person)
            if info_person.identity not in option_dict:
                continue
            if option_dict[info_person.identity].NameEN.startswith('sticker'):
                continue
            lft, top, rig, bot = Rectangle(info_person.box).expand(expand, expand).clip(0, 0, w, h).asInt()
            bgr_copy = np.copy(bgr)
            for i in range(len(info_image)):
                if i != n:
                    face_mask = XPortraitHelper.getFaceRegionByLandmark(h, w, info_image.info_object_list[i].landmark)
                    bgr_copy[face_mask > 0] = 255
            part = bgr_copy[top:bot, lft:rig]
            ajna = np.mean(info_person.landmark[17:27, :], axis=0)
            angle = 0 if ajna[1] < info_person.landmark[30, 1] else 180  # info_person.angle
            part_rot = GeoFunction.rotateImage(part, angle)
            part_rot_parsing = XManager.getModules('portrait_parsing')(part_rot)
            part_parsing = GeoFunction.rotateImage(part_rot_parsing, GeoFunction.rotateBack(angle))
            if with_hair is False:
                part_mask = np.where((0 < part_parsing) & (part_parsing < 15) & (part_parsing != 12) & (part_parsing != 13), 255, 0).astype(np.uint8)
            else:
                part_mask = np.where((0 < part_parsing) & (part_parsing < 15) & (part_parsing != 12), 255, 0).astype(np.uint8)
            part_mask_single = MaskingHelper.getSingleConnectedRegion(part_mask) * 255
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            mask[top:bot, lft:rig] = part_mask_single
            mask_dict[info_person.identity] = mask
            setattr(info_person, 'mask_info', dict(mask=mask, box=(lft, top, rig, bot)))
            # cv2.imwrite(R'N:\archive\2025\0215-masking\error_image\01\parsing\{}-bgr_copy.png'.format(n), bgr_copy)
            # cv2.imwrite(R'N:\archive\2025\0215-masking\error_image\01\parsing\{}-parsing.png'.format(n), XManager.getModules('portrait_parsing').colorize(part_rot_parsing))
        return mask_dict

    @staticmethod
    def getPortraitMaskingWithInfoVideoPlus(frame_index, frame_bgr, cursor_list, option_dict, with_hair=True):
        h, w, c = frame_bgr.shape
        # segmentation
        module = XManager.getModules('ultralytics')['yolo11m-seg']
        result = module(frame_bgr, classes=[0], verbose=False)[0]
        masks = list()
        if len(result) > 0 and result.masks is not None:
            result_masks = np.round(result.masks.cpu().numpy().data * 255).astype(np.uint8)  # note: C,H,W and [0,1]
            masks = [cv2.resize(result_masks[n, :, :], (w, h)) for n in range(len(result_masks))]
        # main pipeline
        mask_dict = dict()
        for n, (info_object, cursor) in enumerate(cursor_list):
            if info_object.identity not in option_dict:
                continue
            if option_dict[info_object.identity].NameEN.startswith('sticker'):
                mask_dict[info_object.identity] = None
                continue
            if isinstance(info_object, InfoVideo_Person) is False:
                continue
            # get mask
            assert isinstance(cursor, AsynchronousCursor), cursor
            frame_info = cursor.current()
            assert isinstance(frame_info, InfoVideo_Person_Frame), frame_info
            if frame_info.frame_index == frame_index:
                if np.sum(frame_info.box_face) == 0:
                    mask_dict[info_object.identity] = None
                    logging.info('skip --> frame_index-{}, person_identity-{}, person_face_box-{}'.format(
                        frame_index, info_object.identity, frame_info.box_face))
                    continue  # invalid face box
                # expand the face box
                lft, top, rig, bot = Rectangle(frame_info.box_face).expand(0.8, 0.8).clip(0, 0, w, h).asInt()
                bgr_copy = np.copy(frame_bgr)
                # 1.
                # for i in range(len(cursor_list)):
                #     if i != n:
                #         face_mask = XPortraitHelper.getFaceRegionByLandmark(h, w, cursor_list[i].landmark)
                #         bgr_copy[face_mask > 0] = 255
                # 2.
                if len(result) > 0 and result.masks is not None and len(masks) > 0:
                    mask_box = np.zeros(shape=masks[0].shape, dtype=np.uint8)
                    l, t, r, b = frame_info.box_face
                    mask_box[t:b, l:r] = 255
                    count_nonzero = []
                    for j in range(len(masks)):
                        mm = ((masks[j] > 0) & (mask_box > 0)).astype(np.uint8)
                        count_nonzero.append(np.count_nonzero(mm))
                    index = int(np.argmax(np.array(count_nonzero, dtype=np.int32)))
                    mask_cur = masks[index]
                    bgr_copy[mask_cur == 0] = 255
                # portrait parsing
                part = bgr_copy[top:bot, lft:rig]
                part_parsing = XManager.getModules('portrait_parsing')(part)
                if with_hair is False:
                    part_mask = np.where(
                        (0 < part_parsing) & (part_parsing < 15) &
                        (part_parsing != 12) & (part_parsing != 13), 255, 0).astype(np.uint8)
                else:
                    part_mask = np.where((0 < part_parsing) & (part_parsing < 15) & (part_parsing != 12), 255, 0).astype(np.uint8)
                part_mask_single = MaskingHelper.getSingleConnectedRegion(part_mask) * 255
                mask = np.zeros(shape=(h, w), dtype=np.uint8)
                mask[top:bot, lft:rig] = part_mask_single
                mask_dict[info_object.identity] = dict(mask=mask, box=(lft, top, rig, bot))
        return mask_dict

    @staticmethod
    def getPortraitMaskingWithInfoVideo(frame_index, frame_bgr, person, frame_info, option_dict, with_hair=True):
        h, w, c = frame_bgr.shape
        assert isinstance(person, InfoVideo_Person), person
        assert isinstance(frame_info, InfoVideo_Person_Frame), frame_info
        assert person.identity in option_dict, (person.identity, str(option_dict))
        if frame_info.frame_index == frame_index:
            if option_dict[person.identity].NameEN.startswith('sticker'):
                return None
            if np.sum(frame_info.box_face) == 0:
                logging.warning('skip --> frame_index-{}, person_identity-{}, person_face_box-{}'.format(
                    frame_index, person.identity, frame_info.box_face))
                return None  # invalid face box
            lft, top, rig, bot = Rectangle(frame_info.box_face).expand(0.1, 0.8).clip(0, 0, w, h).asInt()
            bgr_copy = np.copy(frame_bgr)
            part = bgr_copy[top:bot, lft:rig]
            part_parsing = XManager.getModules('portrait_parsing')(part)
            if with_hair is False:
                part_mask = np.where(
                    (0 < part_parsing) & (part_parsing < 15) &
                    (part_parsing != 12) & (part_parsing != 13), 255, 0).astype(np.uint8)
            else:
                part_mask = np.where((0 < part_parsing) & (part_parsing < 15) & (part_parsing != 12), 255, 0).astype(np.uint8)
            part_mask_single = MaskingHelper.getSingleConnectedRegion(part_mask) * 255
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            mask[top:bot, lft:rig] = part_mask_single
            return dict(mask=mask, box=(lft, top, rig, bot))
        return None

    @staticmethod
    def workOnSelectedMask(source_bgr, blured_bgr, mask=None, mask_blur_k=17, **kwargs) -> np.ndarray:
        if mask is None:
            mask = np.ones_like(source_bgr, dtype=np.uint8) * 255
        if isinstance(mask_blur_k, int) and mask_blur_k > 0:
            mask = cv2.GaussianBlur(mask, (mask_blur_k, mask_blur_k), sigmaX=mask_blur_k // 2, sigmaY=mask_blur_k // 2)
        multi = mask.astype(np.float32)[:, :, None] / 255.
        fusion = source_bgr * (1 - multi) + blured_bgr * multi
        return np.round(fusion).astype(np.uint8)
