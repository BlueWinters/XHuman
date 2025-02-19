import logging

import cv2
import numpy as np
import skimage
from .cursor import AsynchronousCursor
from ..scanning.scanning_image import InfoImage, InfoImage_Person
from ..scanning.scanning_video import InfoVideo, InfoVideo_Person, InfoVideo_Frame
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
    def getPortraitMaskingWithInfoImage(bgr, info_image: InfoImage, option_dict, with_hair=True, expand=0.5):
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
                    face_mask = XPortraitHelper.getFaceRegionByLandmark(h, w, info_image.info_person_list[i].landmark)
                    bgr_copy[face_mask > 0] = 255
            part = bgr_copy[top:bot, lft:rig]
            part_rot = GeoFunction.rotateImage(part, info_person.angle)
            part_rot_parsing = XManager.getModules('portrait_parsing')(part_rot)
            part_parsing = GeoFunction.rotateImage(part_rot_parsing, GeoFunction.rotateBack(info_person.angle))
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
    def getPortraitMaskingWithInfoVideo2(frame_index, frame_bgr, cursor_list, option_dict, with_hair=True, expand=0.5):
        h, w, c = frame_bgr.shape
        mask_dict = dict()
        for n, (person, cursor) in enumerate(cursor_list):
            assert isinstance(person, InfoVideo_Person), person
            assert isinstance(cursor, AsynchronousCursor), cursor
            frame_info = cursor.current()
            assert isinstance(frame_info, InfoVideo_Frame), frame_info
            if frame_info.frame_index == frame_index:
                if person.identity not in option_dict:
                    continue
                if option_dict[person.identity].NameEN.startswith('sticker'):
                    mask_dict[person.identity] = None
                    continue
                if np.sum(frame_info.box_face) == 0:
                    mask_dict[person.identity] = None
                    print('skip --> frame_index-{}, person_identity-{}, person_face_box-{}'.format(
                        frame_index, person.identity, frame_info.box_face))
                    continue  # invalid face box
                lft, top, rig, bot = Rectangle(frame_info.box_face).expand(expand, expand).clip(0, 0, w, h).asInt()
                bgr_copy = np.copy(frame_bgr)
                # for i in range(len(cursor_list)):
                #     if i != n:
                #         face_mask = XPortraitHelper.getFaceRegionByLandmark(h, w, cursor_list[i].landmark)
                #         bgr_copy[face_mask > 0] = 255
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
                mask_dict[person.identity] = dict(mask=mask, box=(lft, top, rig, bot))
                # cv2.imwrite(R'N:\archive\2025\0215-masking\error_video\04\parsing\{}-mask.png'.format(n), mask)
                # cv2.imwrite(R'N:\archive\2025\0215-masking\error_image\01\parsing\{}-parsing.png'.format(n), XManager.getModules('portrait_parsing').colorize(part_rot_parsing))
        return mask_dict

    @staticmethod
    def getPortraitMaskingWithInfoVideo(frame_index, frame_bgr, person, frame_info, option_dict, with_hair=True, expand=0.5):
        h, w, c = frame_bgr.shape
        assert isinstance(person, InfoVideo_Person), person
        assert isinstance(frame_info, InfoVideo_Frame), frame_info
        if frame_info.frame_index == frame_index:
            if person.identity not in option_dict:
                return
            if option_dict[person.identity].NameEN.startswith('sticker'):
                return None
            if np.sum(frame_info.box_face) == 0:
                print('skip --> frame_index-{}, person_identity-{}, person_face_box-{}'.format(
                    frame_index, person.identity, frame_info.box_face))
                return None  # invalid face box
            lft, top, rig, bot = Rectangle(frame_info.box_face).expand(expand, expand).clip(0, 0, w, h).asInt()
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
