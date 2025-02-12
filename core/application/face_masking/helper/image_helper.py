
import cv2
import numpy as np
import skimage
from ....base import XPortrait, XPortraitHelper
from ....geometry import Rectangle
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

    @staticmethod
    def computeIOU(boxes1, boxes2):
        lu = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # lu with shape N,M,2 ; boxes1[:,None,:2] with shape (N,1,2) boxes2 with shape(M,2)
        rd = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # rd same to lu
        intersection_wh = np.maximum(0.0, rd - lu)
        intersection_area = intersection_wh[:, :, 0] * intersection_wh[:, :, 1]  # with shape (N,M)
        boxes1_wh = np.maximum(0.0, boxes1[:, 2:] - boxes1[:, :2])
        boxes1_area = boxes1_wh[:, 0] * boxes1_wh[:, 1]  # with shape (N,)
        boxes2_wh = np.maximum(0.0, boxes2[:, 2:] - boxes2[:, :2])
        boxes2_area = boxes2_wh[:, 0] * boxes2_wh[:, 1]  # with shape (M,)
        union_area = np.maximum(boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8)  # with shape (N,M)
        iou = np.clip(intersection_area / union_area, 0.0, 1.0)
        return iou.astype(np.float32)

    @staticmethod
    def getSingleConnectedRegion(mask):
        label_image = skimage.measure.label(mask.astype(np.uint8), connectivity=2, return_num=False)
        regions = skimage.measure.regionprops(label_image)
        if not regions:
            return np.zeros_like(mask, dtype=bool)
        max_region = max(regions, key=lambda r: r.area)
        max_mask = (label_image == max_region.label).astype(np.uint8)
        return max_mask

    """
    """
    @staticmethod
    def getFaceMaskByPoints(cache, box, top_line='brow', value=255) -> np.ndarray:
        if cache.number == 1:
            return XPortraitHelper.getFaceRegion(cache, top_line=top_line, value=value)[0]
        if cache.number > 1:
            box_src = np.reshape(np.array(box, dtype=np.int32), (1, 4))
            box_cur = np.reshape(np.array(cache.box, dtype=np.int32), (-1, 4))
            iou = MaskingHelper.computeIOU(boxes1=box_src, boxes2=box_cur)  # 1,N
            selected_index = int(np.argmax(iou[0, :]))
            if iou[0, selected_index] > 0:
                return XPortraitHelper.getFaceRegion(cache, top_line=top_line, value=value)[selected_index]
        lft, top, rig, bot = box
        h = bot - top
        w = rig - lft
        mask = np.ones(shape=(h, w), dtype=np.uint8) * 255
        # mask[top:bot, lft:rig] = 255
        return mask

    @staticmethod
    def getHeadMaskByParsing(bgr, box_src, box_tar, value=255) -> np.ndarray:
        module = XManager.getModules('ultralytics')['yolo11m-seg']
        result = module(bgr, classes=[0], verbose=False)[0]
        if len(result) > 0 and result.masks is not None:
            masks = np.round(result.masks.cpu().numpy().data * 255).astype(np.uint8)  # note: C,H,W and [0,1]
            mask_resized = [cv2.resize(masks[n, :, :], bgr.shape[:2][::-1]) for n in range(len(masks))]
            mask_box = np.zeros(shape=mask_resized[0].shape, dtype=np.uint8)
            lft, top, rig, bot = box_src
            mask_box[top:bot, lft:rig] = 255
            count_nonzero = []
            for n in range(len(mask_resized)):
                mm = ((mask_resized[n] > 0) & (mask_box > 0)).astype(np.uint8)
                count_nonzero.append(np.count_nonzero(mm))
            index = int(np.argmax(np.array(count_nonzero, dtype=np.int32)))
            mask_cur = cv2.resize(masks[index, :, :], bgr.shape[:2][::-1])
            bgr_copy = np.copy(bgr)
            bgr_copy[mask_cur == 0] = 255
            # processing with masked-bgr
            lft_tar, top_tar, rig_tar, bot_tar = box_tar if box_tar is not None else box_src
            cache = XPortrait(bgr_copy[top_tar:bot_tar, lft_tar:rig_tar, :])
            parsing = cache.parsing
            mask = np.where((0 < parsing) & (parsing < 15) & (parsing != 12), value, 0).astype(np.uint8)
            mask_single = MaskingHelper.getSingleConnectedRegion(mask) * value
            return mask_single
        else:
            lft, top, rig, bot = box_tar
            cache = XPortrait(bgr[top:bot, lft:rig, :])
            return MaskingHelper.getFaceMaskByPoints(cache, box_tar)

    @staticmethod
    def getHeadMaskByParsing2(bgr, box, value=255) -> np.ndarray:
        lft, top, rig, bot = box
        cache = XPortrait(bgr[top:bot, lft:rig, :])
        if cache.number == 1:
            parsing = cache.parsing
            mask = np.where((0 < parsing) & (parsing < 15) & (parsing != 12), value, 0).astype(np.uint8)
            mask_single = MaskingHelper.getSingleConnectedRegion(mask) * value
            return mask_single
        if cache.number > 1:
            box_src = np.reshape(np.array(box, dtype=np.int32), (1, 4))
            box_cur = np.reshape(np.array(cache.box, dtype=np.int32), (-1, 4))
            iou = MaskingHelper.computeIOU(boxes1=box_src, boxes2=box_cur)  # 1,N
            part_copy = np.copy(cache.bgr)
            selected_index = int(np.argmax(iou[0, :]))
            for n in range(cache.number):
                if n != selected_index:
                    # l, t, r, b = cache.box[n, :]
                    # part_copy[t:b, l:r, :] = 255
                    part_mask = XPortraitHelper.getFaceRegion(cache, index=n, top_line='brow', value=255)[0]
                    part_copy[part_mask > 0] = 255
            parsing = XPortrait(part_copy).parsing
            mask = np.where((0 < parsing) & (parsing < 15) & (parsing != 12), value, 0).astype(np.uint8)
            mask_single = MaskingHelper.getSingleConnectedRegion(mask) * value
            return mask_single
        # note: detect face fail
        parsing = cache.parsing
        mask = np.where((0 < parsing) & (parsing < 15) & (parsing != 12), value, 0).astype(np.uint8)
        mask_single = MaskingHelper.getSingleConnectedRegion(mask) * value
        return mask_single

    @staticmethod
    def getMaskFromBox(h, w, box, ratio, value=255) -> np.ndarray:
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
    def getSelectedMask(bgr, box, align_type) -> (np.ndarray, tuple):
        h, w = bgr.shape[:2]
        lft, top, rig, bot = box
        if align_type == 'face':
            # lft, top, rig, bot = box
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            lft, top, rig, bot = Rectangle(box).expand(0.2, 0.2).clip(0, 0, w, h).asInt()
            mask[top:bot, lft:rig] = MaskingHelper.getFaceMaskByPoints(XPortrait(bgr[top:bot, lft:rig, :]), box)
        elif align_type == 'head':
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            # lft, top, rig, bot = Rectangle(box).toSquare().expand(0.2, 0.2).clip(0, 0, w, h).asInt()
            mask[top:bot, lft:rig] = MaskingHelper.getHeadMaskByParsing(bgr, (lft, top, rig, bot), None)
            # mask[box[3]:, :] = 0
        else:
            mask = MaskingHelper.getMaskFromBox(h, w, box, ratio=0.)
        return mask, (lft, top, rig, bot)

    """
    """
    @staticmethod
    def workOnSelectedMask(source_bgr, blured_bgr, kernel=17, mask=None, **kwargs) -> np.ndarray:
        if mask is None:
            mask = np.ones_like(source_bgr, dtype=np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (kernel, kernel), sigmaX=kernel // 2, sigmaY=kernel // 2)
        multi = mask.astype(np.float32)[:, :, None] / 255.
        fusion = source_bgr * (1 - multi) + blured_bgr * multi
        return np.round(fusion).astype(np.uint8)
