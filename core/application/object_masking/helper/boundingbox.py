
import numpy as np
from ....geometry import Rectangle


class BoundingBox(Rectangle):
    def __init__(self, points):
        super(BoundingBox, self).__init__(points)

    @staticmethod
    def iou(a, b) -> float:
        assert isinstance(a, BoundingBox)
        assert isinstance(b, BoundingBox)
        xx1 = max(a.x_min, b.x_min)
        yy1 = max(a.y_min, b.y_min)
        xx2 = min(a.x_max, b.x_max)
        yy2 = min(a.y_max, b.y_max)
        inter_area = (max(0, xx2 - xx1 + 1) * max(0, yy2 - yy1 + 1))
        area_a = a.area()
        area_b = b.area()
        if area_a == 0 or area_b == 0:
            return 0.
        union_area = area_a + area_b - inter_area
        return float(inter_area / union_area) if union_area > 0. else 0.

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
        ious = np.clip(intersection_area / union_area, 0.0, 1.0)
        return ious.astype(np.float32)

    @staticmethod
    def distance(a, b) -> float:
        assert isinstance(a, BoundingBox)
        assert isinstance(b, BoundingBox)
        a_cx = (a.x_min + a.x_max) / 2.
        a_cy = (a.y_min + a.y_max) / 2.
        b_cx = (b.x_min + b.x_max) / 2.
        b_cy = (b.y_min + b.y_max) / 2.
        return np.linalg.norm(np.array([a_cx-b_cx, a_cy-b_cy], dtype=np.float32))

    @staticmethod
    def findBestMatch(pre_all_boxes, cur_one_box):
        iou_max = 0.
        idx_max = -1
        for n in range(len(pre_all_boxes)):
            pre_one_rect = BoundingBox(pre_all_boxes[n, :, :])
            cur_one_rect = BoundingBox(cur_one_box)
            iou = BoundingBox.iou(pre_one_rect, cur_one_rect)
            if iou > iou_max:
                iou_max = iou
                idx_max = n
        return idx_max, iou_max

    @staticmethod
    def remapBBox(box_src, box_fmt, box_cur):
        if np.count_nonzero(box_cur - box_src) == 0:
            return box_fmt
        src_lft, src_top, src_rig, src_bot = box_src
        fmt_lft, fmt_top, fmt_rig, fmt_bot = box_fmt
        Rectangle.assertInside(Rectangle(np.array(box_src, dtype=np.int32)), Rectangle(np.array(box_fmt, dtype=np.int32)))
        pix_lft = src_lft - fmt_lft
        pix_rig = fmt_rig - src_rig
        pix_top = src_top - fmt_top
        pix_bot = fmt_bot - src_bot
        src_h = src_bot - src_top
        src_w = src_rig - src_lft
        ratio_lft = float(pix_lft / src_w)
        ratio_rig = float(pix_rig / src_w)
        ratio_top = float(pix_top / src_h)
        ratio_bot = float(pix_bot / src_h)
        lft, top, rig, bot = Rectangle(np.array(box_cur, dtype=np.float32)).expand4(
            ratio_lft, ratio_top, ratio_rig, ratio_bot).toSquare().decouple()
        return lft, top, rig, bot
