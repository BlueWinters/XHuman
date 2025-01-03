
import numpy as np
from ...geometry import Rectangle


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

