
import cv2
import numpy as np
import logging
import json
from itertools import product as product
from math import ceil
from .. import XManager



class PriorBox:
    def __init__(self, img_size, min_size, steps, clip=False):
        super(PriorBox, self).__init__()
        self.img_size = img_size
        self.min_size = min_size
        self.steps = steps
        self.clip = clip
        self.feature_maps = [[ceil(self.img_size[0] / step), ceil(self.img_size[1] / step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_size[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.img_size[1]
                    s_ky = min_size / self.img_size[0]
                    dense_cx = [x * self.steps[k] / self.img_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.img_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = np.reshape(np.array(anchors), (-1, 4))
        if self.clip:
            output = np.clip(output, max=1, min=0)
        return output


class LibFaceDetection:
    @staticmethod
    def serialize_to_json(scores, boxes, points):
        data = []
        for n, (s, b, p) in enumerate(zip(scores, boxes, points)):
            data.append({
                'score': float(s),
                'box': np.reshape(b, (4,)).tolist(),
                'points': np.reshape(p, (5, 2)).tolist(),
            })
        return json.dumps(data, indent=4)

    def unserialize_from_json(self, content:str, split=True):
        data = json.loads(content)
        object_list = list()
        for each in data:
            object_list.append({
                'score': float(each['score']),
                'box': np.array(each['box'], dtype=np.int32).reshape(4,),
                'points': np.array(each['points'], dtype=np.int32).reshape(5, 2),
            })
        return object_list

    @staticmethod
    def print_detection(scores, boxes, points):
        print('detect faces: {}'.format(len(scores)))
        for n, (s, b, p) in enumerate(zip(scores, boxes, points)):
            print('#{:3d}: score {}% box {} points{}'.format(
                n,
                int(round(s, 2)*100),
                b.reshape(-1).tolist(),
                p.reshape(-1).tolist())
            )

    @staticmethod
    def visual_targets_cv2(bgr, scores, boxes, points, options=(True,True)):
        visual_score, visual_points = options
        for s, b, p in zip(scores, boxes, points):
            # confidence
            if visual_score:
                text = '{}%'.format(int(round(s, 2)*100))
                cx, cy = b[0], b[1] + 12
                cv2.putText(bgr, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # bounding box
            cv2.rectangle(bgr, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
            # landmarks
            if visual_points:
                radius = 1
                cv2.circle(bgr, (p[0], p[1]), radius, (0, 0, 255), 4)
                cv2.circle(bgr, (p[2], p[3]), radius, (0, 255, 255), 4)
                cv2.circle(bgr, (p[4], p[5]), radius, (255, 0, 255), 4)
                cv2.circle(bgr, (p[6], p[7]), radius, (0, 255, 0), 4)
                cv2.circle(bgr, (p[8], p[9]), radius, (255, 0, 0), 4)
        return bgr

    @staticmethod
    def visual_targets_plt( bgr, scores, boxes, points, options=(True,True)):
        visual_score, visual_points = options
        import matplotlib.pyplot as plt
        rgb = bgr[:, :, ::-1]
        h, w, c = rgb.shape
        # figure
        figure = plt.figure()
        axs = figure.gca()
        axs.imshow(rgb)

        for s, b, p in zip(scores, boxes, points):
            # confidence
            text = '{}%'.format(int(round(s, 2)*100))
            font_size = max(h, w) // 100.
            bbox = dict(facecolor='black', boxstyle='square', pad=0.1, alpha=0.5)
            cx, cy = b[0]+font_size, b[1]+font_size/2.
            if visual_score:
                axs.text(cx, cy, text, bbox=bbox, ha='center', va='center', fontsize=font_size, color='white')
            # bounding box
            ww, hh = b[2] - b[0], b[3] - b[1]
            axs.add_patch(plt.Rectangle(
                xy=(b[0], b[1]), width=ww, height=hh, fill=False, alpha=0.5, edgecolor='blue', linewidth=1.5))
            # landmarks
            if visual_points:
                size = font_size / 5.
                axs.plot(p[0], p[1], 'o', color=(0, 0, 1), markersize=size)
                axs.plot(p[2], p[3], 'o', color=(0, 1, 0), markersize=size)
                axs.plot(p[4], p[5], 'o', color=(1, 0, 0), markersize=size)
                axs.plot(p[6], p[7], 'o', color=(0, 1, 1), markersize=size)
                axs.plot(p[8], p[9], 'o', color=(1, 0, 1), markersize=size)
        # show
        plt.axis('off')
        plt.tight_layout()
        plt.get_current_fig_manager().window.showMaximized()
        plt.show()

    @staticmethod
    def getResources():
        return [
            LibFaceDetection.EngineConfig['parameters'],
        ]

    """
    """
    EngineConfig = {
        'type': 'torch',
        'device': 'cuda:0',
        'parameters': 'base/face_detection.ts'
    }


    """
    """
    def __init__(self, *args, **kwargs):
        self.engine = XManager.createEngine(self.EngineConfig)
        # model config parameters
        self.single_scale = True
        self.rgb_mean = (104, 117, 123)
        self.square_box = True
        self.nms_threshold = 0.4
        self.score_threshold = 0.5
        self.top_k = 5000
        self.img_h, self.img_w = 640, 640
        self.min_size = [[16, 32], [64, 128], [256, 512]]
        self.variance = [0.1, 0.2]
        self.steps = [8, 16, 32]
        self.clip = False
        # for visual
        self.visual_score = False
        self.visual_points = False
        self.visual_handle = self.visual_targets_cv2  # self.visual_targets_plt

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    """
    """
    def initialize(self, *args, **kwargs):
        self.engine.initialize(*args, **kwargs)

    """
    """
    def inference(self, bgr, single_scale:bool=True):
        return self.inference_single_scale(bgr, self.img_h, self.img_w) \
            if single_scale else self.inference_multi_scale(bgr)

    def resize_and_padding(self, image, dH, dW):
        sH, sW, _ = image.shape
        src_ratio = float(sH / sW)
        dst_ratio = float(dH / dW)
        if src_ratio > dst_ratio:
            rH, rW = dH, int(round(float(sW / sH) * dH))
            resized = cv2.resize(image, (rW, rH))
            lpW = (dW - rW) // 2
            rpW = dW - rW - lpW
            resized = np.pad(resized, ((0, 0), (lpW, rpW), (0, 0)), constant_values=255, mode='constant')
            padding = (0, 0, lpW, rpW)
        else:
            rH, rW = int(round(float(sH / sW) * dW)), dW
            resized = cv2.resize(image, (rW, rH))
            tpH = (dH - rH) // 2
            bpH = dH - rH - tpH
            resized = np.pad(resized, ((tpH, bpH), (0, 0), (0, 0)), constant_values=255, mode='constant')
            padding = (tpH, bpH, 0, 0)
        return resized, padding

    def normalize_input(self, image):
        image -= self.rgb_mean
        batch_image_np = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)
        return batch_image_np

    def non_maximum_suppression(self, scores, boxes, nms_threshold):
        """Pure Python NMS baseline."""
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_threshold)[0]
            order = order[inds + 1]
        return keep

    def transform_boxes(self, src_box, r:float=0.02):
        cx, cy = (src_box[:, 0] + src_box[:, 2]) // 2, (src_box[:, 1] + src_box[:, 3]) // 2
        h2, w2 = np.abs(src_box[:, 1] - src_box[:, 3]), np.abs(src_box[:, 0] - src_box[:, 2])
        l = (h2 + w2) / 4.
        x_min, y_min, x_max, y_max = [cx - l, cy - l, cx + l, cy + l]
        diff_h = src_box[:, 3] - y_max
        y_min += diff_h
        y_max += diff_h
        x_min -= 2 * l * r
        y_min -= 2 * l * r
        x_max += 2 * l * r
        y_max += 2 * l * r
        boxes = np.stack((x_min, y_min, x_max, y_max), axis=1)
        return boxes

    def decode_box(self, loc, priors, variances):
        # ref to: box_utils.py:decode
        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def decode_landmark(self, pre, priors, variances):
        landmark = np.concatenate(
            (priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
             priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
             priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
             priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
             priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
             ), axis=1)
        return landmark

    def inference_single_scale(self, bgr, resize_h, resize_w):
        src_h, src_w, c = bgr.shape
        resized, padding = self.resize_and_padding(bgr, resize_h, resize_w)

        normalized = self.normalize_input(resized.astype(np.float32))
        boxes, scores, points = self.engine.inference(normalized)  # forward pass

        priorbox = PriorBox(
            img_size=(resize_h, resize_w), min_size=self.min_size,
            steps=self.steps, clip=self.clip)
        priors = priorbox.forward()

        # 1.scores
        scores = np.squeeze(scores, axis=0)[:, 1]
        # 2.boxes
        boxes = self.decode_box(np.squeeze(boxes, axis=0), priors, self.variance)
        boxes[:, 0::2] *= resize_w
        boxes[:, 1::2] *= resize_h
        # 3.points
        points = self.decode_landmark(np.squeeze(points, axis=0), priors, self.variance)
        points[:, 0::2] *= resize_w
        points[:, 1::2] *= resize_h

        # ignore low scores
        keep = np.where(scores > self.score_threshold)[0]
        scores = scores[keep]
        boxes = boxes[keep]
        points = points[keep]

        # Top-K
        top_k = scores.argsort()[::-1][:self.top_k]
        scores = scores[top_k]
        boxes = boxes[top_k]
        points = points[top_k]

        # NMS
        keep = self.non_maximum_suppression(scores, boxes, self.nms_threshold)
        scores = scores[keep]
        boxes = boxes[keep]
        points = points[keep]

        # for square box
        if self.square_box or False:
            boxes = self.transform_boxes(boxes)

        # calculate the coordinates
        tp, bp, lp, rp = padding
        rescale_w = (resize_w - lp - rp) / src_w
        rescale_h = (resize_h - tp - bp) / src_h
        boxes[:, 0::2] = np.clip((boxes[:, 0::2] - lp) / rescale_w, 0, src_w)
        boxes[:, 1::2] = np.clip((boxes[:, 1::2] - tp) / rescale_h, 0, src_h)
        points[:, 0::2] = np.clip((points[:, 0::2] - lp) / rescale_w, 0, src_w)
        points[:, 1::2] = np.clip((points[:, 1::2] - tp) / rescale_h, 0, src_h)

        boxes = np.round(boxes).astype(np.int32)
        points = np.round(points).astype(np.int32)
        return (scores, boxes, points)

    def inference_multi_scale(self, bgr):
        src_h, src_w, c = bgr.shape
        scores_collect, boxes_collect, points_collect = [], [], []
        max_side = max(src_h, src_w)
        resize_h, resize_w = self.img_h, self.img_w
        while resize_h/2 < max_side or resize_w/2 < max_side:
            # print(resize_h, resize_w, max_side)
            scores, boxes, points = self.inference_single_scale(bgr, resize_h=resize_h, resize_w=resize_w)
            scores_collect.append(scores)
            boxes_collect.append(boxes)
            points_collect.append(points)
            resize_h, resize_w = resize_h * 2, resize_w * 2
        # NMS
        scores = np.concatenate(scores_collect, axis=0)
        boxes = np.concatenate(boxes_collect, axis=0)
        points = np.concatenate(points_collect, axis=0)
        keep = self.non_maximum_suppression(scores, boxes, self.nms_threshold)
        scores = scores[keep]
        boxes = boxes[keep]
        points = points[keep]
        return (scores, boxes, points)

    """
    """
    def _extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        single_scale = bool(kwargs.pop('single_scale', True))
        return targets, single_scale

    def _returnResult(self, bgr, output, targets):
        def _formatResult(target):
            if target == 'source': return output
            if target == 'json':
                scores, boxes, points = output
                data = list()
                for s, b, p in zip(scores, boxes, points):
                    data.append(dict(scores=s.tolist(),
                        boxes=b.tolist(), points=p.tolist()))
                return json.dumps(data, indent=4)
            if target == 'visual-cv2':
                scores, boxes, points = output
                return self.visual_targets_cv2(bgr, scores, boxes, points)
            if target == 'visual-plt':
                scores, boxes, points = output
                return self.visual_targets_plt(bgr, scores, boxes, points)
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, bgr, *args, **kwargs):
        targets, single_scale = self._extractArgs(*args, **kwargs)
        output = self.inference(bgr, single_scale)
        return self._returnResult(bgr, output, targets)
