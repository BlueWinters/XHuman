
import logging
import cv2
import os
import json
import numpy as np
import scipy
import sympy
from .libtennis_minicourt import LibTennisMiniCourt
from ... import XManager


class LibTennisCourtTracker:
    """
    """
    @staticmethod
    def packageAsStream(frames_stream):
        if isinstance(frames_stream, list):
            assert isinstance(frames_stream[0], np.ndarray), 'frames_stream should be list of numpy.ndarray'
            return frames_stream
        if isinstance(frames_stream, str):
            assert os.path.exists(frames_stream), 'frames_stream should be path to video file'
            from core.utils.video import XVideoReaderOpenCVCache
            return XVideoReaderOpenCVCache(frames_stream)
        raise TypeError('frames_stream should be path to video file or list of numpy.ndarray')

    @staticmethod
    def visualize(canvas, points):
        for j in range(len(points)):
            if points[j][0] is not None:
                canvas = cv2.circle(
                    canvas, (int(points[j][0]), int(points[j][1])),
                    radius=0, color=(0, 0, 255), thickness=10)
        return canvas

    @staticmethod
    def visualizeAsVideo(frames_stream, writer, result):
        assert len(frames_stream) == len(result)
        for num, (frame, points) in enumerate(zip(frames_stream, result)):
            for j in range(len(points)):
                if points[j][0] is not None:
                    frame = cv2.circle(
                        frame, (int(points[j][0]), int(points[j][1])),
                        radius=0, color=(0, 0, 255), thickness=10)
            writer.write(frame)

    EngineConfig = {
        'type': 'torch',
        'device': 'cuda:0',
        'parameters': 'thirdparty/tennis/tennis_court_tracking.ts',
    }

    """
    """
    def __init__(self, *args, **kwargs):
        self.engine = XManager.createEngine(self.EngineConfig)
        self.device = XManager.CommonDevice
        self.height = 360
        self.width = 640
        self.court = LibTennisMiniCourt()

    def __del__(self):
        # logging.warning('delete module {}'.format(self.__class__.__name__))
        pass

    def initialize(self, *args, **kwargs):
        self.engine.initialize(*args, **kwargs)

    @staticmethod
    def resizeAndPadding(image, d_h, d_w):
        s_h, s_w, _ = image.shape
        src_ratio = float(s_h / s_w)
        dst_ratio = float(d_h / d_w)
        if src_ratio > dst_ratio:
            r_h, r_w = d_h, int(round(float(s_w / s_h) * d_h))
            resized = cv2.resize(image, (r_w, r_h))
            lp_w = (d_w - r_w) // 2
            rp_w = d_w - r_w - lp_w
            resized = np.pad(resized, ((0, 0), (lp_w, rp_w), (0, 0)), constant_values=255, mode='constant')
            padding = (0, 0, lp_w, rp_w)
        else:
            r_h, r_w = int(round(float(s_h / s_w) * d_w)), d_w
            resized = cv2.resize(image, (r_w, r_h))
            tp_h = (d_h - r_h) // 2
            bp_h = d_h - r_h - tp_h
            resized = np.pad(resized, ((tp_h, bp_h), (0, 0), (0, 0)), constant_values=255, mode='constant')
            padding = (tp_h, bp_h, 0, 0)
        return resized, padding

    def clipAndResizeB(self, bgr, points, padding):
        src_h, src_w = bgr.shape[:2]
        fmt_h, fmt_w = self.height, self.width
        tp, bp, lp, rp = padding
        rescale_w = (fmt_w - lp - rp) / src_w
        rescale_h = (fmt_h - tp - bp) / src_h
        # xxx = np.clip(xx, 0, src_w).round().astype(np.int32)
        # yyy = np.clip(yy, 0, src_h).round().astype(np.int32)
        # return np.stack([xxx, yyy], axis=1)
        result = []
        for x, y in points:
            if x is None or y is None:
                continue
            xx = (x - lp) / rescale_w
            yy = (y - tp) / rescale_h
            xxx = min(max(0, xx), src_w)
            yyy = min(max(0, yy), src_h)
            result.append((xxx, yyy))
        return result

    def format(self, frame):
        assert isinstance(frame, np.ndarray), type(frame)
        resized, padding = self.resizeAndPadding(frame, self.height, self.width)
        batch_input = cv2.dnn.blobFromImage(resized, scalefactor=1 / 255.0, swapRB=False)  # (1, 3, 360, 640)
        return batch_input, padding

    @staticmethod
    def heatmap2Point(heatmap, low_thresh=155, min_radius=10, max_radius=30):
        x, y = None, None
        ret, heatmap = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(
            heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2,
            minRadius=min_radius, maxRadius=max_radius)
        if circles is not None:
            x = circles[0][0][0]
            y = circles[0][0][1]
        return x, y

    def postprocess(self, frame, pred, padding, use_refine_kps=True):
        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            x, y = LibTennisCourtTracker.heatmap2Point(heatmap, low_thresh=170, max_radius=25)
            if use_refine_kps and kps_num not in [8, 12, 9] and x and y:
                x, y = LibTennisCourtTracker.refinePoints(frame, int(y), int(x))
            points.append((x, y))
        matrix_trans = self.getTransMatrix(points)
        if matrix_trans is not None:
            points = cv2.perspectiveTransform(self.court.key_points, matrix_trans)
            points = [np.squeeze(x) for x in points]
        return self.clipAndResizeB(frame, points, padding)

    @staticmethod
    def detectLines(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]
        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30)
        lines = np.squeeze(lines)
        if len(lines.shape) > 0:
            if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
                lines = [lines]
        else:
            lines = []
        return lines

    @staticmethod
    def mergeLines(lines):
        lines = sorted(lines, key=lambda item: item[0])
        mask = [True] * len(lines)
        new_lines = []
        for i, line in enumerate(lines):
            if mask[i]:
                for j, s_line in enumerate(lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        dist1 = scipy.spatial.distance.euclidean((x1, y1), (x3, y3))
                        dist2 = scipy.spatial.distance.euclidean((x2, y2), (x4, y4))
                        if dist1 < 20 and dist2 < 20:
                            line = np.array([
                                int((x1 + x3) / 2), int((y1 + y3) / 2),
                                int((x2 + x4) / 2), int((y2 + y4) / 2)
                            ], dtype=np.int32)
                            mask[i + j + 1] = False
                new_lines.append(line)
        return new_lines

    @staticmethod
    def refinePoints(img, x_ct, y_ct, crop_size=40):
        refined_x_ct, refined_y_ct = x_ct, y_ct
        img_height, img_width = img.shape[:2]
        x_min = max(x_ct - crop_size, 0)
        x_max = min(img_height, x_ct + crop_size)
        y_min = max(y_ct - crop_size, 0)
        y_max = min(img_width, y_ct + crop_size)
        img_crop = img[x_min:x_max, y_min:y_max]
        lines = LibTennisCourtTracker.detectLines(img_crop)
        if len(lines) > 1:
            lines = LibTennisCourtTracker.mergeLines(lines)
            if len(lines) == 2:
                inters = LibTennisCourtTracker.getLineIntersection(lines[0], lines[1])
                if inters:
                    new_x_ct = int(inters[1])
                    new_y_ct = int(inters[0])
                    valid_x = bool(0 < new_x_ct < img_crop.shape[0])
                    valid_y = bool(0 < new_y_ct < img_crop.shape[1])
                    if valid_x and valid_y:
                        refined_x_ct = x_min + new_x_ct
                        refined_y_ct = y_min + new_y_ct
        return refined_y_ct, refined_x_ct

    @staticmethod
    def getLineIntersection(line1, line2):
        l1 = sympy.Line((line1[0], line1[1]), (line1[2], line1[3]))
        l2 = sympy.Line((line2[0], line2[1]), (line2[2], line2[3]))
        intersection = l1.intersection(l2)
        point = None
        if len(intersection) > 0:
            if isinstance(intersection[0], sympy.geometry.point.Point2D):
                point = intersection[0].coordinates
        return point

    def getTransMatrix(self, points):
        matrix_trans = None
        dist_max = np.Inf
        for conf_ind in range(1, 13):
            conf = self.court.court_conf[conf_ind]
            inds = self.court.court_conf_ind[conf_ind]
            inters = [points[inds[0]], points[inds[1]], points[inds[2]], points[inds[3]]]
            if not any([None in x for x in inters]):
                matrix, _ = cv2.findHomography(np.float32(conf), np.float32(inters), method=0)
                trans_kps = cv2.perspectiveTransform(self.court.key_points, matrix)
                dists = []
                for i in range(12):
                    if i not in inds and points[i][0] is not None:
                        dists.append(scipy.spatial.distance.euclidean(points[i], trans_kps[i, 0, :]))
                dist_median = np.mean(dists)
                if dist_median < dist_max:
                    matrix_trans = matrix
                    dist_max = dist_median
        return matrix_trans

    def inference(self, frames, **kwargs):
        result = []
        for n, frame in enumerate(frames):
            batch_input, padding = self.format(frame)
            prediction = self.engine.inference(batch_input)
            result.append(self.postprocess(frame, prediction[0, :, :, :], padding))
        return result

    """
    """
    def extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        inference_kwargs = dict()
        return targets, inference_kwargs

    def returnResult(self, frames_stream, output, targets):
        def _formatResult(target):
            if target == 'source':
                return output
            if target == 'mean':
                array = np.array(output, np.float32)
                mean_points = np.nanmean(array, axis=0)
                matrix_trans = self.getTransMatrix(mean_points)
                if matrix_trans is not None:
                    matrix_trans_invert = cv2.invert(matrix_trans)[1]
                    return mean_points, matrix_trans_invert
                return mean_points, None
            if target == 'visual':
                return [self.visualize(frame, output) for frame in frames_stream]
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, path_or_frames, *args, **kwargs):
        frames_stream = self.packageAsStream(path_or_frames)
        targets, inference_kwargs = self.extractArgs(*args, **kwargs)
        output = self.inference(frames_stream, **inference_kwargs)
        return self.returnResult(frames_stream, output, targets)
