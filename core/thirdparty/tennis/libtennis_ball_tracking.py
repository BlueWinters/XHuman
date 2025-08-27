
import logging
import cv2
import os
import json
import numpy as np
from scipy.spatial import distance
from ... import XManager


class LibTennisBallTracking:
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
    def visualize(frames_stream, ball_track, writer, trace=7):
        for num, frame in enumerate(frames_stream):
            for i in range(trace):
                if num - i > 0:
                    if ball_track[num - i][0]:
                        x = int(ball_track[num - i][0])
                        y = int(ball_track[num - i][1])
                        frame = cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness=10 - i)
                    else:
                        break
            writer.write(frame)

    EngineConfig = {
        'type': 'torch',
        'device': 'cuda:0',
        'parameters': 'thirdparty/tennis/tennis_ball_tracking.ts',
    }

    """
    """
    def __init__(self, *args, **kwargs):
        self.engine = XManager.createEngine(self.EngineConfig)
        self.device = XManager.CommonDevice
        self.height = 360
        self.width = 640

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

    def format(self, frames):
        assert len(frames) == 3, len(frames)
        padding = None
        normal_list = []
        for n, bgr in enumerate(frames):
            resized, padding = self.resizeAndPadding(bgr, self.height, self.width)
            # format_image = resized.astype(np.float32) / 255.0
            # normal_list.append(np.transpose(format_image, (2, 0, 1))[None, ...])
            normal_list.append(cv2.dnn.blobFromImage(resized, scalefactor=1/255.0, swapRB=False))
        batch_input = np.concatenate(normal_list, axis=1)  # (1, 9, 360, 640)
        return batch_input, padding

    def postprocess(self, feature_map, frame, padding):
        feature_map *= 255
        feature_map = feature_map.reshape((360, 640))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(
            heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
        if circles is not None:
            if len(circles) == 1:
                x = circles[0][0][0]
                y = circles[0][0][1]
                return self.clipAndResizeB(frame, x, y, padding)
        return None, None

    def clipAndResizeB(self, bgr, x, y, padding) -> tuple:
        src_h, src_w = bgr.shape[:2]
        fmt_h, fmt_w = self.height, self.width
        tp, bp, lp, rp = padding
        rescale_w = (fmt_w - lp - rp) / src_w
        rescale_h = (fmt_h - tp - bp) / src_h
        xx = (x - lp) / rescale_w
        yy = (y - tp) / rescale_h
        xxx = min(max(0, xx), src_w)
        yyy = min(max(0, yy), src_h)
        return xxx, yyy

    @staticmethod
    def remove_outliers(ball_track, dists, max_dist=100):
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        for i in outliers:
            if 0 < i < len(ball_track) - 1:
                if (dists[i + 1] > max_dist) | (dists[i + 1] == -1):
                    ball_track[i] = (None, None)
                    outliers.remove(i)
                elif dists[i - 1] == -1:
                    ball_track[i - 1] = (None, None)
        return ball_track

    def inference(self, frames_stream, *args, **kwargs):
        assert len(frames_stream) >= 3, len(frames_stream)
        dists = [-1] * 2
        ball_track = [(None, None)] * 2
        for n in range(2, len(frames_stream)):
            frames3 = [frames_stream[n], frames_stream[n - 1], frames_stream[n - 2]]
            batch_input, padding = self.format(frames3)
            output = self.engine.inference(batch_input)
            x, y = self.postprocess(np.argmax(output, axis=1), frames3[-1], padding)
            ball_track.append((x, y))
            if ball_track[-1][0] is not None and ball_track[-2][0] is not None:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1
            dists.append(dist)
        ball_track = self.remove_outliers(ball_track, dists)
        return ball_track

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
            if target == 'json':
                trace = [dict(index=n, x=pos[0], y=pos[1]) for n, pos in enumerate(output)]
                return json.dumps(trace, indent=4)
            if target == 'visual':
                pass
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


