
import logging
import os
import cv2
import onnxruntime
import numpy as np
from .. import XManager


class LibHeadDetectionInterface:
    """
    """
    @staticmethod
    def getResources():
        raise NotImplementedError

    @staticmethod
    def visual(canvas_bgr, boxes, color=(0, 0, 255)):
        for box in boxes:
            lft = int(box[0])
            top = int(box[1])
            rig = int(box[2])
            bot = int(box[3])
            cv2.rectangle(canvas_bgr, (lft, top), (rig, bot), color=color, thickness=2)
        return canvas_bgr

    @staticmethod
    def detectVideo(path_video_in, path_video_out, name='large'):
        from core.utils import XVideoReader, XVideoWriter
        module = XManager.getModules('head_detector_{}'.format(name))
        reader = XVideoReader(path_video_in)
        config = reader.desc(True)
        config['backend'] = 'ffmpeg'
        writer = XVideoWriter(config)
        writer.open(path_video_out)
        for n, frame in enumerate(reader):
            scores, boxes = module(frame)
            canvas = LibHeadDetectionInterface.visual(np.copy(frame), boxes)
            writer.write(canvas)
        writer.release()

    """
    """
    def __init__(self, *args, **kwargs):
        self.score_threshold = 0.5
        self.nms_threshold = 0.6
        self.top_k = 64
        self.root = None
        # session
        self.onnx_session = None
        self.input_shapes = None
        self.input_names = None
        self.output_names = None
        self.input_height = None
        self.input_width = None

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def getConfig(self):
        raise NotImplementedError

    def initialize(self, *args, **kwargs):
        if self.onnx_session is None:
            root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
            config = self.getConfig()
            path = '{}/{}'.format(root, config['checkpoint'])
            assert os.path.exists(path), path
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self.onnx_session = onnxruntime.InferenceSession(
                path, sess_options=session_option, providers=config['providers'])
            self.input_shapes = [each.shape for each in self.onnx_session.get_inputs()]
            self.input_names = [each.name for each in self.onnx_session.get_inputs()]
            self.output_names = [each.name for each in self.onnx_session.get_outputs()]
            self.input_height = self.input_shapes[0][2]
            self.input_width = self.input_shapes[0][3]

    @staticmethod
    def formatResult(bgr, format_size, outputs, padding):
        src_h, src_w = bgr.shape[0], bgr.shape[1]
        fmt_h, fmt_w = format_size
        if isinstance(outputs, list) and len(outputs) == 2:
            scores, boxes = outputs[0], outputs[1][:, 2:]
            boxes_new = np.copy(boxes)
            boxes[:, 0::2] = boxes_new[:, 1::2]
            boxes[:, 1::2] = boxes_new[:, 0::2]
        else:
            boxes, scores = outputs[0][:, :4], outputs[0][:, 4:5]
            boxes[:, 0::2] *= fmt_w
            boxes[:, 1::2] *= fmt_h
        tp, bp, lp, rp = padding
        rescale_w = (fmt_w - lp - rp) / src_w
        rescale_h = (fmt_h - tp - bp) / src_h
        boxes[:, 0::2] = np.clip((boxes[:, 0::2] - lp) / rescale_w, 0, src_w)
        boxes[:, 1::2] = np.clip((boxes[:, 1::2] - tp) / rescale_h, 0, src_h)
        return scores, boxes

    @staticmethod
    def formatSizeWithPaddingForward(bgr, dst_h, dst_w, padding_value=255) -> (np.ndarray, tuple):
        # base on long side, padding to target size
        src_h, src_w, _ = bgr.shape
        src_ratio = float(src_h / src_w)
        dst_ratio = float(dst_h / dst_w)
        if src_ratio > dst_ratio:
            rsz_h, rsz_w = dst_h, int(round(float(src_w / src_h) * dst_h))
            resized = cv2.resize(bgr, (max(1, rsz_w), max(1, rsz_h)))
            lp = (dst_w - rsz_w) // 2
            rp = dst_w - rsz_w - lp
            resized = np.pad(resized, ((0, 0), (lp, rp), (0, 0)), constant_values=padding_value, mode='constant')
            padding = (0, 0, lp, rp)
        else:
            rsz_h, rsz_w = int(round(float(src_h / src_w) * dst_w)), dst_w
            resized = cv2.resize(bgr, (max(1, rsz_w), max(1, rsz_h)))
            tp = (dst_h - rsz_h) // 2
            bp = dst_h - rsz_h - tp
            resized = np.pad(resized, ((tp, bp), (0, 0), (0, 0)), constant_values=255, mode='constant')
            padding = (tp, bp, 0, 0)
        return resized, padding

    @staticmethod
    def doNonMaximumSuppression(scores, boxes, nms_threshold):
        # pure python NMS baseline
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]
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
            indexes = np.where(ovr <= nms_threshold)[0]
            order = order[indexes + 1]
        return keep

    def preprocess(self, image: np.ndarray, swap=(2, 0, 1)):
        # Normalization + BGR->RGB
        resized_image, padding = self.formatSizeWithPaddingForward(image, self.input_height, self.input_width)
        fmt_h, fmt_w = resized_image.shape[0], resized_image.shape[1]
        resized_image = np.divide(resized_image, 255.0)
        resized_image = resized_image[..., ::-1]
        resized_image = resized_image.transpose(swap)
        resized_image = np.ascontiguousarray(resized_image, dtype=np.float32)
        return resized_image, padding, (fmt_h, fmt_w)

    def inference(self, bgr: np.ndarray, **kwargs):
        # PreProcess
        resized_image, padding, format_size = self.preprocess(bgr)
        # Inference
        inference_image = np.asarray([resized_image], dtype=np.float32)
        outputs = self.onnx_session.run(
            self.output_names, {input_name: inference_image for input_name in self.input_names}, )
        scores, boxes = self.formatResult(bgr, format_size, outputs, padding)
        # PostProcess
        scores, boxes = self.postprocess(scores, boxes)
        return scores, boxes

    def postprocess(self, scores, boxes):
        scores = np.reshape(scores, (-1,))
        keep = np.where(scores > self.score_threshold)[0]
        scores = scores[keep]
        boxes = boxes[keep]
        # Top-K
        top_k = scores.argsort()[::-1][:self.top_k]
        scores = scores[top_k]
        boxes = boxes[top_k]
        # NMS
        keep = self.doNonMaximumSuppression(scores, boxes, self.nms_threshold)
        scores = scores[keep]
        boxes = boxes[keep]
        return scores, boxes

    """
    """
    def extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        return targets, dict()

    def returnResult(self, bgr, output, targets):
        def _formatResult(target):
            if target == 'source':
                return output
            if target == 'visual':
                scores, boxes = output[:3]
                return self.visual(np.copy(bgr), scores, boxes)
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, bgr, *args, **kwargs):
        targets, inference_kwargs = self.extractArgs(*args, **kwargs)
        output = self.inference(bgr, **inference_kwargs)
        return self.returnResult(bgr, output, targets)


class LibHeadDetectionTiny(LibHeadDetectionInterface):
    """
    """
    Config = {
        'checkpoint': 'thirdparty/yolov7_tiny_head_0.768_post_480x640.onnx',
        'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
    }

    @staticmethod
    def getResources():
        return [
            LibHeadDetectionTiny.Config['checkpoint'],
        ]

    def __init__(self, *args, **kwargs):
        super(LibHeadDetectionTiny, self).__init__()

    def getConfig(self):
        return self.Config


class LibHeadDetectionLarge(LibHeadDetectionInterface):
    """
    """
    Config = {
        'checkpoint': 'thirdparty/yolov4_headdetection_480x640_post.onnx',
        'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
    }

    @staticmethod
    def getResources():
        return [
            LibHeadDetectionTiny.Config['checkpoint'],
        ]

    def __init__(self, *args, **kwargs):
        super(LibHeadDetectionLarge, self).__init__()

    def getConfig(self):
        return self.Config
