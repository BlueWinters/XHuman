
import logging
import os
import cv2
import onnxruntime
import numpy as np
import copy
from ... import XManager


class LibHeadDetectionInterface:
    """
    """
    @staticmethod
    def constructUITab():
        import gradio
        with gradio.Tab('Head-Detection'):
            with gradio.Row():
                # checkpoints = [LibHeadDetectionLarge.Config['checkpoint'], LibHeadDetectionTiny.Config['checkpoint']]
                # model_list = [os.path.split(ckpt)[1] for ckpt in checkpoints]
                model_list = ['head_detection_large', 'head_detection_tiny']
                dropdown_models = gradio.Dropdown(model_list, label='model')
                annotated_types = ['box', 'box_corner']
                dropdown_annotator = gradio.Dropdown(annotated_types, annotated_types[0], label='annotator')
            with gradio.Tab('image'):
                with gradio.Row():
                    with gradio.Column():
                        input_image = gradio.Image(label='input image', height=512)
                        with gradio.Row():
                            input_score_threshold = gradio.Number(0.5, interactive=True, label='score-threshold', minimum=0)
                            input_number_index = gradio.Number(0, interactive=True, label='frame-index', minimum=0)
                        with gradio.Group():
                            with gradio.Row():
                                input_image_path1 = gradio.Textbox('', interactive=False, show_label=False, scale=3)
                                action_run_image_from_upload = gradio.Button('run (from image)', scale=1)
                        with gradio.Group():
                            with gradio.Row():
                                input_image_path2 = gradio.Textbox('', interactive=True, show_label=False, scale=3)
                                action_run_image_from_video = gradio.Button('run (from video)', scale=1)
                    output_image = gradio.Image(label='output image', height=512, interactive=False)
            with gradio.Tab('video'):
                with gradio.Row():
                    with gradio.Column():
                        input_video = gradio.Video(label='input video', height=512)
                        with gradio.Group():
                            with gradio.Row():
                                input_video_path1 = gradio.Textbox('', interactive=False, show_label=False, scale=3)
                                action_run_video1 = gradio.Button('run (from upload)', scale=1)
                        with gradio.Group():
                            with gradio.Row():
                                input_video_path2 = gradio.Textbox('', interactive=True, show_label=False, scale=3)
                                action_run_video2 = gradio.Button('run (from local)', scale=1)
                    output_video = gradio.Image(label='output video', height=512, interactive=False)

        # action
        action_run_image_from_upload.click(
            fn=LibHeadDetectionInterface.actionDetectImage,
            inputs=[input_image, dropdown_models],
            outputs=output_image)
        action_run_image_from_video.click(
            fn=LibHeadDetectionInterface.actionDetectImageFromVideo,
            inputs=[input_image_path2, dropdown_models, input_number_index, input_score_threshold],
            outputs=output_image)
        input_number_index.change(
            fn=LibHeadDetectionInterface.actionDetectImageFromVideo,
            inputs=[input_image_path2, dropdown_models, input_number_index, input_score_threshold],
            outputs=output_image)
        # action_run_video1.click(
        #     fn=LibHeadDetectionInterface.actionDetectVideo,
        #     inputs=[input_video, dropdown_models],
        #     outputs=output_video)
        # action_run_video2.click(
        #     fn=LibHeadDetectionInterface.actionDetectVideo,
        #     inputs=[input_video_path2, dropdown_models],
        #     outputs=output_video)

    @staticmethod
    def actionDetectImage(rgb, model, score_threshold=0.5):
        import supervision
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        canvas_bgr = bgr.copy()
        scores, boxes = XManager.getModules(model)(bgr, conf=score_threshold)
        detections = supervision.Detections(
            xyxy=boxes, confidence=scores, class_id=np.ones_like(scores, dtype=np.int32))
        box_annotator = supervision.BoxAnnotator()
        canvas_bgr = box_annotator.annotate(canvas_bgr, detections)
        # canvas_bgr = LibHeadDetectionInterface.visual(canvas_bgr, scores, boxes)
        return np.copy(canvas_bgr[:, :, ::-1])

    @staticmethod
    def actionDetectImageFromVideo(path_video, model, n_index, score_threshold):
        from ...utils import XVideoReader, FFMPEGHelper
        assert os.path.exists(path_video), path_video
        reader = XVideoReader(path_video)
        flag = False
        bgr = None
        if FFMPEGHelper.checkVideoFPS(path_video, 0) is True:
            reader.resetPositionByIndex(n_index)
            flag, bgr = reader.read()
        else:
            for _ in range(n_index):
                if reader.read()[0] is False:
                    flag = False
                    bgr = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
            flag, bgr = reader.read()
        if flag is True:
            return LibHeadDetectionInterface.actionDetectImage(bgr[:, :, ::-1], model, score_threshold)
        else:
            return np.zeros(shape=(512, 512, 3), dtype=np.uint8)

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
        self.input_height = 480
        self.input_width = 640

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def formatResult(bgr, format_size, outputs, padding):
        src_h, src_w = bgr.shape[0], bgr.shape[1]
        fmt_h, fmt_w = format_size
        if isinstance(outputs, list) and len(outputs) == 2:
            # tiny-post onnx
            scores, boxes = outputs[0], outputs[1][:, 2:]
            boxes_new = np.copy(boxes)
            boxes[:, 0::2] = boxes_new[:, 1::2]
            boxes[:, 1::2] = boxes_new[:, 0::2]
        else:
            # large-post onnx
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
    def formatSizeWithPaddingForward(bgr, dst_h, dst_w, padding_value=255):
        # base on long side, padding to target size
        src_h, src_w, _ = bgr.shape
        src_ratio = float(src_h / src_w)
        dst_ratio = float(dst_h / dst_w)
        if src_ratio > dst_ratio:
            rsz_h, rsz_w = dst_h, int(round(float(src_w / src_h) * dst_h))
            resized = cv2.resize(bgr, (max(1, rsz_w), max(1, rsz_h)))
            lp = (dst_w - rsz_w) // 2
            rp = dst_w - rsz_w - lp
            # resized = np.pad(resized, ((0, 0), (lp, rp), (0, 0)), constant_values=padding_value, mode='constant')
            resized = cv2.copyMakeBorder(resized, 0, 0, lp, rp, cv2.BORDER_CONSTANT, value=padding_value)
            padding = (0, 0, lp, rp)
        else:
            rsz_h, rsz_w = int(round(float(src_h / src_w) * dst_w)), dst_w
            resized = cv2.resize(bgr, (max(1, rsz_w), max(1, rsz_h)))
            tp = (dst_h - rsz_h) // 2
            bp = dst_h - rsz_h - tp
            # resized = np.pad(resized, ((tp, bp), (0, 0), (0, 0)), constant_values=255, mode='constant')
            resized = cv2.copyMakeBorder(resized, 0, 0, lp, rp, cv2.BORDER_CONSTANT, value=padding_value)
            padding = (tp, bp, 0, 0)
        return resized, padding

    @staticmethod
    def doNonMaximumSuppression(scores, boxes, nms_threshold):
        # pure python NMS baseline
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

    def preprocess(self, image: np.ndarray, swap=(2, 0, 1)):
        # Normalization + BGR->RGB
        padding_value = 255
        dst_h, dst_w = self.input_height, self.input_width
        src_h, src_w, _ = image.shape
        src_ratio = float(src_h / src_w)
        dst_ratio = float(dst_h / dst_w)
        if src_ratio > dst_ratio:
            rsz_h, rsz_w = dst_h, int(round(float(src_w / src_h) * dst_h))
            resized = cv2.resize(image, (max(1, rsz_w), max(1, rsz_h)))
            lp = (dst_w - rsz_w) // 2
            rp = dst_w - rsz_w - lp
            resized = cv2.copyMakeBorder(resized, 0, 0, lp, rp, cv2.BORDER_CONSTANT, value=padding_value)
            padding = (0, 0, lp, rp)
        else:
            rsz_h, rsz_w = int(round(float(src_h / src_w) * dst_w)), dst_w
            resized = cv2.resize(image, (max(1, rsz_w), max(1, rsz_h)))
            tp = (dst_h - rsz_h) // 2
            bp = dst_h - rsz_h - tp
            resized = cv2.copyMakeBorder(resized, tp, bp, 0, 0, cv2.BORDER_CONSTANT, value=padding_value)
            padding = (tp, bp, 0, 0)
        fmt_h, fmt_w = resized.shape[0], resized.shape[1]
        inference_image = cv2.dnn.blobFromImage(resized, scalefactor=1/255., mean=(0, 0, 0), swapRB=True)
        return inference_image, padding, (fmt_h, fmt_w), resized
    
    def forward(self, batch_image):
        raise NotImplementedError

    def inference(self, bgr: np.ndarray, score_threshold, nms_threshold, **kwargs):
        # PreProcess
        inference_image, padding, format_size, format_image = self.preprocess(bgr)
        # Inference
        outputs = self.forward(inference_image)
        scores, boxes = self.formatResult(bgr, format_size, outputs, padding)
        # PostProcess
        scores, boxes = self.postprocess(scores, boxes, score_threshold, nms_threshold)
        return scores, boxes, format_image

    def postprocess(self, scores, boxes, score_threshold, nms_threshold):
        scores = np.reshape(scores, (-1,))
        keep = np.where(scores > score_threshold)[0]
        scores = scores[keep]
        boxes = boxes[keep]
        # Top-K
        top_k = scores.argsort()[::-1][:self.top_k]
        scores = scores[top_k]
        boxes = boxes[top_k]
        # NMS
        keep = self.doNonMaximumSuppression(scores, boxes, nms_threshold)
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
        score_threshold = kwargs.pop('score_threshold', self.score_threshold)
        nms_threshold = kwargs.pop('nms_threshold', self.nms_threshold)
        return targets, dict(score_threshold=score_threshold, nms_threshold=nms_threshold)

    def returnResult(self, bgr, output, targets):
        def _formatResult(target):
            scores, boxes, format_image = output[:3]
            if target == 'source':
                return scores, boxes
            if target == 'visual':
                return self.visual(np.copy(bgr), boxes)
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


class LibHeadDetectionOnnx(LibHeadDetectionInterface):
    """
    """
    Config = {
        'tiny': {
            'checkpoint': 'thirdparty/yolov7_tiny_head_0.768_post_480x640.onnx',
            'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider']
         },
        'large': {
            'checkpoint': 'thirdparty/yolov4_headdetection_480x640_post.onnx',
            'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        },
    }

    """
    """
    def __init__(self, model, *args, **kwargs):
        super(LibHeadDetectionOnnx, self).__init__()
        self.config = LibHeadDetectionOnnx.Config[model]
        self.session_onnx = None
        self.input_shapes = None
        self.input_names = None
        self.output_names = None
        self.input_height = None
        self.input_width = None

    def initialize(self, *args, **kwargs):
        if self.root is None:
            self.root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
        if self.session_onnx is None:
            path = '{}/{}'.format(self.root, self.config['checkpoint'])
            assert os.path.exists(path), path
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self.session_onnx = onnxruntime.InferenceSession(
                path, sess_options=session_option, providers=self.config['providers'])
            self.input_shapes = [each.shape for each in self.session_onnx.get_inputs()]
            self.input_names = [each.name for each in self.session_onnx.get_inputs()]
            self.output_names = [each.name for each in self.session_onnx.get_outputs()]
            self.input_height = self.input_shapes[0][2]
            self.input_width = self.input_shapes[0][3]

    def forward(self, batch_image):
        return self.session_onnx.run(
            self.output_names, {input_name: batch_image for input_name in self.input_names}, )


class LibHeadDetectionTensorRT(LibHeadDetectionInterface):
    """
    """
    TrtEngineType = ['fp32', 'fp16', 'int8']

    @staticmethod
    def formatPath(path_onnx, mode):
        return '{}-{}.engine'.format(os.path.splitext(path_onnx)[0], mode)

    @staticmethod
    def exportAsTensorRT(root, name, mode, **kwargs):
        from .trt_calibrator import EntropyCalibratorLibHeadDetection
        from ...utils.tensorrt import build_engine
        assert mode in ['fp32', 'fp16', 'int8'], 'mode should be in ["fp32", "fp16", "int8"]'
        path_onnx = '{}/{}'.format(root, LibHeadDetectionOnnx.Config[name]['checkpoint'])
        path_trt = LibHeadDetectionTensorRT.formatPath(path_onnx, mode)
        calib = EntropyCalibratorLibHeadDetection(kwargs.pop('data')) if mode == 'int8' else None
        build_engine(path_onnx, path_trt, 1, mode, calib)

    """
    """
    def __init__(self, model, trt_engine, *args, **kwargs):
        super(LibHeadDetectionTensorRT, self).__init__()
        assert trt_engine in LibHeadDetectionTensorRT.TrtEngineType, trt_engine
        self.model = model
        self.trt_engine = trt_engine
        # onnx engine
        self.trt_session = XManager.createEngine(self.getConfig())
        self.input_height = 480
        self.input_width = 640
        self.root = None

    def getConfig(self, device='cuda:0'):
        path_onnx = copy.deepcopy(LibHeadDetectionOnnx.Config[self.model]['checkpoint'])
        path_engine = '{}-{}.engine'.format(path_onnx[:-5], self.trt_engine)
        return dict(type='tensorrt', device='cuda:0', parameters=path_engine)

    def initialize(self, *args, **kwargs):
        if self.root is None:
            self.root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
        self.trt_session.initialize(*args, **kwargs)

    def forward(self, batch_image):
        return self.trt_session.inference(batch_image)


class LibHeadDetection:
    """
    """
    Model = ['tiny', 'large']
    Engine = ['fp32', 'fp16', 'int8', 'onnx']

    @staticmethod
    def getResources():
        raise [
            LibHeadDetectionOnnx.Config['tiny']['checkpoint'],
            LibHeadDetectionOnnx.Config['large']['checkpoint'],
        ]

    """
    """
    def __init__(self, *args, **kwargs):
        self.dict = dict()
        self.root = None

    def __getitem__(self, item):
        assert isinstance(item, str), item
        model, engine = item.split('-')
        assert model in self.Model, model
        assert engine in self.Engine, engine
        if item not in self.dict:
            if engine == 'onnx':
                module = LibHeadDetectionOnnx(model)
                module.initialize(root=self.root)
                self.dict[item] = module
            else:
                module = LibHeadDetectionTensorRT(model, engine)
                module.initialize(root=self.root)
                self.dict[item] = module
        return self.dict[item]

    def initialize(self, *args, **kwargs):
        if self.root is None:
            self.root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='build tensorrt')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--mode', type=str, default='int8', required=True)
    parser.add_argument('--data', type=str, default='')
    parse_args = parser.parse_args()
    LibHeadDetectionTensorRT.exportAsTensorRT(parse_args.root, parse_args.name, parse_args.mode, data=parse_args.data)

