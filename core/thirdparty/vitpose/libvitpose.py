
import logging
import os
import cv2
import onnxruntime
import numpy as np
import copy
from .top2down import keypoints_from_heatmaps
from .visual import visualEachPerson
from ... import XManager


class LibVitPoseInterface:
    """
    """
    @staticmethod
    def constructUITab():
        import gradio
        with gradio.Tab('Vit-Pose'):
            pass

    @staticmethod
    def getResources():
        raise NotImplementedError

    @staticmethod
    def visual(canvas_bgr, boxes, key_points):
        from ...utils.color import ColorMap
        assert len(boxes) == len(key_points), (len(boxes), len(key_points))
        for n in range(len(boxes)):
            visualEachPerson(canvas_bgr, boxes[n, :], key_points[n, :, :], ColorMap.getColorValue(n))
        return canvas_bgr

    """
    """
    def __init__(self, *args, **kwargs):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.score_threshold = 0.5
        self.nms_threshold = 0.6
        self.top_k = 64
        self.root = None
        # session
        self.target_size = [192, 256]
        self.fmt_h = 256
        self.fmt_w = 192

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def pad_image(image: np.ndarray, aspect_ratio: float):
        # Get the current aspect ratio of the image
        image_height, image_width = image.shape[:2]
        current_aspect_ratio = image_width / image_height
        left_pad = 0
        top_pad = 0
        # Determine whether to pad horizontally or vertically
        if current_aspect_ratio < aspect_ratio:
            # Pad horizontally
            target_width = int(aspect_ratio * image_height)
            pad_width = target_width - image_width
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            padded_image = np.pad(image, pad_width=((0, 0), (left_pad, right_pad), (0, 0)), mode='constant')
        else:
            # Pad vertically
            target_height = int(image_width / aspect_ratio)
            pad_height = target_height - image_height
            top_pad = pad_height // 2
            bottom_pad = pad_height - top_pad
            padded_image = np.pad(image, pad_width=((top_pad, bottom_pad), (0, 0), (0, 0)), mode='constant')
        return padded_image, (left_pad, top_pad)

    def format(self, image_bgr):
        org_h, org_w = image_bgr.shape[:2]
        img_input = cv2.resize(image_bgr, self.target_size, interpolation=cv2.INTER_LINEAR) / 255.
        img_input = ((img_input - self.mean) / self.std).transpose(2, 0, 1)[None].astype(np.float32)
        return img_input, org_h, org_w

    def preprocess(self, image_bgr, detections, pad_bbox=10):
        scores, boxes = detections
        if len(scores) > 0:
            batch_images = np.zeros(shape=(len(scores), 3, self.fmt_h, self.fmt_w), dtype=np.float32)
            clip_info_list = []
            for n, box in enumerate(boxes):
                # note: Slightly bigger bbox
                box[[0, 2]] = np.clip(box[[0, 2]] + [-pad_bbox, pad_bbox], 0, image_bgr.shape[1])
                box[[1, 3]] = np.clip(box[[1, 3]] + [-pad_bbox, pad_bbox], 0, image_bgr.shape[0])
                # Crop image and pad to 3/4 aspect ratio
                img_inf = image_bgr[box[1]:box[3], box[0]:box[2]]
                img_inf, (left_pad, top_pad) = self.pad_image(img_inf, 3 / 4)
                format_image, org_h, org_w = self.format(img_inf)
                batch_images[n] = format_image
                clip_info_list.append((np.copy(box), left_pad, top_pad, org_w, org_h))
            return batch_images, clip_info_list
        return [], []

    def forward(self, batch_image):
        raise NotImplementedError

    def detect(self, image_bgr, detections):
        if detections is None:
            module = XManager.getModules('ultralytics')['yolo11n']
            results = module(image_bgr, verbose=False, conf=0.5, classes=[0])[0]
            if len(results) > 0:
                scores = np.reshape(results.boxes.conf.cpu().numpy().astype(np.float32), (-1,))
                boxes = np.reshape(np.round(results.boxes.xyxy.cpu().numpy()).astype(np.int32), (-1, 4,))
                return scores, boxes
            return [], []
        if isinstance(detections, dict):
            # example: {'model': ultralytics-yolo11n, 'confidence': 0.5}
            model = detections['model']
            confidence = float(detections['confidence'] if 'confidence' in detections else 0.5)
            assert 0 < confidence < 1, confidence
            if 'ultralytics' in model:
                # ultralytics-yolo11n
                yolo_model = model.split('-')[1]
                module = XManager.getModules('ultralytics')[yolo_model]
                results = module(image_bgr, verbose=False, conf=confidence, classes=[0])[0]
                if len(results) > 0:
                    scores = np.reshape(results.boxes.conf.cpu().numpy().astype(np.float32), (-1,))
                    boxes = np.reshape(np.round(results.boxes.xyxy.cpu().numpy()).astype(np.int32), (-1, 4,))
                    return scores, boxes
                return [], []
        raise NotImplementedError(detections)

    def inference(self, image_bgr: np.ndarray, detections=None, **kwargs):
        detections = self.detect(image_bgr, detections)
        batch_input, clip_info_list = self.preprocess(image_bgr, detections)
        outputs = self.forward(batch_input) if len(batch_input) > 0 else []
        key_points = self.postprocess(outputs, clip_info_list)
        return *detections, key_points

    @staticmethod
    def postprocess(heatmaps, clip_info_list):
        if len(clip_info_list) > 0:
            boxes = np.zeros(shape=(len(clip_info_list), 4), dtype=np.int32)
            centers = np.zeros(shape=(len(clip_info_list), 2), dtype=np.int32)
            scales = np.zeros(shape=(len(clip_info_list), 2), dtype=np.int32)
            padding = np.zeros(shape=(len(clip_info_list), 2), dtype=np.int32)
            for n, clip_info in enumerate(clip_info_list):
                box, left_pad, top_pad, org_w, org_h = clip_info
                boxes[n, :] = box
                centers[n, :] = np.array([org_w // 2, org_h // 2])
                scales[n, :] = np.array([org_w, org_h])
                padding[n, :] = np.array([left_pad, top_pad])
            points, prob = keypoints_from_heatmaps(
                    heatmaps=heatmaps,
                    center=centers,
                    scale=scales,
                    unbiased=True, use_udp=True)
            points += (boxes[:, :2] - padding)[:, None, :]  # N,2 --> N,1,2 --> N,K,2
            key_points = np.concatenate([points, prob], axis=2)
            return key_points
        return []

    """
    """
    def extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        detections = kwargs.pop('detections', None)
        return targets, dict(detections=detections)

    def returnResult(self, bgr, output, targets):
        def _formatResult(target):
            scores, boxes, key_points = output
            if target == 'source':
                return scores, boxes, key_points
            if target == 'visual':
                return self.visual(np.copy(bgr), boxes, key_points)
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


class LibVitPoseOnnx(LibVitPoseInterface):
    """
    """
    Config = {
        'vitpose_s_coco_25': {
            'checkpoint': 'thirdparty/vitpose-s-coco_25.onnx',
            'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider']
         },
    }

    """
    """
    def __init__(self, model, *args, **kwargs):
        super(LibVitPoseOnnx, self).__init__()
        self.config = LibVitPoseOnnx.Config[model]
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
            self.input_names = [each.prefix for each in self.session_onnx.get_inputs()]
            self.output_names = [each.prefix for each in self.session_onnx.get_outputs()]
            self.input_height = self.input_shapes[0][2]
            self.input_width = self.input_shapes[0][3]

    def forward(self, batch_image):
        assert len(batch_image) > 0
        return self.session_onnx.run(
            self.output_names, {input_name: batch_image for input_name in self.input_names}, )[0]

