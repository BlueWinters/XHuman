
import logging
import os
import copy
import cv2
import numpy as np
import onnxruntime
import dataclasses
import typing
from ... import XManager


@dataclasses.dataclass(frozen=False)
class InfoBox:
    class_id: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def box(self):
        if not hasattr(self, 'box_array'):
            self.box_array = np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.int32)
        return self.box_array


@dataclasses.dataclass(frozen=False)
class InfoHF:
    face: typing.Union[InfoBox, None]
    head: typing.Union[InfoBox, None]


class LibHeadFaceDetectionInterface:
    @staticmethod
    def draw_dashed_line(image_bgr: np.ndarray, pt1, pt2, color, thickness: int = 1, dash_length: int = 10):
        """Function to draw a dashed line"""
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        dashes = int(dist / dash_length)
        for i in range(dashes):
            start = [int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)]
            end = [int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)]
            cv2.line(image_bgr, tuple(start), tuple(end), color, thickness)

    @staticmethod
    def drawDashedRectangle(
        image_bgr: np.ndarray,
        top_left,
        bottom_right,
        color,
        thickness: int = 1,
        dash_length: int = 10
    ):
        """Function to draw a dashed rectangle"""
        tl_tr = (bottom_right[0], top_left[1])
        bl_br = (top_left[0], bottom_right[1])
        LibHeadFaceDetectionInterface.draw_dashed_line(image_bgr, top_left, tl_tr, color, thickness, dash_length)
        LibHeadFaceDetectionInterface.draw_dashed_line(image_bgr, tl_tr, bottom_right, color, thickness, dash_length)
        LibHeadFaceDetectionInterface.draw_dashed_line(image_bgr, bottom_right, bl_br, color, thickness, dash_length)
        LibHeadFaceDetectionInterface.draw_dashed_line(image_bgr, bl_br, top_left, color, thickness, dash_length)

    @staticmethod
    def visual(canvas_bgr, boxes):
        for n, box in enumerate(boxes):
            if isinstance(box, InfoBox):
                class_id: int = box.class_id
                color_map = [(255, 0, 0), (0, 0, 255), (0, 200, 255)]
                color = color_map[class_id]
                if class_id == 1:
                    cv2.rectangle(canvas_bgr, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), 3)
                    cv2.rectangle(canvas_bgr, (box.x1, box.y1), (box.x2, box.y2), color, 2)
                elif class_id == 2:
                    LibHeadFaceDetectionInterface.drawDashedRectangle(canvas_bgr, (box.x1, box.y1), (box.x2, box.y2), color, 2, 10)
                continue
            if isinstance(box, InfoHF):
                from core.utils import Colors
                color = Colors.getColorValue(n)
                assert box.face is not None or box.head is not None, box
                if box.head is not None:
                    cv2.rectangle(canvas_bgr, (box.head.x1, box.head.y1), (box.head.x2, box.head.y2), color, 2)
                if box.face is not None:
                    LibHeadFaceDetectionInterface.drawDashedRectangle(
                        canvas_bgr, (box.face.x1, box.face.y1), (box.face.x2, box.face.y2), color, 2, 10)
                continue
            raise TypeError(box)
        return canvas_bgr

    """
    """
    DtypeDict = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    ClassMap = {
        0: 'Body',
        1: 'Head',
        2: 'Face',
        3: 'Hand',
        4: 'Left-Hand',
        5: 'Right-Hand',
    }

    # reference: https://github.com/PINTO0309/PINTO_model_zoo/blob/main/442_YOLOX-Body-Head-Face-HandLR-Dist/demo/demo_yolox_onnx_handLR.py

    def __init__(self):
        self.score_threshold = 0.5
        self.root = None
        # session
        self.input_height = None
        self.input_width = None

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

    def preprocess(self, image_bgr: np.ndarray, swap=(2, 0, 1), force_resized=True):
        # Normalization + BGR->RGB
        if force_resized is True:
            print(self.input_width, self.input_height)
            resized_image = cv2.resize(image_bgr, (self.input_width, self.input_height))
            resized_image = resized_image.transpose((2, 0, 1))
            resized_image = np.ascontiguousarray(resized_image, dtype=np.float32)
            inference_image = np.asarray([resized_image], dtype=np.float32)
            return inference_image, (0, 0, 0, 0), (self.input_height, self.input_width), resized_image
        else:
            resized_image, padding = self.formatSizeWithPaddingForward(image_bgr, self.input_height, self.input_width)
            fmt_h, fmt_w = resized_image.shape[0], resized_image.shape[1]
            format_image = resized_image.transpose(swap)
            format_image = np.ascontiguousarray(format_image, dtype=np.float32)
            batch_image = np.asarray([format_image], dtype=np.float32)
            return batch_image, padding, (fmt_h, fmt_w), resized_image

    def forward(self, batch_image):
        raise NotImplementedError

    def inference(self, image_bgr: np.ndarray, score_threshold: float):
        # Predicted boxes: [class_id, score, x1, y1, x2, y2, cx, cy, handedness, is_hand_used=False]
        # PreProcess
        batch_image, padding, format_size, format_image = self.preprocess(image_bgr)
        # Inference
        outputs = self.forward(batch_image)[0]
        # PostProcess
        result_boxes = self.postprocess(image_bgr, outputs[0], padding, score_threshold)
        return result_boxes
    
    def inference2(self, image_bgr_list: np.ndarray, score_threshold: float):
        # Predicted boxes: [class_id, score, x1, y1, x2, y2, cx, cy, handedness, is_hand_used=False]
        # PreProcess
        format_list = [self.preprocess(image_bgr)[:2] for image_bgr in image_bgr_list]
        batch_image = np.concatenate([image[0] for image in format_list], axis=0)
        # Inference
        outputs = self.forward(batch_image)
        # PostProcess
        result_boxes = [self.postprocess(
            image_bgr_list[n], outputs[n], format_list[n][1], score_threshold) for n in range(len(image_bgr_list))]
        return result_boxes

    def postprocess(self, image_bgr: np.ndarray, boxes: np.ndarray, padding, score_threshold):
        # boxes: batch_num, class_id, score, x1, y1, x2, y2
        src_h = image_bgr.shape[0]
        src_w = image_bgr.shape[1]
        result_boxes = []
        print(boxes.shape)
        if len(boxes) > 0:
            scores = boxes[:, 2:3]
            keep_index = scores[:, 0] > score_threshold
            scores_keep = scores[keep_index, :]
            boxes_keep = boxes[keep_index, :]
            if len(boxes_keep) > 0:
                fmt_w = self.input_width
                fmt_h = self.input_height
                for box, score in zip(boxes_keep, scores_keep):
                    class_id = int(box[1])
                    if class_id not in [1, 2]:
                        continue  # just keep 'head', 'face'
                    tp, bp, lp, rp = padding
                    rescale_w = (fmt_w - lp - rp) / src_w
                    rescale_h = (fmt_h - tp - bp) / src_h
                    boxes[:, 0::2] = np.clip((boxes[:, 0::2] - lp) / rescale_w, 0, src_w)
                    boxes[:, 1::2] = np.clip((boxes[:, 1::2] - tp) / rescale_h, 0, src_h)
                    x_min = int(max(0, box[3] - lp) / rescale_w)
                    y_min = int(max(0, box[4] - tp) / rescale_h)
                    x_max = int(min(box[5] - lp, fmt_w) / rescale_w)
                    y_max = int(min(box[6] - tp, fmt_h) / rescale_h)
                    if bool(x_min < x_max and y_min < y_max) is True:
                        result_box = InfoBox(
                            class_id, float(score), x_min, y_min, x_max, y_max)
                        result_boxes.append(result_box)
        return result_boxes

    @staticmethod
    def estimateInsideRatio(base_obj, target_obj) -> float:
        # Calculate areas of overlap
        inter_xmin = max(base_obj.x1, target_obj.x1)
        inter_ymin = max(base_obj.y1, target_obj.y1)
        inter_xmax = min(base_obj.x2, target_obj.x2)
        inter_ymax = min(base_obj.y2, target_obj.y2)
        # If there is no overlap
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        # Calculate area of overlap and area of each bounding box
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (base_obj.x2 - base_obj.x1) * (base_obj.y2 - base_obj.y1)
        # Calculate IoU
        iou = inter_area / float(area1)
        return iou

    @staticmethod
    def matchHeadAndFace(box_info_list, auto_filter=True, iou_threshold=0.99):
        result_list = []
        info_head_list = [box_info for box_info in box_info_list if box_info.class_id == 1]
        info_face_list = [box_info for box_info in box_info_list if box_info.class_id == 2]
        info_head_list = sorted(info_head_list, key=lambda info: info.score, reverse=True)
        while len(info_face_list) > 0:
            info_face = info_face_list.pop(0)
            if len(info_head_list) > 0:
                iou_list = [LibHeadFaceDetectionInterface.estimateInsideRatio(
                    info_face, info_head) for info_head in info_head_list]
                iou = np.array(iou_list, dtype=np.float32)
                index = np.argmax(iou)
                if iou[index] >= iou_threshold:
                    info_head = info_head_list.pop(index)
                    result_list.append(InfoHF(face=info_face, head=info_head))
                else:
                    if auto_filter is False:
                        result_list.append(InfoHF(face=info_face, head=None))
        for info_head in info_head_list:
            result_list.append(InfoHF(face=None, head=info_head))
        return result_list

    """
    """
    def extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        score_threshold = kwargs.pop('score_threshold', self.score_threshold)
        return targets, dict(score_threshold=score_threshold)

    def returnResult(self, bgr, output, targets):
        def _formatResult(target):
            result_boxes = output
            if target == 'source':
                return result_boxes
            if target == 'format':
                return LibHeadFaceDetectionInterface.matchHeadAndFace(result_boxes)
            if target == 'visual-source':
                return self.visual(np.copy(bgr), result_boxes)
            if target == 'visual-format':
                format_boxes = LibHeadFaceDetectionInterface.matchHeadAndFace(result_boxes)
                return self.visual(np.copy(bgr), format_boxes)
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


class LibHeadFaceDetectionOnnx(LibHeadFaceDetectionInterface):
    """
    """
    Config = {
        'yolox_n': {
            'checkpoint': 'thirdparty/yolox_n_body_head_face_handLR_dist_0219_0.3453_post_1x3x640x640.onnx',
            'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
         },
        'yolox_s': {
            'checkpoint': 'thirdparty/yolox_s_body_head_face_handLR_dist_0189_0.4482_post_1x3x640x640.onnx',
            'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        },
        'yolox_s_nchw': {
            'checkpoint': 'thirdparty/yolox_s_body_head_face_handLR_dist_Nx3xHxW.onnx',
            'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        },
    }

    """
    """
    def __init__(self, model, *args, **kwargs):
        super(LibHeadFaceDetectionOnnx, self).__init__()
        self.model = model
        self.config = LibHeadFaceDetectionOnnx.Config[model]
        self.session_onnx = None
        self.input_shapes = None
        self.input_names = None
        self.output_names = None

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
            self.session_onnx.get_providers()
            self.input_shapes = [each.shape for each in self.session_onnx.get_inputs()]
            self.input_names = [each.name for each in self.session_onnx.get_inputs()]
            self.output_names = [each.name for each in self.session_onnx.get_outputs()]
            self.input_height = self.input_shapes[0][2]
            self.input_width = self.input_shapes[0][3]
            if self.model == 'yolox_s_nchw':
                self.input_height = 640
                self.input_width = 640

    def forward(self, batch_image):
        data = {str(self.input_names[0]): batch_image}
        outputs = [output for output in self.session_onnx.run(self.output_names, data)]
        return outputs


class LibHeadFaceDetectionTensorRT(LibHeadFaceDetectionInterface):
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
        path_onnx = '{}/{}'.format(root, LibHeadFaceDetectionOnnx.Config[name]['checkpoint'])
        path_trt = LibHeadFaceDetectionTensorRT.formatPath(path_onnx, mode)
        calib = EntropyCalibratorLibHeadDetection(kwargs.pop('data')) if mode == 'int8' else None
        build_engine(path_onnx, path_trt, 1, mode, calib)

    """
    """
    def __init__(self, model, trt_engine, *args, **kwargs):
        super(LibHeadFaceDetectionTensorRT, self).__init__()
        assert trt_engine in LibHeadFaceDetectionTensorRT.TrtEngineType, trt_engine
        self.model = model
        self.trt_engine = trt_engine
        # onnx engine
        self.trt_session = XManager.createEngine(self.getConfig())
        self.input_height = 640
        self.input_width = 640
        self.root = None

    def getConfig(self, device='cuda:0'):
        path_onnx = copy.deepcopy(LibHeadFaceDetectionOnnx.Config[self.model]['checkpoint'])
        path_engine = '{}-{}.engine'.format(path_onnx[:-5], self.trt_engine)
        return dict(type='tensorrt', device='cuda:0', parameters=path_engine)

    def initialize(self, *args, **kwargs):
        if self.root is None:
            self.root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
        self.trt_session.initialize(*args, **kwargs)

    def forward(self, batch_image):
        return self.trt_session.inference(batch_image)


class LibHeadFaceDetection:
    """
    """
    Model = ['yolox_n', 'yolox_s', 'yolox_s_nchw']
    Engine = ['fp32', 'fp16', 'int8', 'onnx']

    @staticmethod
    def getResources():
        raise [
            LibHeadFaceDetectionOnnx.Config['yolox_n']['checkpoint'],
            LibHeadFaceDetectionOnnx.Config['yolox_s']['checkpoint'],
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
                module = LibHeadFaceDetectionOnnx(model)
                module.initialize(root=self.root)
                self.dict[item] = module
            else:
                module = LibHeadFaceDetectionTensorRT(model, engine)
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
    LibHeadFaceDetectionTensorRT.exportAsTensorRT(parse_args.root, parse_args.name, parse_args.mode, data=parse_args.data)
