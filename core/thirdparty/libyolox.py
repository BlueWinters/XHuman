
import logging
import cv2
import numpy as np
import rtmlib
# from rtmlib import YOLOX
from rtmlib.tools.object_detection.post_processings import multiclass_nms
from .. import XManager



class LibYoloXWrapper:
    """
    """
    @staticmethod
    def visualBoxes(bgr, boxes, color=(0, 255, 0)):
        for bbox in boxes:
            bgr = cv2.rectangle(bgr, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        return bgr

    @staticmethod
    def benchmark():
        module = LibYoloXWrapper()
        module.initialize()
        bgr = cv2.imread('benchmark/asset/rtmpose/input.png')
        scores, boxes = module(bgr, targets='source')
        print(boxes.shape, scores.shape)
        bgr = module.visualBoxes(bgr, boxes)
        cv2.imwrite('benchmark/asset/rtmpose/output_yolox.png', bgr)

    """
    """
    ConfigDict = {
        # reference: https://github.com/Tau-J/rtmlib
        'yolox-m': {
            'onnx_model': 'thirdparty/rtmpose/yolox_onnx/yolox_m_8xb8-300e_humanart-c2c7a14a/end2end.onnx',
            'model_input_size': (640, 640),
            'nms_thr': 0.45,
            'score_thr': 0.7,
            'backend': 'onnxruntime',
            'device': 'cuda',
        },
    }

    def __init__(self, *args, **kwargs):
        self.model_name = kwargs['model_name'] if 'model_name' in kwargs else 'yolox-m'
        self.config = self.ConfigDict[self.model_name]

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        if hasattr(self, '_detector') is False:
            root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
            self.config['onnx_model'] = '{}/{}'.format(root, self.config['onnx_model'])
            self._detector = rtmlib.YOLOX(**self.config)

    @property
    def detector(self):
        if hasattr(self, '_detector') is False:
            self.initialize()
        return self._detector

    """
    """
    def postprocess(self, outputs, ratio):
        """Do postprocessing for RTMPose model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of RTMPose model.
            ratio (float): Ratio of preprocessing.

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_scores (np.ndarray): Final scores.
        """

        model_input_size = self.config['model_input_size']
        score_thr = self.config['score_thr']
        nms_thr = self.config['nms_thr']

        if outputs.shape[-1] == 4:
            # onnx without nms module
            grids = []
            expanded_strides = []
            strides = [8, 16, 32]

            hsizes = [model_input_size[0] // stride for stride in strides]
            wsizes = [model_input_size[1] // stride for stride in strides]

            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                expanded_strides.append(np.full((*shape, 1), stride))

            grids = np.concatenate(grids, 1)
            expanded_strides = np.concatenate(expanded_strides, 1)
            outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
            outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

            predictions = outputs[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)
            if dets is not None:
                pack_dets = (dets[:, :4], dets[:, 4], dets[:, 5])
                final_boxes, final_scores, final_cls_inds = pack_dets
                isscore = final_scores > 0.3
                iscat = final_cls_inds == 0
                isbbox = [i and j for (i, j) in zip(isscore, iscat)]
                final_boxes = final_boxes[isbbox]
                final_scores = final_scores[isbbox]

        elif outputs.shape[-1] == 5:
            # onnx contains nms module
            pack_dets = (outputs[0, :, :4], outputs[0, :, 4])
            final_boxes, final_scores = pack_dets
            final_boxes /= ratio
            isscore = final_scores > 0.5
            isbbox = [i for i in isscore]
            final_boxes = final_boxes[isbbox]
            final_scores = final_scores[isbbox]

        return final_scores, np.round(final_boxes).astype(np.int32)

    """
    """
    def _extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        return targets

    def _returnResult(self, bgr, output, targets):
        def _formatResult(target):
            scores, boxes = output
            if target == 'source':
                return output
            if target == 'visual':
                return self.visualBoxes(np.copy(bgr), boxes)
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, bgr, *args, **kwargs):
        targets = self._extractArgs(*args, **kwargs)
        bgr, ratio = self.detector.preprocess(bgr)
        outputs = self.detector.inference(bgr)[0]
        output = self.postprocess(outputs, ratio)
        return self._returnResult(bgr, output, targets)
