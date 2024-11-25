
import logging
import os
import cv2
import numpy as np
from .segmentation import SapiensSegmentation
from .depth import SapiensDepth
from .normal import SapiensNormal
from ... import XManager



class LibSapiensWrapper:
    """
    """
    @staticmethod
    def drawMaps(img, person_boxes, normal_maps, segmentation_maps, depth_maps):
        draw_img = []
        segmentation_img = img.copy()
        for segmentation_map, box in zip(segmentation_maps, person_boxes):
            mask = segmentation_map > 0
            crop = segmentation_img[box[1]:box[3], box[0]:box[2]]
            segmentation_draw = SapiensSegmentation.visualSegmentationMap(segmentation_map)
            crop_draw = cv2.addWeighted(crop, 0.5, segmentation_draw, 0.7, 0)
            segmentation_img[box[1]:box[3], box[0]:box[2]] = crop_draw * mask[..., None] + crop * ~mask[..., None]
        draw_img.append(segmentation_img)

        if len(normal_maps) > 0:
            normal_img = img.copy()
            for i, (normal_map, box) in enumerate(zip(normal_maps, person_boxes)):
                mask = segmentation_maps[i] > 0
                crop = normal_img[box[1]:box[3], box[0]:box[2]]
                normal_draw = SapiensNormal.visualNormalMap(normal_map)
                crop_draw = cv2.addWeighted(crop, 0.5, normal_draw, 0.7, 0)
                normal_img[box[1]:box[3], box[0]:box[2]] = crop_draw * mask[..., None] + crop * ~mask[..., None]
            draw_img.append(normal_img)

        if len(depth_maps) > 0:
            depth_img = img.copy()
            for i, (depth_map, box) in enumerate(zip(depth_maps, person_boxes)):
                mask = segmentation_maps[i] > 0
                crop = depth_img[box[1]:box[3], box[0]:box[2]]
                depth_map[~mask] = 0
                depth_draw = SapiensDepth.visualDepthMap(depth_map)
                crop_draw = cv2.addWeighted(crop, 0.5, depth_draw, 0.7, 0)
                depth_img[box[1]:box[3], box[0]:box[2]] = crop_draw * mask[..., None] + crop * ~mask[..., None]
            draw_img.append(depth_img)

        return np.hstack(draw_img)
    
    @staticmethod
    def benchmark():
        module = LibSapiensWrapper()
        module.initialize()
        bgr = cv2.imread('benchmark/asset/sapiens/input.png')
        segmentation_maps, depth_maps, normal_maps, person_boxes = module(
            bgr, targets=['segmentation_03b', 'depth_03b', 'normal_03b', 'person_boxes'])
        # cv2.imwrite('benchmark/asset/sapiens/output_segment.png', segmentation_maps[0])
        # cv2.imwrite('benchmark/asset/sapiens/output_depth.png', (255-depth_maps[0]*255).astype(np.uint8))
        # cv2.imwrite('benchmark/asset/sapiens/output_normal.png', normal_maps[0])
        visual = LibSapiensWrapper.drawMaps(bgr, person_boxes, normal_maps, segmentation_maps, depth_maps)
        cv2.imwrite('benchmark/asset/sapiens/output_visual.png', visual)

    @staticmethod
    def getResources():
        return [value for key, value in LibSapiensWrapper.ConfigDict.items()]

    """
    """
    # reference: https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference
    ConfigDict = {
        'segmentation_03b': 'thirdparty/sapiens/sapiens-seg-0.3b-torchscript/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2',
        # 'segmentation_06b': 'sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2',
        # 'segmentation_1b': 'sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2',
        'depth_03b': "thirdparty/sapiens/sapiens-depth-0.3b-torchscript/sapiens_0.3b_render_people_epoch_100_torchscript.pt2",
        # 'depth_06b': "sapiens_0.6b_render_people_epoch_70_torchscript.pt2",
        # 'depth_2b': "sapiens_1b_render_people_epoch_88_torchscript.pt2",
        # 'depth_1b': "sapiens_2b_render_people_epoch_25_torchscript.pt2",
        'normal_03b': "thirdparty/sapiens/sapiens-normal-0.3b-torchscript/sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2",
        # 'normal_06b': "sapiens_0.6b_normal_render_people_epoch_200_torchscript.pt2",
        # 'normal_1b': "sapiens_1b_normal_render_people_epoch_115_torchscript.pt2",
        # 'normal_2b': "sapiens_2b_normal_render_people_epoch_70_torchscript.pt2",
        'pose_03b': 'thirdparty/sapiens/sapiens-pose-0.3b-torchscript/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2',
    }

    """
    """
    def __init__(self, *args, **kwargs):
        self.model_dict = dict()
        self.minimum_person_height = 0.5  # 50% of the image height

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        self.root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter

    def getPredictor(self, name, ckpt):
        model_name = '{}_{}'.format(name, ckpt)
        path = '{}/{}'.format(self.root, self.ConfigDict[model_name])
        if model_name not in self.model_dict:
            if 'segmentation' == name:
                self.model_dict[model_name] = SapiensSegmentation(path)
            if 'depth' == name:
                self.model_dict[model_name] = SapiensDepth(path)
            if 'normal' == name:
                self.model_dict[model_name] = SapiensNormal(path)
        return self.model_dict[model_name]

    def detect(self, bgr, pre_detection=False):
        def filterSmallBoxes(boxes: np.ndarray, img_height: int, height_thres: float = 0.1) -> np.ndarray:
            person_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                person_height = y2 - y1
                if person_height < height_thres * img_height:
                    continue
                person_boxes.append(box)
            return np.array(person_boxes)

        def expandBoxes(boxes: np.ndarray, img_shape, padding: int = 50) -> np.ndarray:
            expanded_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(img_shape[1], x2 + padding)
                y2 = min(img_shape[0], y2 + padding)
                expanded_boxes.append([x1, y1, x2, y2])
            return np.array(expanded_boxes)

        shape = bgr.shape
        if pre_detection is True or isinstance(pre_detection, str):
            # detecting people
            detector = XManager.getModules('human_detection_yolox')(bgr)
            person_boxes = detector.detect(bgr)
            person_boxes = filterSmallBoxes(person_boxes, shape[0], self.minimum_person_height)
            if len(person_boxes) == 0:
                return []
            person_boxes = expandBoxes(person_boxes, shape)
        else:
            person_boxes = [[0, 0, shape[1], shape[0]]]
        return person_boxes

    def inference(self, bgr, output_maps, pre_detection, *args, **kwargs):
        person_boxes = self.detect(bgr, pre_detection)
        result_dict = dict()
        for box in person_boxes:
            crop = bgr[box[1]:box[3], box[0]:box[2]]
            for map_type in output_maps:
                if map_type in self.ConfigDict:
                    name, ckpt = map_type.split('_')
                    if name not in result_dict:
                        result_dict[name] = list()
                    result_dict[name].append(self.getPredictor(name, ckpt)(crop))
        # visual results
        normal_maps = result_dict.pop('normal', [])
        segmentation_maps = result_dict.pop('segmentation', [])
        depth_maps = result_dict.pop('depth', [])
        return person_boxes, (segmentation_maps, depth_maps, normal_maps)

    """
    """
    def _extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        # segmentation, depth, normal
        targets = kwargs.pop('targets', 'segmentation_03b')
        output_maps = targets if isinstance(targets, (list, tuple)) else [targets]
        pre_detection = kwargs.pop('pre_detection', False)
        return targets, dict(output_maps=output_maps, pre_detection=pre_detection)

    def _returnResult(self, person_boxes, output, targets):
        def _formatResult(target):
            segmentation_maps, depth_maps, normal_maps = output
            if target[:12] == 'segmentation':
                return segmentation_maps
            if target[:5] == 'depth':
                return depth_maps
            if target[:6] == 'normal':
                return normal_maps
            if target == 'person_boxes':
                return person_boxes
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, bgr, *args, **kwargs):
        targets, inference_kwargs = self._extractArgs(*args, **kwargs)
        person_boxes, output = self.inference(bgr, **inference_kwargs)
        return self._returnResult(person_boxes, output, targets)
