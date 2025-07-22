
import logging
import cv2
import os
import numpy as np
import ultralytics
import threading
from ..utils import XVideoReader, FFMPEGHelper
from ..geometry import GeoFunction
from .. import XManager


class LibUltralyticsWrapper:
    """
    """
    @staticmethod
    def constructUITab():
        import gradio
        with gradio.Tab('Ultralytics'):
            with gradio.Row():
                model_list = LibUltralyticsWrapper.ModelList
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
            fn=LibUltralyticsWrapper.actionDetectImage,
            inputs=[input_image, dropdown_models],
            outputs=output_image)
        action_run_image_from_video.click(
            fn=LibUltralyticsWrapper.actionDetectImageFromVideo,
            inputs=[input_image_path2, dropdown_models, input_number_index, input_score_threshold],
            outputs=output_image)
        input_number_index.change(
            fn=LibUltralyticsWrapper.actionDetectImageFromVideo,
            inputs=[input_image_path2, dropdown_models, input_number_index, input_score_threshold],
            outputs=output_image)
        action_run_video1.click(
            fn=LibUltralyticsWrapper.actionDetectVideo,
            inputs=[input_video, dropdown_models],
            outputs=output_video)
        action_run_video2.click(
            fn=LibUltralyticsWrapper.actionDetectVideo,
            inputs=[input_video_path2, dropdown_models],
            outputs=output_video)

    @staticmethod
    def actionDetectImage(rgb, model, score_threshold=0.5):
        import supervision
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        canvas_bgr = bgr.copy()
        results = XManager.getModules('ultralytics')[model](bgr, conf=score_threshold)[0]
        detections = supervision.Detections.from_ultralytics(results)
        box_annotator = supervision.BoxAnnotator()
        canvas_bgr = box_annotator.annotate(canvas_bgr, detections)
        if 'pose' in model:
            key_points = supervision.KeyPoints.from_ultralytics(results)
            canvas_bgr = supervision.VertexAnnotator(supervision.Color.GREEN, 10).annotate(canvas_bgr, key_points)
            canvas_bgr = supervision.EdgeAnnotator(supervision.Color.GREEN, 5).annotate(canvas_bgr, key_points)
        if 'seg' in model:
            pass
        return np.copy(canvas_bgr[:, :, ::-1])

    @staticmethod
    def actionDetectVideo(path_video, model):
        import shutil
        from core.utils import Resource
        module = XManager.getModules('ultralytics')[model]
        uuid_name, (path_in_video, path_out_video) = Resource.createRandomCacheFileName(['-in.mp4', '-out.mp4'])
        shutil.copyfile(path_video, path_in_video)
        return None
    
    @staticmethod
    def actionDetectImageFromVideo(path_video, model, n_index, score_threshold):
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
            return LibUltralyticsWrapper.actionDetectImage(bgr[:, :, ::-1], model, score_threshold)
        else:
            return np.zeros(shape=(512, 512, 3), dtype=np.uint8)

    @staticmethod
    def getResources():
        return ['{}/{}.pt'.format(LibUltralyticsWrapper.CheckpointBase, name)
                for name in LibUltralyticsWrapper.ModelList]

    @staticmethod
    def exportAsTensorRT(root, name, mode, **kwargs):
        # demo code for export
        assert mode in ['fp32', 'fp16', 'int8']
        assert name in LibUltralyticsWrapper.ModelList, name
        path_base = '{}/{}'.format(root, LibUltralyticsWrapper.CheckpointBase)
        yolo = ultralytics.YOLO('{}/{}'.format(path_base, name))
        config = dict()
        path_yaml_cache = ''
        if mode == 'fp16':
            config['half'] = True
            config['nms'] = True
        if mode == 'int8':
            import yaml
            config['half'] = False
            config['int8'] = True
            config['nms'] = True
            # https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/datasets
            path_coco = '{}/asset/ultralytics_coco8_pose.yaml'.format(os.path.dirname(__file__))
            yaml_data = yaml.load(open(path_coco, 'r'), Loader=yaml.FullLoader)
            yaml_data['path'] = kwargs.pop('data', yaml_data['path'])
            path_yaml_cache = '{}/asset/ultralytics_coco8_pose_cache.yaml'.format(os.path.dirname(__file__))
            with open(path_yaml_cache, 'w', encoding='utf-8') as file:
                yaml.dump(data=yaml_data, stream=file, allow_unicode=True)
            config['data'] = path_yaml_cache
        yolo.export(format='engine', **config)
        if name.endswith('pt'):
            name = name.replace('.pt', '')
        else:
            name = name
        # rename engine file
        path_src = '{}/{}.engine'.format(path_base, name)
        path_dst = '{}/{}-{}.engine'.format(path_base, name, mode)
        os.rename(path_src, path_dst)
        print('rename engine file: {} -> {}'.format(path_src, path_dst))
        # rename onnx file
        path_onnx = '{}/{}.onnx'.format(path_base, name)
        if os.path.exists(path_onnx) is True:
            print('auto-remove: {}'.format(path_onnx))
            os.remove(path_onnx)
        # remove yaml file
        if os.path.exists(path_yaml_cache) is True:
            print('auto-remove: {}'.format(path_yaml_cache))
            os.remove(path_yaml_cache)
        # remove cache files
        path_trt_cache = '{}/{}.cache'.format(path_base, name)
        if os.path.exists(path_trt_cache) is True:
            print('auto-remove: {}'.format(path_trt_cache))
            os.remove(path_trt_cache)

    """
    """
    CheckpointBase = 'thirdparty/ultralytics'
    ModelList = [
        # official
        'yolo11n',  # 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x',
        'yolo11n-pose', 'yolo11m-pose', 'yolo11x-pose',  # 'yolo11s-pose', 'yolo11m-pose', 'yolo11l-pose',
        'yolo11n-seg', 'yolo11m-seg', 'yolo11x-seg',
        'yolo11x-pose-fp16.engine', 'yolo11x-pose-int8.engine',
        # others
        'yolo8s-plate.pt', 'yolo8s-plate.20250414.pt', 'yolo8s-plate.20250416.pt', 'yolo8s-plate.20250422.pt',
        'yolo11s-plate.20250425.pt', 'yolo11s-plate.20250430.pt'
    ]

    def __init__(self, *args, **kwargs):
        self.model_dict = dict()
        self.root = None

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        if self.root is None:
            root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
            self.root = '{}/{}'.format(root, self.CheckpointBase)

    def getSpecific(self, name):
        assert name in LibUltralyticsWrapper.ModelList, name
        if name not in self.model_dict:
            self.model_dict[name] = ultralytics.YOLO('{}/{}'.format(self.root, name))
            logging.info('load ultralytics model: {}'.format(name))
        return self.model_dict[name]

    def __getitem__(self, name: str, with_lock=True):
        if with_lock is True:
            lock = threading.Lock()
            lock.acquire()
            try:
                return self.getSpecific(name)
            finally:
                lock.release()
        else:
            return self.getSpecific(name)

    def resetTracker(self, name):
        if name in self.model_dict:
            self.model_dict[name].predictor.trackers[0].reset()

    """
    """
    def detect(self, bgr, name, rotations=None, **kwargs):
        detect_kwargs = kwargs.pop('detect_kwargs', dict(classes=[0]))
        assert isinstance(detect_kwargs, dict), detect_kwargs
        scores_collect, boxes_collect, points_collect, masks_collect, angles_collect = [], [], [], [], []
        rotations = rotations if isinstance(rotations, (list, tuple)) else [0]
        for n, rot in enumerate(rotations):
            scores, boxes, angles, points, masks = self.inferenceWithRotation(
                bgr, name, rot, detect_kwargs)
            if len(scores) > 0:
                scores_collect.append(scores)
                boxes_collect.append(boxes)
                points_collect.append(points)
                masks_collect.append(masks)
                angles_collect.append(angles)
        # NMS
        if len(scores_collect) > 0:
            scores = np.concatenate(scores_collect, axis=0)
            boxes = np.concatenate(boxes_collect, axis=0)
            angles = np.concatenate(angles_collect, axis=0)
            points = np.concatenate(points_collect, axis=0) if 'pose' in name else None
            masks = np.concatenate(masks_collect, axis=0) if 'seg' in name else None
            keep = self.doNonMaximumSuppression(scores, boxes, 0.6)
            scores = scores[keep]
            boxes = boxes[keep]
            points = points[keep] if 'pose' in name else None
            masks = masks[keep] if 'seg' in name else None
            angles = angles[keep]
        else:
            h, w, c = bgr.shape
            scores = np.zeros(shape=(0,), dtype=np.float32)
            boxes = np.zeros(shape=(0, 4), dtype=np.int32)
            points = np.zeros(shape=(0, 10), dtype=np.int32) if 'pose' in name else None
            masks = np.zeros(shape=(0, h, w), dtype=np.int32) if 'seg' in name else None
            angles = np.zeros(shape=(0,), dtype=np.int32)
        return scores, boxes, angles, points, masks

    def inferenceWithRotation(self, bgr, name, image_angle, detect_kwargs):
        module = self.getSpecific(name)
        if image_angle == 0:
            result = module(bgr, **detect_kwargs, verbose=False)[0]
            scores, boxes, points, masks = self.detachResult(result)
            angles = np.zeros(shape=len(scores), dtype=np.int32)
            return scores, boxes, angles, points, masks
        if image_angle in GeoFunction.CVRotationDict:
            rot = cv2.rotate(bgr, GeoFunction.CVRotationDict[image_angle])
            result = module(rot, classes=[0], verbose=False)[0]
            scores, boxes, points, masks = self.detachResult(result)
            h, w, c = rot.shape
            angle_back = GeoFunction.rotateBack(image_angle)
            boxes = GeoFunction.rotateBoxes(np.reshape(boxes, (len(scores), 4)), angle_back, h, w)
            boxes = np.reshape(boxes, (len(scores), 4))
            if isinstance(points, np.ndarray):
                points = GeoFunction.rotatePoints(np.reshape(points, (len(scores), -1, 2)), angle_back, h, w)
                points = np.reshape(points, (len(scores), 10))
            if isinstance(masks, np.ndarray):
                masks = np.stack([GeoFunction.rotateImage(mask, angle_back) for mask in masks], axis=0)
            angles = np.ones(shape=len(scores), dtype=np.int32) * image_angle
            return scores, boxes, angles, points, masks
        raise ValueError('angle {} not in [0,90,180,270]'.format(image_angle))

    @staticmethod
    def detachResult(result):
        classify = np.reshape(np.round(result.boxes.cls.cpu().numpy()).astype(np.int32), (-1,))
        scores = np.reshape(result.boxes.conf.cpu().numpy().astype(np.float32), (-1,))
        boxes = np.reshape(np.round(result.boxes.xyxy.cpu().numpy()).astype(np.int32), (-1, 4,))
        points = None
        masks = None
        if result.keypoints is not None:
            points = np.reshape(result.keypoints.data.cpu().numpy().astype(np.float32), (-1, 17, 3))
        if result.masks is not None:
            h, w = result.orig_shape
            masks = np.zeros(shape=(len(classify), h, w), dtype=np.uint8)
            instance_masks = np.round(result.masks.cpu().numpy().data * 255).astype(np.uint8)  # note: C,H,W and [0,1]
            for n in range(len(classify)):
                masks[n, :, :] = cv2.resize(instance_masks[n, :, :], (w, h))
        return scores, boxes, points, masks

    @staticmethod
    def doNonMaximumSuppression(scores, boxes, nms_threshold):
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='build tensorrt')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--mode', type=str, default='int8', required=True)
    parser.add_argument('--data', type=str, default='')
    args = parser.parse_args()
    LibUltralyticsWrapper.exportAsTensorRT(args.root, args.name, args.mode, data=args.data)
