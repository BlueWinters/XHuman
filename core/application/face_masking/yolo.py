
import copy
import json
import os
import cv2
import numpy as np


class YOLOResult:
    """
    """
    NMS_Threshold = 0.5
    Score_Threshold = 0.5

    """
    """
    def __init__(self, frame_index, result):
        self._frame_index = frame_index
        # refine by nms
        classify, scores, identity, boxes, points = YOLOResult.refineByNMS(result)
        self._classify = classify
        self._boxes = boxes
        self._points = points
        self._scores = scores
        self._identity = identity

    def __len__(self):
        return len(self.score)

    @property
    def classify(self):
        return self._classify

    @property
    def boxes(self):
        return self._boxes

    @property
    def points(self):
        return self._points

    @property
    def score(self):
        return self._scores

    @property
    def identity(self):
        return self._identity

    """
    """
    @staticmethod
    def doNonMaximumSuppression(scores, boxes, nms_threshold=0.7):
        """Pure Python NMS baseline."""
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
            index = np.where(ovr <= nms_threshold)[0]
            order = order[index + 1]
        return keep

    @staticmethod
    def doNonMaximumSuppressionPlus(scores, boxes, points, order=None, nms_threshold=0.7):
        """Pure Python NMS baseline."""
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = order if isinstance(order, np.ndarray) else scores.argsort()[::-1]
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
            index = np.where(ovr <= nms_threshold)[0]
            order = YOLOResult.updateIndexByPoints(i, order[index + 1], ovr[index], points)
        return keep

    @staticmethod
    def getOrder(scores, identity_list_cur, identity_list_pre):
        if len(identity_list_pre) > 0:
            order = scores.argsort()[::-1]
            order_list = np.array(order).tolist()
            index_true = [each for each in order_list if identity_list_cur[each] in identity_list_pre]
            index_false = [each for each in order_list if identity_list_cur[each] not in identity_list_pre]
            return np.array([*index_true, *index_false], dtype=order.dtype)
        return None

    @staticmethod
    def updateIndexByPoints(i, index, over, points):
        points17 = points[i, :, :]
        remain = []
        for n, iou in zip(index, over):
            if float(iou) > 0:
                if YOLOResult.isOverLap(points17, points[n, :, :]) is True:
                    continue
            remain.append(n)
        return np.array(remain, dtype=index.dtype)

    @staticmethod
    def isOverLap(points1, points2, ratio=0.3):
        scores = np.round(points1[:, 2]).astype(np.float32) * np.round(points2[:, 2]).astype(np.float32)
        index = np.nonzero(scores)[0]  # K,
        distance = np.mean(np.abs(points1[index, 0:2] - points2[index, 0:2]), axis=1)  # K,2 --> K,
        good_points_index = np.where(np.ceil(distance) < 5)[0]
        return bool(len(good_points_index) >= int(ratio * len(index) + 0.5))

    @staticmethod
    def refineByNMS(result, identity_list_pre):
        classify = np.reshape(np.round(result.boxes.cls.cpu().numpy()).astype(np.int32), (-1,))
        boxes = np.reshape(np.round(result.boxes.xyxy.cpu().numpy()).astype(np.int32), (-1, 4,))
        points = np.reshape(result.keypoints.data.cpu().numpy().astype(np.float32), (-1, 17, 3))
        scores = np.reshape(result.boxes.conf.cpu().numpy().astype(np.float32), (-1,))
        identity = np.reshape(result.boxes.id.cpu().numpy().astype(np.int32), (-1,))
        order = YOLOResult.getOrder(scores, identity, identity_list_pre)  # reset nms order
        keep = YOLOResult.doNonMaximumSuppressionPlus(scores, boxes, points, order, YOLOResult.NMS_Threshold)
        return classify[keep], scores[keep], identity[keep], boxes[keep], points[keep]
        # return classify, scores, identity, boxes, points

    """
    """
    @staticmethod
    def saveTrackingOfflineResult(path_video_in, path_json_out):
        def formatAsString(array):
            assert isinstance(array, np.ndarray)
            if len(array.shape) == 1:
                return str(array.tolist())
            if len(array.shape) == 2:
                return [str(each) for each in array.tolist()]
            if len(array.shape) == 3:
                return [formatAsString(each) for each in array]

        parameters = dict(persist=True, conf=0.3, iou=0.7, classes=[0], tracker='bytetrack.yaml', verbose=False)
        result = yolo_model.track(path_video_in, **parameters)
        # save as json
        data = []
        for n in range(len(result)):
            if result[n].boxes.id is None:
                continue
            identity = np.reshape(result[n].boxes.id.cpu().numpy().astype(np.int32), (-1,))
            classify = np.reshape(np.round(result[n].boxes.cls.cpu().numpy()).astype(np.int32), (-1,))
            scores = np.reshape(result[n].boxes.conf.cpu().numpy().astype(np.float32), (-1,))
            boxes = np.reshape(np.round(result[n].boxes.xyxy.cpu().numpy()).astype(np.int32), (-1, 4,))
            points = np.reshape(result[n].keypoints.data.cpu().numpy().astype(np.float32), (-1, 17, 3))
            data.append(dict(
                identity=['{:d}'.format(v) for v in identity.tolist()],
                # classify=int(classify),
                scores=['{:.2f}'.format(v) for v in scores.tolist()],
                boxes=formatAsString(boxes),
                points=formatAsString(points),
            ))
        json.dump(data, open(path_json_out, 'w'), indent=4)


if __name__ == '__main__':
    import ultralytics
    yolo_model = ultralytics.YOLO('X:/checkpoints/ultralytics/yolo11m-pose')
    path_video_in = R'N:\archive\2024\1126-video\DanceShow4\input.mp4'
    path_json_out = R'N:\archive\2024\1126-video\DanceShow4\input-yolo_result.json'
    YOLOResult.saveTrackingOfflineResult(path_video_in, path_json_out)
