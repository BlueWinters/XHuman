
import logging
import os
import cv2
import onnxruntime
import numpy as np
import supervision


class LibHeadDetectionTracker:
    def __init__(self, *args, **kwargs):
        self.detector = None
        self.tracker = supervision.ByteTrack()
        self.annotator_box = supervision.BoxAnnotator()
        self.annotator_label = supervision.LabelAnnotator()

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        pass

    def detect(self, bgr):
        if self.detector is None:
            self.detector = LibHeadDetection()
            self.detector.initialize(root='/home/ranger/Documents/xhuman/checkpoints')
        return self.detector(bgr)

    def inference(self, frame_bgr: np.ndarray):
        scores, boxes = self.detect(frame_bgr)
        if len(scores) > 0:
            class_id = np.zeros_like(scores, dtype=np.int32)
            detections = supervision.Detections(xyxy=boxes, confidence=scores, class_id=class_id)
            detections = self.tracker.update_with_detections(detections)
            return detections
        return None

    def visual(self, frame_bgr, detections):
        if detections is not None:
            labels = [f'#{tracker_id}' for tracker_id in detections.tracker_id]
            annotated_frame_bgr = self.annotator_box.annotate(frame_bgr.copy(), detections)
            annotated_frame_bgr = self.annotator_label.annotate(annotated_frame_bgr, detections, labels)
            return annotated_frame_bgr
        return frame_bgr


def track():
    from core.utils import XVideoReader, XVideoWriter
    tracker = LibHeadDetectionTracker()
    reader = XVideoReader('/home/ranger/Documents/xhuman/cache/video/27/input.mp4')
    config = reader.desc(True)
    config['backend'] = 'ffmpeg'
    writer = XVideoWriter(config)
    writer.open('/home/ranger/Documents/xhuman/cache/video/27/head-tracking-tiny.mp4')
    for n, frame in enumerate(reader):
        # frame = cv2.rotate(frame, 90)
        detections = tracker.inference(frame)
        visual = tracker.visual(frame, detections)
        writer.write(visual)
    writer.release()