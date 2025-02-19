
import logging
import copy
import os
import typing
import cv2
import numpy as np
from ....utils import Colors


class ScanningVisor:
    @staticmethod
    def getVisColor(index):
        return Colors.getColor(index)

    @staticmethod
    def visualSinglePerson(canvas: np.ndarray, identity, box_track, box_face=None):
        color = ScanningVisor.getVisColor(identity)
        rect_th = max(round(sum(canvas.shape) / 2 * 0.003), 2)
        text_th = max(rect_th - 1, 1)
        text_size = rect_th / 4
        assert isinstance(box_track, np.ndarray) and len(box_track) == 4
        box_tracker = np.array(box_track).astype(np.int32)
        point1 = np.array([box_tracker[0], box_tracker[1]], dtype=np.int32)
        point2 = np.array([box_tracker[2], box_tracker[3]], dtype=np.int32)
        canvas = cv2.rectangle(canvas, point1, point2, color, 2)
        if isinstance(box_face, np.ndarray) and len(box_face) == 4:
            if len(box_face.shape) == 1:
                box_face = np.array(box_face).astype(np.int32)
                point1_face = np.array([box_face[0], box_face[1]], dtype=np.int32)
                point2_face = np.array([box_face[2], box_face[3]], dtype=np.int32)
                canvas = cv2.rectangle(canvas, point1_face, point2_face, color, 1)
            if len(box_face.shape) == 2 and box_face.shape[1] == 2:
                box_face = np.array(box_face).astype(np.int32)
                cv2.line(canvas, box_face[0], box_face[1], color, 1)
                cv2.line(canvas, box_face[1], box_face[2], color, 1)
                cv2.line(canvas, box_face[2], box_face[3], color, 1)
                cv2.line(canvas, box_face[3], box_face[0], color, 1)
        label = str(identity)
        box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]
        outside = point1[1] - box_height - 3 >= 0  # label fits outside box
        point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
        # add bounding box text
        cv2.rectangle(canvas, point1, point2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(canvas, label, (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
                    0, text_size, (255, 255, 255), thickness=text_th)
        return canvas
