
import logging
import copy
import os
import typing
import cv2
import numpy as np
import functools
from ..helper.align_helper import AlignHelper
from ....utils import Colors


class Visor:
    @staticmethod
    def getVisColor(index):
        return Colors.getColor(index)

    @staticmethod
    def visualEachSkeleton(canvas, key_points_xy, key_points_score, color, n1, n2, threshold_score=0.5):
        if key_points_score[n1] > threshold_score and key_points_score[n2] > threshold_score:
            cv2.line(canvas, key_points_xy[n1], key_points_xy[n2], color, 2)

    @staticmethod
    def visualSinglePerson(canvas: np.ndarray, identity, box_track, box_face=None, key_points=None, suffix=''):
        color = Visor.getVisColor(identity)
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
                cv2.line(canvas, box_face[0], box_face[1], (255, 255, 255), 2)
                cv2.line(canvas, box_face[1], box_face[2], color, 2)
                cv2.line(canvas, box_face[2], box_face[3], color, 2)
                cv2.line(canvas, box_face[3], box_face[0], color, 2)
        if isinstance(key_points, np.ndarray) and key_points.shape == (68, 2):
            for n in range(68):
                x, y = key_points[n, :].tolist()
                position = (int(round(x)), int(round(y)))
                cv2.circle(canvas, position, 2, color)
        if isinstance(key_points, np.ndarray) and key_points.shape == (5, 3):
            key_points_xy, key_points_score = np.round(key_points[:, :2]).astype(np.int32), key_points[:, 2]
            Visor.visualEachSkeleton(canvas, key_points_xy, key_points_score, color, 0, 1)
            Visor.visualEachSkeleton(canvas, key_points_xy, key_points_score, color, 0, 2)
            Visor.visualEachSkeleton(canvas, key_points_xy, key_points_score, color, 1, 3)
            Visor.visualEachSkeleton(canvas, key_points_xy, key_points_score, color, 2, 4)
        label = '{}-{}'.format(identity, suffix) if len(suffix) > 0 else str(identity)
        box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]
        outside = point1[1] - box_height - 3 >= 0  # label fits outside box
        point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
        # add bounding box text
        cv2.rectangle(canvas, point1, point2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(canvas, label, (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
                    0, text_size, (255, 255, 255), thickness=text_th)
        return canvas

    @staticmethod
    def visualSinglePersonFromInfoFrame(frame_canvas: np.ndarray, person, info_frame, vis_box_rot, vis_key_points):
        visual_function = functools.partial(
            Visor.visualSinglePerson, canvas=frame_canvas, identity=person.identity, box_track=info_frame.box_track,
            key_points=info_frame.key_points if vis_key_points is True else None, suffix='yid_{}'.format(person.yolo_identity))
        if vis_box_rot is True:
            key_points = np.concatenate([info_frame.key_points_xy, info_frame.key_points_score[:, None]], axis=1)
            box_face, box_face_rot = AlignHelper.transformPoints2FaceBox(frame_canvas, key_points, None)
            frame_canvas = visual_function(box_face=box_face_rot)
        else:
            frame_canvas = visual_function(box_face=info_frame.box_face)
        return frame_canvas

    """
    """
    @staticmethod
    def visualSinglePlate(canvas: np.ndarray, identity, suffix, box):
        color = Colors.getColor(identity, True)
        rect_th = max(round(sum(canvas.shape) / 2 * 0.003), 2)
        text_th = max(rect_th - 1, 1)
        text_size = rect_th / 4
        assert isinstance(box, np.ndarray) and len(box) == 4
        box_tracker = np.array(box).astype(np.int32)
        point1 = np.array([box_tracker[0], box_tracker[1]], dtype=np.int32)
        point2 = np.array([box_tracker[2], box_tracker[3]], dtype=np.int32)
        canvas = cv2.rectangle(canvas, point1, point2, color, 2)
        string = '{}-{}'.format(str(identity), str(suffix))
        box_width, box_height = cv2.getTextSize(string, 0, fontScale=text_size, thickness=text_th)[0]
        outside = point1[1] - box_height - 3 >= 0  # label fits outside box
        point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
        # add bounding box text
        cv2.rectangle(canvas, point1, point2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(canvas, string, (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
                    0, text_size, (255, 255, 255), thickness=text_th)
        return canvas

