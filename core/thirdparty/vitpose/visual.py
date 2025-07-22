
import numpy as np
import cv2


Joints = {
    "coco_25": {
        "keypoints": {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "neck",
            6: "left_shoulder",
            7: "right_shoulder",
            8: "left_elbow",
            9: "right_elbow",
            10: "left_wrist",
            11: "right_wrist",
            12: "left_hip",
            13: "right_hip",
            14: "hip",
            15: "left_knee",
            16: "right_knee",
            17: "left_ankle",
            18: "right_ankle",
            19: "left_big_toe",
            20: "left_small_toe",
            21: "left_heel",
            22: "right_big_toe",
            23: "right_small_toe",
            24: "right_heel",
        },
        "skeleton": [
            [17, 15], [15, 12], [18, 16], [16, 13], [12, 14], [13, 14], [5, 14],
            [6, 5], [7, 5], [6, 8], [7, 9], [8, 10], [9, 11], [1, 2], [0, 1], [0, 2],
            [1, 3], [2, 4], [17, 21], [18, 24], [19, 20], [22, 23], [19, 21],
            [22, 24], [5, 0]
        ]
    }
}


def visualSkeleton(canvas, key_points_xy, key_points_score, color, n1, n2, threshold_score=0.5):
    size = max(1, min(canvas.shape[:2]) // 150)
    if key_points_score[n1] > threshold_score and key_points_score[n2] > threshold_score:
        cv2.line(canvas, key_points_xy[n1], key_points_xy[n2], color, 2)
    if key_points_score[n1] > threshold_score:
        cv2.circle(canvas, key_points_xy[n1], size, color, -1)
    if key_points_score[n2] > threshold_score:
        cv2.circle(canvas, key_points_xy[n2], size, color, -1)


def visualEachPerson(canvas_bgr, box, key_points, color, dataset='coco_25'):
    assert isinstance(box, np.ndarray) and len(box) == 4
    box = np.array(box).astype(np.int32)
    point1 = np.array([box[0], box[1]], dtype=np.int32)
    point2 = np.array([box[2], box[3]], dtype=np.int32)
    thickness = max(max(canvas_bgr.shape[:2]) // 1000, 2)
    canvas_bgr = cv2.rectangle(canvas_bgr, point1, point2, color, thickness)
    key_points_xy, key_points_score = np.round(key_points[:, :2]).astype(np.int32), key_points[:, 2]
    skeleton = Joints[dataset]['skeleton']
    for i, joint in enumerate(skeleton):
        visualSkeleton(canvas_bgr, key_points_xy, key_points_score, color, joint[0], joint[1])
    return canvas_bgr

