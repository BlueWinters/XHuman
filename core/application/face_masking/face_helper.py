
import cv2
import numpy as np


def getFaceMaskByPoints(cache, n=0, top_line='brow', value=255):
    def getTopLinePoints():
        if top_line == 'brow':
            points_rig = cache.landmark[n][22:27, :][::-1, :]
            points_lft = cache.landmark[n][17:22, :][::-1, :]
            return points_rig, points_lft
        if top_line == 'eye':
            points_rig = cache.landmark[n][42:46, :][::-1, :]
            points_lft = cache.landmark[0][36:40, :][::-1, :]
            return points_rig, points_lft
        if top_line == 'brow-eye':
            points_eye_rig = cache.landmark[n][42:46, :][::-1, :]
            points_eye_lft = cache.landmark[n][36:40, :][::-1, :]
            points_brow_rig = cache.landmark[n][22:26, :][::-1, :]
            points_brow_lft = cache.landmark[n][18:22, :][::-1, :]
            points_rig = np.round((points_eye_rig + points_brow_rig) / 2).astype(np.int32)
            points_lft = np.round((points_eye_lft + points_brow_lft) / 2).astype(np.int32)
            return points_rig, points_lft

    # parsing = cache.parsing
    # mask = np.where(((0 < parsing) & (parsing < 15)) & (parsing != 12), 255, 0).astype(np.uint8)
    # return mask
    points_rig, points_lft = getTopLinePoints()
    points_profile = cache.landmark[n][0:17, :]
    points_all = np.concatenate([points_profile, points_rig, points_lft], axis=0).round().astype(np.int32)
    mask_face = np.zeros(cache.shape, dtype=np.uint8)
    cv2.fillPoly(mask_face, [points_all], (value, value, value))
    return mask_face
