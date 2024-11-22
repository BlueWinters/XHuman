
import cv2
import numpy as np
import skimage
from .xportrait import XPortrait


"""
"""
def getEyesLength(xcache) -> float:
    assert isinstance(xcache, XPortrait), type(xcache)
    lft_eye_len = [np.linalg.norm(xcache.landmark[n][36, :] - xcache.landmark[n][39, :]) for n in range(xcache.number)]
    rig_eye_len = [np.linalg.norm(xcache.landmark[n][42, :] - xcache.landmark[n][45, :]) for n in range(xcache.number)]
    return lft_eye_len, rig_eye_len

def getEyesMeanLength(xcache):
    lft_eye_len, rig_eye_len = getEyesLength(xcache)
    return [(lft_len + rig_len) / 2. for lft_len, rig_len in zip(lft_eye_len, rig_eye_len)]

def getAjna(xcache):
    assert isinstance(xcache, XPortrait), type(xcache)
    return [np.mean(xcache.landmark[n][17:27, :], axis=0) for n in range(xcache.number)]

def getCenterOfEyes(xcache):
    assert isinstance(xcache, XPortrait), type(xcache)
    return [np.mean(xcache.landmark[n][36:48, :], axis=0) for n in range(xcache.number)]
