
import numpy as np
from .s2b import finer
from .estimate_foreground_ml import estimate_foreground_ml
from .estimatefb import estimate_foreground_cf


def estimateForeground(bgr, alpha, use_cf=False):
    bgr_big = bgr.astype(np.float64) / 255
    alpha_big = alpha.astype(np.float64) / 255
    foreground = estimate_foreground_cf(bgr_big, alpha_big, return_background=False) \
        if use_cf is True else estimate_foreground_ml(bgr_big, alpha_big, return_background=False)
    foreground = np.clip(np.round(foreground * 255), 0, 255).astype(np.uint8)
    return foreground

def estimateComposite(alpha, foreground, background):
    alpha_format = alpha[:, :, np.newaxis].astype(np.float32) / 255
    foreground_format = foreground.astype(np.float32) / 255
    composite = foreground_format * alpha_format + (1 - alpha_format) * background  # [0, 0, 1]
    composite = np.clip(np.round(composite * 255), 0, 255).astype(np.uint8)
    return composite

def formatBackground(bgr, background):
    if len(background) == 3:
        color = np.array(background, dtype=np.uint8)
        background = np.ones_like(bgr) * np.reshape(color, (1, 1, 3))
    assert isinstance(background, np.ndarray), type(background)
    assert background.shape == bgr.shape
    background = background.astype(np.float32)
    if np.max(background) > 1:
        background = np.clip(background / 255, 0, 1)
    return background