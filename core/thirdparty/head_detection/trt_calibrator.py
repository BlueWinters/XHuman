"""
build example:
https://github.com/NVIDIA/TensorRT/tree/release/10.9/samples/python
"""

import cv2
import numpy as np
from ...utils.tensorrt import EntropyCalibrator
from .libheaddetection import LibHeadDetectionInterface


class EntropyCalibratorLibHeadDetection(EntropyCalibrator):
    def __init__(self, path_images):
        super(EntropyCalibratorLibHeadDetection, self).__init__((8, 3, 640, 640), path_images)

    def transform(self, image: np.ndarray, *args, **kwargs):
        # Normalization + BGR->RGB
        resized_image, padding = LibHeadDetectionInterface.formatSizeWithPaddingForward(image, self.height, self.width)
        fmt_h, fmt_w = resized_image.shape[0], resized_image.shape[1]
        format_image = np.divide(resized_image, 255.0)
        format_image = format_image[..., ::-1]
        format_image = format_image.transpose((2, 0, 1))
        format_image = np.ascontiguousarray(format_image, dtype=np.float32)
        return np.asarray([format_image], dtype=np.float32)

