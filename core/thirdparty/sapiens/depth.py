
import os
import enum
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .common import create_preprocessor



class SapiensDepth:
    """
    """
    @staticmethod
    def visualDepthMap(depth_map: np.ndarray) -> np.ndarray:
        min_depth, max_depth = np.min(depth_map), np.max(depth_map)

        norm_depth_map = 1 - (depth_map - min_depth) / (max_depth - min_depth)
        norm_depth_map = (norm_depth_map * 255).astype(np.uint8)

        # Normalize and color the image
        color_depth = cv2.applyColorMap(norm_depth_map, cv2.COLORMAP_INFERNO)
        color_depth[depth_map == 0] = 128
        return color_depth

    @staticmethod
    def postprocess(results, shape) -> np.ndarray:
        result = results[0].cpu()

        # Upsample the result to the original image size
        logits = F.interpolate(result.unsqueeze(0), size=shape, mode="bilinear").squeeze(0)

        # Covert to numpy array
        depth_map = logits.float().numpy().squeeze()
        return depth_map

    """
    """
    """
    """
    def __init__(self, path, **kwargs):
        self.device = kwargs['device'] if 'device' in kwargs else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = kwargs['dtype'] if 'dtype' in kwargs else torch.float32
        self.preprocessor = create_preprocessor(input_size=(1024, 768))  # Only these values seem to work well
        self.model = self.initialize(path)

    def initialize(self, path, *arg, **kwargs):
        assert os.path.exists(path), path
        return torch.jit.load(path).eval().to(self.device).to(self.dtype)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        # Model expects BGR, but we change to RGB here because the preprocessor will switch the channels also
        input = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocessor(input).to(self.device).to(self.dtype)
        with torch.inference_mode():
            results = self.model(tensor)
        depth_map = self.postprocess(results, bgr.shape[:2])
        return depth_map
