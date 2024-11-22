
import os
import enum
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .common import create_preprocessor



class SapiensNormal:
    """
    """
    @staticmethod
    def visualNormalMap(normal_map: np.ndarray) -> np.ndarray:
        # Normalize the normal map
        normal_map_norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
        normal_map_normalized = normal_map / (normal_map_norm + 1e-5)  # Add a small epsilon to avoid division by zero
        normal_map = ((normal_map_normalized + 1) / 2 * 255).astype(np.uint8)
        # Convert to BGR
        return cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR)

    @staticmethod
    def postprocess(results, img_shape) -> np.ndarray:
        result = results[0].detach().cpu()
        # Upsample the result to the original image size
        logits = F.interpolate(result.unsqueeze(0), size=img_shape, mode="bilinear").squeeze(0)
        # Covert to numpy array
        normal_map = logits.float().numpy().transpose(1, 2, 0)
        return normal_map

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
        normals = self.postprocess(results, bgr.shape[:2])
        return normals
