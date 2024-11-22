
import os
import enum
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .common import create_preprocessor



random = np.random.RandomState(11)
classes = ["Background", "Apparel", "Face Neck", "Hair", "Left Foot", "Left Hand", "Left Lower Arm", "Left Lower Leg",
           "Left Shoe", "Left Sock", "Left Upper Arm", "Left Upper Leg", "Lower Clothing", "Right Foot", "Right Hand",
           "Right Lower Arm", "Right Lower Leg", "Right Shoe", "Right Sock", "Right Upper Arm", "Right Upper Leg",
           "Torso", "Upper Clothing", "Lower Lip", "Upper Lip", "Lower Teeth", "Upper Teeth", "Tongue"]

colors = random.randint(0, 255, (len(classes) - 1, 3))
colors = np.vstack((np.array([128, 128, 128]), colors)).astype(np.uint8)  # Add background color
colors = colors[:, ::-1]



class SapiensSegmentation():
    """
    """
    @staticmethod
    def visualSegmentationMap(segmentation_map: np.ndarray) -> np.ndarray:
        h, w = segmentation_map.shape
        segmentation_img = np.zeros((h, w, 3), dtype=np.uint8)
        for i, color in enumerate(colors):
            segmentation_img[segmentation_map == i] = color
        return segmentation_img

    @staticmethod
    def postprocess(results, shape) -> np.ndarray:
        result = results[0].cpu()
        # Upsample the result to the original image size
        logits = F.interpolate(result.unsqueeze(0), size=shape, mode="bilinear").squeeze(0)
        # Perform argmax to get the segmentation map
        segmentation_map = logits.argmax(dim=0, keepdim=True)
        # Covert to numpy array
        segmentation_map = segmentation_map.float().numpy().squeeze()
        return segmentation_map

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
        segmentation_map = self.postprocess(results, bgr.shape[:2])
        return segmentation_map
