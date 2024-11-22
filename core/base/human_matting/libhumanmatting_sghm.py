
import logging
import cv2
import numpy as np
import torchvision
from PIL import Image
from .libhumanmatting_base import LibHumanMatting_Wrapper
from ... import XManager


class LibHumanMatting_SGHM(LibHumanMatting_Wrapper):
    """
    """
    @staticmethod
    def getResources():
        return [
            LibHumanMatting_SGHM.EngineConfig['parameters'],
        ]

    EngineConfig = {
        'type': 'torch',
        'device': 'cuda:0',
        'parameters': 'base/human_matting_sghm.ts'
    }

    """
    """
    def __init__(self, *args, **kwargs):
        super(LibHumanMatting_SGHM, self).__init__(*args, **kwargs)
        self.engine = XManager.createEngine(self.EngineConfig)
        self.pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    """
    """
    def initialize(self, *args, **kwargs):
        self.engine.initialize(*args, **kwargs)

    def getUnknownTensorFromPred(self, pred, rand_width=30, train_mode=True):
        # pred: N, 1 ,H, W
        Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]
        N, C, H, W = pred.shape
        # pred = pred.data.cpu().numpy()
        uncertain_area = np.ones_like(pred, dtype=np.uint8)
        uncertain_area[pred < 1.0 / 255.0] = 0
        uncertain_area[pred > 1 - 1.0 / 255.0] = 0

        for n in range(N):
            uncertain_area_ = uncertain_area[n, 0, :, :]  # H, W
            if train_mode:
                width = np.random.randint(1, rand_width)
            else:
                width = rand_width // 2
            uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
            uncertain_area[n, 0, :, :] = uncertain_area_
        weight = np.zeros_like(uncertain_area)
        weight[uncertain_area == 1] = 1
        # weight = torch.from_numpy(weight).cuda()
        return weight

    def inference(self, source):
        infer_size = 1280
        image = Image.fromarray(source).convert('RGB')
        h = image.height
        w = image.width
        if w >= h:
            rh = infer_size
            rw = int(w / h * infer_size)
        else:
            rw = infer_size
            rh = int(h / w * infer_size)
        rh = rh - rh % 64
        rw = rw - rw % 64
        image = image.resize(size=(rw, rh), resample=Image.Resampling.BILINEAR)

        image = self.pil_to_tensor(image)
        input_tensor = image[None, :, :, :].to(self.device)

        pred = self.engine(input_tensor)

        # progressive refine alpha
        alpha_pred_os1 = pred['alpha_os1']
        alpha_pred_os4 = pred['alpha_os4']
        alpha_pred_os8 = pred['alpha_os8']
        pred_alpha = np.copy(alpha_pred_os8)
        weight_os4 = self.getUnknownTensorFromPred(pred_alpha, rand_width=30, train_mode=False)
        pred_alpha[weight_os4 > 0] = alpha_pred_os4[weight_os4 > 0]
        weight_os1 = self.getUnknownTensorFromPred(pred_alpha, rand_width=15, train_mode=False)
        pred_alpha[weight_os1 > 0] = alpha_pred_os1[weight_os1 > 0]
        pred_alpha = np.transpose(pred_alpha[0], axes=(1, 2, 0))
        alpha_np = cv2.resize(pred_alpha, (w, h), interpolation=cv2.INTER_LINEAR)

        # output segment
        pred_segment = np.transpose(pred['segment'][0], axes=(1, 2, 0))
        pred_segment = cv2.resize(pred_segment, (w, h), interpolation=cv2.INTER_LINEAR)
        segment_np = pred_segment.data.cpu().numpy()
        return alpha_np, segment_np
