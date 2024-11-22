
import logging
import numpy as np
import cv2
from .. import XManager



class LibFaceAttribute:
    """
    Sex: 0-Male, 1-Female
    Age: [0,116]
    """
    @staticmethod
    def getResources():
        return [
            LibFaceAttribute.EngineConfig['parameters'],
        ]

    """
    """
    EngineConfig = {
        'type': 'torch',
        'device': 'cuda:0',
        'parameters': 'base/face_attribute.ts'
    }

    """
    """
    def __init__(self, *args, **kwargs):
        self.engine = XManager.createEngine(self.EngineConfig)
        self.size = 128
        self.max_age = 116
        self.rank = np.arange(self.max_age).astype(np.float32)
        # self.ratio = (-0.25, -0.55, 1.5, 1.7)
        self.ratio = [(-0.1, -0.5, 1.2, 1.6), (-0.25, -0.5, 1.5, 1.6)]
        self.to_text = lambda sex: 'M' if int(sex) == 0 else 'F'
        self.to_value = lambda age: float(np.sum(age * self.rank))

    def __del__(self):
        logging.warning('delete module {}'.format(self.__class__.__name__))

    def initialize(self, *args, **kwargs):
        self.engine.initialize(*args, **kwargs)

    """
    """
    def _cropFaceByLandmark(self, bgr, landmark):
        h, w, c = bgr.shape
        maxwh = max(h, w)
        ratio = self.ratio[0] if maxwh > 4000 else self.ratio[1]
        xmin = landmark[0, 0]
        ymin = landmark[0, 1]
        xmax = landmark[0, 0]
        ymax = landmark[0, 1]
        for i in range(0, 68):
            if landmark[i, 0] >= xmax: xmax = landmark[i, 0]
            if landmark[i, 0] < xmin: xmin = landmark[i, 0]
            if landmark[i, 1] >= ymax: ymax = landmark[i, 1]
            if landmark[i, 1] < ymin: ymin = landmark[i, 1]
        bw = xmax - xmin
        bh = ymax - ymin
        nbx1 = xmin + ratio[0] * bw
        nby1 = ymin + ratio[1] * bh
        nbx2 = nbx1 + ratio[2] * bw
        nby2 = nby1 + ratio[3] * bh
        rx1 = round(int(max(nbx1, 0)))
        ry1 = round(int(max(nby1, 0)))
        rx2 = round(int(min(nbx2, w)))
        ry2 = round(int(min(nby2, h)))
        return bgr[ry1:ry2, rx1:rx2, :]

    def _formatSize(self, crop, size, color=(0, 0, 0)):
        h0, w0 = crop.shape[:2]  # orig hw
        r = size / max(h0, w0)
        im = cv2.resize(crop, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        shape = im.shape[:2]
        # Compute padding
        dh = size - shape[0]
        dw = size - shape[1]
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im

    def _estimateandmarks(self, bgr, landmarks):
        if landmarks is None:
            module = XManager.getModules('face_landmark_v1')
            landmarks = module(bgr)
        landmarks = np.reshape(landmarks, (-1, 68, 2))
        return landmarks

    def _format(self, bgr, landmarks):
        landmarks = self._estimateandmarks(bgr, landmarks)
        list_format = list()
        for n, each in enumerate(landmarks):
            assert isinstance(each, np.ndarray), type(each)
            crop = self._cropFaceByLandmark(bgr, each)
            format = self._formatSize(crop, size=self.size, color=(0, 0, 0))
            list_format.append(np.transpose(np.float32(format), axes=(2, 0, 1)) / 255.)
        return np.reshape(np.array(list_format), (-1, 3, self.size, self.size))

    def _post(self, age_array, sex_array):
        age = [self.to_value(age_array[n, :]) for n in range(len(age_array))]
        return np.reshape(sex_array, (-1, 2)), \
            np.reshape(np.round(age).astype(np.int32), (-1,))

    def inference(self, bgr, landmarks):
        batch_input = self._format(bgr, landmarks)
        age_array, sex_array = self.engine.inference(batch_input)
        return self._post(age_array, sex_array)

    """
    """
    def _extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        landmarks = kwargs.pop('landmarks', None)
        inference_kwargs = dict(landmarks=landmarks)
        return targets, inference_kwargs

    def _returnResult(self, sex, age, targets):
        def _formatResult(target):
            if target == 'source':
                # 0 is male, 1 is female
                sex_format = [np.argmax(sex[n, :]) for n in range(len(sex))]
                return np.array(sex_format, dtype=np.int32), age
            if target == 'sex-string':
                return [self.to_text(np.argmax(sex[n, :])) for n in range(len(sex))]
            if target == 'age-array':
                return age
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, bgr, *args, **kwargs):
        targets, inference_kwargs = self._extractArgs(*args, **kwargs)
        sex, age = self.inference(bgr, **inference_kwargs)
        return self._returnResult(sex, age, targets)