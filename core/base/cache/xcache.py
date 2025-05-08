
import numpy as np


class XCache:
    """
    """
    def __init__(self, **kwargs):
        self._bgr = np.copy(kwargs['bgr'])
        self._shape = self._bgr.shape
        self._super = kwargs.pop('super', None)

    @property
    def shape(self):
        return self._shape[:2]

    @property
    def channel(self):
        return 1 if len(self._shape) == 2 \
            else self._shape[2]

    @property
    def bgr(self):
        return self._bgr

    @property
    def rgb(self):
        return self._bgr[:, :, ::-1]
