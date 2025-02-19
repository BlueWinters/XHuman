
from .libmasking_image import LibMaskingImage
from .libmasking_video import LibMaskingVideo
from .masking_function import MaskingFunction


class LibMasking:
    """
    just interface for api
    """
    @staticmethod
    def scanningImage(*args, **kwargs):
        return LibMaskingImage.scanningImage(*args, **kwargs)

    @staticmethod
    def maskingImage(*args, **kwargs):
        return LibMaskingImage.maskingImage(*args, **kwargs)

    """
    """
    @staticmethod
    def scanningVideo(*args, **kwargs):
        return LibMaskingVideo.scanningVideo(*args, **kwargs)

    @staticmethod
    def maskingVideo(*args, **kwargs):
        return LibMaskingVideo.maskingVideo(*args, **kwargs)

    @staticmethod
    def getMaskingOption(*args, **kwargs):
        return MaskingFunction.getMaskingOption(*args, **kwargs)
