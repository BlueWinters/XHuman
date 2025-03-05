
from .libmasking_image import LibMaskingImage
from .libmasking_video import LibMaskingVideo
from .masking_options import MaskingOptions


class LibMasking:
    """
    """
    def __init__(self, *args, **kwargs):
        pass

    def __del__(self):
        pass

    def initialize(self, *args, **kwargs):
        pass

    """
    just interface for api
    """
    @staticmethod
    def scanningImage(*args, **kwargs) -> LibMaskingImage:
        masking_image = LibMaskingImage()
        masking_image.scanningImage(*args, **kwargs)
        return masking_image

    @staticmethod
    def maskingImage(*args, **kwargs):
        return LibMaskingImage.maskingImage(*args, **kwargs)

    """
    """
    @staticmethod
    def scanningVideo(*args, **kwargs) -> LibMaskingVideo:
        masking_video = LibMaskingVideo()
        masking_video.scanningVideo(*args, **kwargs)
        return masking_video

    @staticmethod
    def maskingVideo(*args, **kwargs):
        return LibMaskingVideo.maskingVideo(*args, **kwargs)

    """
    """
    @staticmethod
    def getMaskingOption(*args, **kwargs):
        return MaskingOptions.getMaskingOption(*args, **kwargs)
