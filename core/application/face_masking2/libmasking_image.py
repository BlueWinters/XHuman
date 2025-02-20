
import logging
import os
import cv2
import numpy as np
from .masking_function import MaskingFunction
from .helper.masking_helper import MaskingHelper
from .scanning.scanning_image import InfoImage, InfoImage_Person
from ...utils import Resource, XContextTimer


class LibMaskingImage:
    """
    image masking
    """
    @staticmethod
    def scanningImage(bgr, **kwargs) -> InfoImage:
        path_out_json = kwargs.pop('path_out_json', None) or Resource.createRandomCacheFileName('.json')
        schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
        path_out_image = kwargs.pop('visual_scanning', None)
        # main pipeline
        info_image = InfoImage(bgr)
        info_image.doScanning(schedule_call)
        info_image.saveAsJson(path_out_json, schedule_call)
        info_image.saveVisualScanning(path_out_image)
        return info_image

    @staticmethod
    def maskingImage(path_image_or_bgr, options_dict, **kwargs):
        schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
        parameters = dict(
            path_in_json=kwargs.pop('path_in_json', None),
            video_info_string=kwargs.pop('video_info_string', ''),)
        with_hair = kwargs.pop('with_hair', True)

        with XContextTimer(True):
            info_image = InfoImage.createFromJson(**parameters)
            bgr = cv2.imread(path_image_or_bgr, cv2.IMREAD_COLOR) if isinstance(path_image_or_bgr, str) \
                else np.array(path_image_or_bgr, dtype=np.uint8)
            MaskingHelper.getPortraitMaskingWithInfoImage(
                bgr, info_image, options_dict, with_hair=with_hair, expand=0.5)
            canvas_bgr = np.copy(bgr)
            for n, info_person in enumerate(info_image):
                assert isinstance(info_person, InfoImage_Person)
                if info_person.identity in options_dict:
                    masking_option = options_dict[info_person.identity]
                    canvas_bgr = MaskingFunction.maskingImageFace(bgr, canvas_bgr, info_person, masking_option)
                    schedule_call('打码图片', float((n + 1) / len(info_image)))
            return canvas_bgr



