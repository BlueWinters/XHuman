
import logging
import numpy as np
from .scanning import *
from .masking_blur import *
from .masking_mosaic import *
from .masking_sticker import *


class MaskingFunction:
    """
    """
    @staticmethod
    def maskingImageFace(frame_bgr, canvas_bgr, info_person, masking_option, **kwargs) -> np.ndarray:
        if isinstance(masking_option, MaskingBlur):
            return masking_option.inferenceOnMaskingImage(
                frame_bgr, canvas_bgr, angle=info_person.angle, box=info_person.box, mask_info=getattr(info_person, 'mask_info'), **kwargs)
        if isinstance(masking_option, MaskingMosaic):
            return masking_option.inferenceOnMaskingImage(
                frame_bgr, canvas_bgr, angle=info_person.angle, box=info_person.box, mask_info=getattr(info_person, 'mask_info'), **kwargs)
        if isinstance(masking_option, MaskingSticker):
            return masking_option.inferenceOnMaskingImage(
                frame_bgr, canvas_bgr, angle=info_person.angle, box=info_person.box, landmark=info_person.landmark, auto_rot=True, **kwargs)
        raise NotImplementedError(masking_option)

    @staticmethod
    def maskingVideoFace(frame_bgr, canvas_bgr, info_frame, masking_option, **kwargs) -> np.ndarray:
        if isinstance(masking_option, MaskingBlur):
            return masking_option.inferenceOnMaskingVideo(
                frame_bgr, canvas_bgr, info_frame.box_face, info_frame.key_points_xy, info_frame.key_points_score, **kwargs)
        if isinstance(masking_option, MaskingMosaic):
            return masking_option.inferenceOnMaskingVideo(
                frame_bgr, canvas_bgr, info_frame.box_face, info_frame.key_points_xy, info_frame.key_points_score, **kwargs)
        if isinstance(masking_option, MaskingSticker):
            return masking_option.inferenceOnMaskingVideo(
                frame_bgr, canvas_bgr, info_frame.box_face, info_frame.key_points_xy, info_frame.key_points_score, **kwargs)
        raise NotImplementedError(masking_option)

    """
    """
    @staticmethod
    def maskingImagePlate(frame_bgr, canvas_bgr, info_plate, masking_option, **kwargs) -> np.ndarray:
        if isinstance(masking_option, MaskingBlur):
            return masking_option.inferenceOnMaskingImage(
                frame_bgr, canvas_bgr, angle=None, box=info_plate.box, mask_info=getattr(info_plate, 'mask_info'), **kwargs)
        if isinstance(masking_option, MaskingMosaic):
            return masking_option.inferenceOnMaskingImage(
                frame_bgr, canvas_bgr, angle=None, box=info_plate.box, mask_info=getattr(info_plate, 'mask_info'), **kwargs)
        raise NotImplementedError(masking_option)

    """
    """
    @staticmethod
    def maskingImage(frame_bgr, canvas_bgr, info_object, masking_option, **kwargs):
        if isinstance(info_object, InfoImage_Person):
            return MaskingFunction.maskingImageFace(frame_bgr, canvas_bgr, info_object, masking_option, **kwargs)
        if isinstance(info_object, InfoImage_Plate):
            return MaskingFunction.maskingImagePlate(frame_bgr, canvas_bgr, info_object, masking_option, **kwargs)
        raise NotImplementedError(info_object)
