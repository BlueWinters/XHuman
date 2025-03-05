
import logging
import numpy as np
from .masking_blur import *
from .masking_mosaic import *
from .masking_sticker import *
from .scanning_image import InfoImage_Person
from .scanning_image import InfoImage_Plate
from .scanning_video import InfoVideo_Person_Frame, InfoVideo_Person
from .scanning_video import InfoVideo_Plate_Frame, InfoVideo_Plate


class MaskingFunction:
    """
    """
    @staticmethod
    def maskingImage(frame_bgr, canvas_bgr, info_object, masking_option, **kwargs):
        if isinstance(info_object, InfoImage_Person):
            return MaskingFunction.maskingImageFace(frame_bgr, canvas_bgr, info_object, masking_option, **kwargs)
        if isinstance(info_object, InfoImage_Plate):
            return MaskingFunction.maskingImagePlate(frame_bgr, canvas_bgr, info_object, masking_option, **kwargs)
        raise NotImplementedError(info_object)

    @staticmethod
    def maskingImageFace(frame_bgr, canvas_bgr, info_person, masking_option, **kwargs) -> np.ndarray:
        if isinstance(masking_option, MaskingBlur):
            return masking_option.inferenceOnMaskingImage(frame_bgr, canvas_bgr, mask_info=getattr(info_person, 'mask_info'))
        if isinstance(masking_option, MaskingMosaic):
            return masking_option.inferenceOnMaskingImage(frame_bgr, canvas_bgr, mask_info=getattr(info_person, 'mask_info'))
        if isinstance(masking_option, MaskingSticker):
            return masking_option.inferenceOnMaskingImage(
                frame_bgr, canvas_bgr, angle=info_person.angle, box=info_person.box, landmark=info_person.landmark, auto_rot=True)
        raise NotImplementedError(masking_option)

    @staticmethod
    def maskingImagePlate(frame_bgr, canvas_bgr, info_plate, masking_option, **kwargs) -> np.ndarray:
        if isinstance(masking_option, MaskingBlur):
            return masking_option.inferenceOnMaskingImage(frame_bgr, canvas_bgr, box=info_plate.box)
        if isinstance(masking_option, MaskingMosaic):
            return masking_option.inferenceOnMaskingImage(frame_bgr, canvas_bgr, box=info_plate.box)
        raise NotImplementedError(masking_option)

    """
    """
    @staticmethod
    def maskingVideo(frame_bgr, canvas_bgr, info_object, masking_option, **kwargs):
        if isinstance(info_object, InfoVideo_Person_Frame):
            return MaskingFunction.maskingVideoFace(frame_bgr, canvas_bgr, info_object, masking_option, **kwargs)
        if isinstance(info_object, InfoVideo_Plate_Frame):
            return MaskingFunction.maskingVideoPlate(frame_bgr, canvas_bgr, info_object, masking_option, **kwargs)
        raise NotImplementedError(info_object)

    @staticmethod
    def maskingVideoFace(frame_bgr, canvas_bgr, info_frame, masking_option, **kwargs) -> np.ndarray:
        if isinstance(masking_option, MaskingBlur):
            return masking_option.inferenceOnMaskingVideo(frame_bgr, canvas_bgr, mask_info=kwargs.pop('mask_info'))
        if isinstance(masking_option, MaskingMosaic):
            return masking_option.inferenceOnMaskingVideo(frame_bgr, canvas_bgr, mask_info=kwargs.pop('mask_info'))
        if isinstance(masking_option, MaskingSticker):
            return masking_option.inferenceOnMaskingVideo(
                frame_bgr, canvas_bgr, face_box=info_frame.box_face, key_points_xy=info_frame.key_points_xy,
                key_points_score=info_frame.key_points_score)
        raise NotImplementedError(masking_option)

    @staticmethod
    def maskingVideoPlate(frame_bgr, canvas_bgr, info_frame, masking_option, **kwargs) -> np.ndarray:
        if isinstance(masking_option, MaskingBlur):
            return masking_option.inferenceOnMaskingVideo(frame_bgr, canvas_bgr, box=info_frame.box_track)
        if isinstance(masking_option, MaskingMosaic):
            return masking_option.inferenceOnMaskingVideo(frame_bgr, canvas_bgr, box=info_frame.box_track)
        raise NotImplementedError(masking_option)


