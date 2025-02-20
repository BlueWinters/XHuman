
import logging
import cv2
import random
import numpy as np
from .scanning.scanning_image import InfoImage_Person
from .scanning.scanning_video import InfoVideo_Frame
from .masking_blur import *
from .masking_mosaic import *
from .masking_sticker import *


class MaskingFunction:
    """
    """
    @staticmethod
    def maskingImageFace(frame_bgr, canvas_bgr, info_person, masking_option, **kwargs) -> np.ndarray:
        assert isinstance(info_person, InfoImage_Person)
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
        assert isinstance(info_frame, InfoVideo_Frame)
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
    MaskingClassDict = {
        # mask-blur
        MaskingBlurGaussian.NameEN: MaskingBlurGaussian,
        MaskingBlurMotion.NameEN: MaskingBlurMotion,
        MaskingBlurWater.NameEN: MaskingBlurWater,
        MaskingBlurPencil.NameEN: MaskingBlurPencil,
        MaskingBlurDiffusion.NameEN: MaskingBlurDiffusion,
        # mask-mosaic
        MaskingMosaicSquare.NameEN: MaskingMosaicSquare,
        MaskingMosaicPolygon.NameEN: MaskingMosaicPolygon,
        # mask-sticker
        MaskingStickerAlignPoints.NameEN: MaskingStickerAlignPoints,
        MaskingStickerAlignBoxStatic.NameEN: MaskingStickerAlignBoxStatic,
        MaskingStickerAlignBoxDynamic.NameEN: MaskingStickerAlignBoxDynamic,
    }

    MaskingOptionDict = {
        # mask-blur
        101: MaskingBlurGaussian.parameterize(),
        102: MaskingBlurMotion.parameterize(),
        103: MaskingBlurWater.parameterize(),
        104: MaskingBlurDiffusion.parameterize(),
        105: MaskingBlurPencil.parameterize(),
        # mask-mosaic
        201: MaskingMosaicSquare.parameterize(),
        202: MaskingMosaicPolygon.parameterize(),
        203: MaskingMosaicPolygon.parameterize(n_div=24, visual_boundary=True),   # mosaic_pixel_polygon_small_line
        204: MaskingMosaicPolygon.parameterize(n_div=24, visual_boundary=False),  # mosaic_pixel_polygon_small
        205: MaskingMosaicPolygon.parameterize(n_div=16, visual_boundary=True),   # mosaic_pixel_polygon_big_line
        206: MaskingMosaicPolygon.parameterize(n_div=16, visual_boundary=False),  # mosaic_pixel_polygon_big
        # mask-sticker
        301: MaskingStickerAlignPoints.parameterize(),
        302: MaskingStickerAlignBoxStatic.parameterize(),
        303: MaskingStickerAlignBoxDynamic.parameterize(),
        # 311: MaskingStickerAlignPoints.parameterize(resource=('eyes_center_affine', '01')),
        # 312: MaskingStickerAlignPoints.parameterize(resource=('eyes_center_similarity', '01')),
        # 313: MaskingStickerAlignPoints.parameterize(resource=('eyes_center_affine', '31')),
    }

    @staticmethod
    def getRandomMaskingOptionDict(person_list):
        mask_code_list = list(MaskingFunction.MaskingOptionDict.keys())
        options_dict = dict()
        for n, person in enumerate(person_list):
            code = random.choice(mask_code_list)
            option_object = MaskingFunction.MaskingOptionDict[code]()
            options_dict[person.identity] = option_object
            print('identity-{}: {}'.format(person.identity, str(option_object)))
        return options_dict

    @staticmethod
    def getMaskingOptionDict(option_code_list, person_identity_list):
        options_dict = dict()
        for code, person_identity in zip(option_code_list, person_identity_list):
            if code in MaskingFunction.MaskingOptionDict:
                option_object = MaskingFunction.MaskingOptionDict[code]()
                options_dict[person_identity] = option_object
                logging.info('identity-{}, {}'.format(str(person_identity), str(option_object)))
        return options_dict

    """
    """
    @staticmethod
    def getMaskingOption(code, parameters):
        assert isinstance(parameters, dict)
        config = dict()
        if code in [101, 102, 103, 104, 105]:
            if 'focus_type' in parameters:
                config['align_type'] = parameters.pop('focus_type')
            # if 'blur_type' in parameters:
            #     parameters.pop('blur_type')
        if code in [201, 202]:
            if 'focus_type' in parameters:
                config['align_type'] = parameters.pop('focus_type')
            if 'mosaic_type' in parameters:
                mosaic_type = parameters.pop('mosaic_type')
                if mosaic_type == 'mosaic_pixel_polygon_small_line':
                    code = 203
                if mosaic_type == 'mosaic_pixel_polygon_small':
                    code = 204
                if mosaic_type == 'mosaic_pixel_polygon_big_line':
                    code = 205
                if mosaic_type == 'mosaic_pixel_polygon_big':
                    code = 206
        if code in [301, ]:
            if 'bgr' in parameters:
                config['sticker'] = parameters.pop('bgr')
            if 'eyes_center' in parameters:
                code = 301
                config['eyes_center_affine'] = parameters.pop('eyes_center')
            if 'eyes_center_fix' in parameters:
                code = 301
                config['eyes_center_similarity'] = parameters.pop('eyes_center_fix')
            if 'align' in parameters:
                NotImplementedError('sticker-align')
            if 'paste' in parameters:
                code = 302
                config['box_tuple'] = parameters['paste']
            if 'box' in parameters:
                code = 303
                config['box_tuple'] = parameters['box']
        return MaskingFunction.MaskingOptionDict[code](**config)

