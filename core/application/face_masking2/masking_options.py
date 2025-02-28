
import logging
import random
from .masking_blur import *
from .masking_mosaic import *
from .masking_sticker import *


class MaskingOptions:
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
        MaskingStickerCartoon.NameEN: MaskingStickerCartoon,
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
        203: MaskingMosaicPolygon.parameterize(n_div=24, visual_boundary=True),  # mosaic_pixel_polygon_small_line
        204: MaskingMosaicPolygon.parameterize(n_div=24, visual_boundary=False),  # mosaic_pixel_polygon_small
        205: MaskingMosaicPolygon.parameterize(n_div=16, visual_boundary=True),  # mosaic_pixel_polygon_big_line
        206: MaskingMosaicPolygon.parameterize(n_div=16, visual_boundary=False),  # mosaic_pixel_polygon_big
        # mask-sticker
        301: MaskingStickerAlignPoints.parameterize(),
        302: MaskingStickerAlignBoxStatic.parameterize(),
        303: MaskingStickerAlignBoxDynamic.parameterize(),
        304: MaskingStickerCartoon.parameterize(),
        311: MaskingStickerAlignPoints.parameterize(resource=('eyes_center_affine', '01')),
        312: MaskingStickerAlignPoints.parameterize(resource=('eyes_center_similarity', '01')),
        313: MaskingStickerAlignPoints.parameterize(resource=('eyes_center_affine', '31')),
        # specific config
        'person_blur_gaussian': MaskingBlurGaussian.parameterize(kernel=20),
        'person_blur_motion': MaskingBlurMotion.parameterize(kernel=15),
        'person_blur_water': MaskingBlurWater.parameterize(a=2., b=8.),
        'person_mosaic_square': MaskingMosaicSquare.parameterize(num_pixels=20),
        'person_mosaic_polygon_small': MaskingMosaicPolygon.parameterize(n_div=24, visual_boundary=False),
        'person_mosaic_polygon_small_line': MaskingMosaicPolygon.parameterize(n_div=24, visual_boundary=True),
        'person_mosaic_polygon_big': MaskingMosaicPolygon.parameterize(n_div=16, visual_boundary=False),
        'person_mosaic_polygon_big_line': MaskingMosaicPolygon.parameterize(n_div=16, visual_boundary=True),
        'person_sticker_align_points': MaskingStickerAlignPoints.parameterize(),
        'person_sticker_cartoon': MaskingStickerCartoon.parameterize(),
        'person_sticker_custom': MaskingStickerCustom.parameterize(),
        'plate_blur_gaussian': MaskingBlurGaussian.parameterize(kernel=20, fmt_w=220, fmt_h=70),
        'plate_mosaic_square': MaskingMosaicSquare.parameterize(num_pixels=8),
    }

    @staticmethod
    def getRandomMaskingOptionDict(person_list):
        mask_code_list = list(MaskingOptions.MaskingOptionDict.keys())
        options_dict = dict()
        for n, person in enumerate(person_list):
            code = random.choice(mask_code_list)
            option_object = MaskingOptions.MaskingOptionDict[code]()
            options_dict[person.identity] = option_object
            print('identity-{}: {}'.format(person.identity, str(option_object)))
        return options_dict

    @staticmethod
    def getMaskingOptionDict(option_code_list, person_identity_list):
        options_dict = dict()
        for code, person_identity in zip(option_code_list, person_identity_list):
            if code in MaskingOptions.MaskingOptionDict:
                option_object = MaskingOptions.MaskingOptionDict[code]()
                options_dict[person_identity] = option_object
                logging.info('identity-{}, {}'.format(str(person_identity), str(option_object)))
        return options_dict

    """
    """
    @staticmethod
    def getMaskingOption(code, *args, **kwargs):
        if isinstance(code, int):
            return MaskingOptions.getMaskingOptionV1(code, args[0])  # args[0] --> parameters
        if isinstance(code, str):
            return MaskingOptions.getMaskingOptionV2(code, *args, **kwargs)
        raise NotImplementedError([code, args, kwargs])

    @staticmethod
    def getMaskingOptionV1(code, parameters):
        # assert isinstance(parameters, dict)
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
                config['eyes_center_similarity'] = parameters.pop('eyes_center')
            if 'eyes_center_fix' in parameters:
                code = 301
                config['eyes_center_similarity'] = parameters.pop('eyes_center_fix')
            if 'align' in parameters:
                NotImplementedError('sticker-align')
            if 'paste' in parameters:
                code = 304
                config['box_tuple'] = parameters['paste']
            if 'box' in parameters:
                code = 303
                config['box_tuple'] = parameters['box']
        return MaskingOptions.MaskingOptionDict[code](**config)

    @staticmethod
    def getMaskingOptionV2(name, *args, **parameters):
        assert isinstance(name, str), name
        return MaskingOptions.MaskingOptionDict[name](*args, **parameters)
