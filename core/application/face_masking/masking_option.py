
import logging
import os
import random
import typing
import cv2
import numpy as np
from ...utils.resource import Resource
from ... import XManager


class MaskingOption:
    """
    """
    MaskingOption_Blur = [101, 102, 103, 104, 105]
    MaskingOption_Mosaic = [201, 202]
    MaskingOption_Sticker = [301]

    @staticmethod
    def packageAsBlur():
        seed = random.randint(0, 10) % 5
        seed = random.choice([0, 4])
        if seed == 0:
            return MaskingOption(101, dict(blur_type='blur_gaussian', focus_type='head'))
        if seed == 1:
            return MaskingOption(102, dict(blur_type='blur_motion', focus_type='head'))
        if seed == 2:
            return MaskingOption(103, dict(blur_type='blur_water', focus_type='head'))
        if seed == 3:
            return MaskingOption(103, dict(blur_type='blur_pencil', focus_type='head'))
        if seed == 4:
            return MaskingOption(103, dict(blur_type='blur_diffuse', focus_type='head'))

    @staticmethod
    def packageAsMosaic():
        seed = random.randint(0, 10) % 2
        seed = 1
        if seed == 0:
            return MaskingOption(201, dict(mosaic_type='mosaic_pixel_square', focus_type='head'))
        if seed == 1:
            return MaskingOption(202, dict(mosaic_type='mosaic_pixel_polygon', focus_type='head'))

    @staticmethod
    def packageAsSticker():
        seed = 0#random.randint(0, 10) % 2
        if seed == 0:
            index = random.randint(0, 14)
            path = '{}/resource/sticker/cartoon/{:02d}.png'.format(os.path.split(__file__)[0], index)
            sticker_bgr = Resource.loadImage(path)
            return MaskingOption(301, sticker_bgr)
        if seed == 1:
            sticker = cv2.imread('{}/resource/sticker/retro/01.png'.format(os.path.split(__file__)[0]), cv2.IMREAD_UNCHANGED)
            points = np.array([[88, 227], [190, 227]], dtype=np.int32)
            return MaskingOption(301, dict(bgr=sticker, eyes_center=points))

    @staticmethod
    def package():
        code = random.choice([1,])
        return [MaskingOption.packageAsBlur,
                MaskingOption.packageAsMosaic,
                MaskingOption.packageAsSticker][code]()

    @staticmethod
    def getRandomMaskingOptionDict(person_list):
        options_dict = dict()
        for n, person in enumerate(person_list):
            options_dict[person.identity] = MaskingOption.package()
            print(person.identity, options_dict[person.identity].option_code)
            # if n == 2: break
        return options_dict

    """
    """
    def __init__(self, option_code, parameters):
        self.option_code = option_code
        self.parameters = parameters

    def __str__(self):
        return '{} --> {}'.format(self.option_code, self.parameters)