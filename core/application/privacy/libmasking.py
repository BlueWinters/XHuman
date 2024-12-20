
import logging
import os
import typing
import cv2
import numpy as np
import tqdm
from .libscaner import *
from .libmasking_blur import LibMasking_Blur
from .libmasking_mosaic import LibMasking_Mosaic
from .libmasking_sticker import LibMasking_Sticker
from ...base import XPortrait, XPortraitHelper
from ...utils.context import XContextTimer
from ...utils.video import XVideoReader, XVideoWriter
from ...utils.resource import Resource
from ... import XManager


class MaskingOption:
    """
    """
    MaskingOption_Default = 0
    MaskingOption_Blur = 0
    MaskingOption_Mosaic = 1
    MaskingOption_Sticker = 2

    @staticmethod
    def packageAsBlur(blur_kernel=15):
        assert isinstance(blur_kernel, int)
        return MaskingOption(MaskingOption.MaskingOption_Blur, blur_kernel)

    @staticmethod
    def packageAsMosaic(num_pixel=5):
        assert isinstance(num_pixel, int)
        return MaskingOption(MaskingOption.MaskingOption_Mosaic, num_pixel)

    @staticmethod
    def packageAsSticker(sticker_bgr=None):
        if sticker_bgr is None:
            path = '{}/cartoon/00.png'.format(os.path.split(__file__)[0])
            sticker_bgr = Resource.loadImage(path)
        assert isinstance(sticker_bgr, np.ndarray)
        return MaskingOption(MaskingOption.MaskingOption_Sticker, sticker_bgr)

    @staticmethod
    def package(code):
        return [MaskingOption.packageAsBlur,
                MaskingOption.packageAsMosaic,
                MaskingOption.packageAsSticker][code]()

    @staticmethod
    def getRandomMaskingOptionDict(person_list):
        options_dict = dict()
        for n, person in enumerate(person_list):
            code = int(n % 3)
            options_dict[person.identity] = MaskingOption.package(code)
        return options_dict

    """
    """
    def __init__(self, option_code, parameters):
        self.option_code = option_code
        self.parameters = parameters


class LibMasking:
    """
    """
    @staticmethod
    def benchmark():
        path_in_video = R'N:\archive\2024\1126-video\DanceShow2\06\input-06.mp4'
        path_out_video = R'N:\archive\2024\1126-video\DanceShow2\06\input-06-pro.mp4'
        path_out_json = R'N:\archive\2024\1126-video\DanceShow2\06\input-06.json'
        cache_iter = XPortraitHelper.getXPortraitIterator(path_video=path_in_video)
        # video_info = LibMasking.scan(path_video_in)
        video_info = LibScaner.inference(cache_iter, fixed_num=5, path_out_json=path_out_json)
        # video_info = VideoInfo.loadFromJson(path_out_json)
        options_dict = MaskingOption.getRandomMaskingOptionDict(video_info.person_identity_history)
        LibMasking.maskingVideo(path_in_video, path_out_video, video_info, options_dict)

    """
    """
    def __init__(self, *args, **kwargs):
        pass

    def __del__(self):
        # logging.warning('delete module {}'.format(self.__class__.__name__))
        pass

    def initialize(self, *args, **kwargs):
        pass

    """
    """
    @staticmethod
    def maskingSingleFace(bgr, box, masking_option: MaskingOption):
        if masking_option.option_code == MaskingOption.MaskingOption_Blur:
            bgr = LibMasking_Blur.inferenceOnBox(bgr, box, masking_option.parameters)
        if masking_option.option_code == MaskingOption.MaskingOption_Mosaic:
            bgr = LibMasking_Mosaic.inferenceOnBox(bgr, box, masking_option.parameters)
        if masking_option.option_code == MaskingOption.MaskingOption_Sticker:
            bgr = LibMasking_Sticker.inferenceWithBox(bgr, box, masking_option.parameters)
        return bgr

    """
    """
    @staticmethod
    def getFixedNumFromVideo(path_in_video, fixed_num=-1, num_preview=1):
        if fixed_num == -1:
            if 0 < num_preview <= 8:
                reader = XVideoReader(path_in_video)
                frame_list = reader.sampleFrames(beg=0, end=num_preview, step=1)
                video_info = LibScaner.inference(frame_list)
                return len(video_info.person_identity_history)
            else:
                return -1
        else:
            return fixed_num

    @staticmethod
    def scanningVideo(path_in_video, path_out_json, **kwargs) -> VideoInfo:
        parameters = dict()
        parameters['path_out_json'] = path_out_json or Resource.createRandomCacheFileName('.json')
        parameters['fixed_num'] = LibMasking.getFixedNumFromVideo(path_in_video, kwargs.pop('fixed_num', -1), kwargs.pop('num_preview', -1))
        parameters['sample_step'] = kwargs.pop('sample_step', 1)
        iterator = XPortraitHelper.getXPortraitIterator(path_video=path_in_video)
        video_info = LibScaner.inference(iterator, **parameters)
        LibScaner.visualAllFrames(path_in_video, kwargs.pop('path_out_video', None), video_info)
        return video_info

    """
    """
    @staticmethod
    def maskingVideo(path_in_video: str, path_in_json: str, options_dict: typing.Dict[int, MaskingOption], path_video_out: str):
        video_info = VideoInfo.loadFromJson(path_in_json)
        if len(options_dict) == 0:
            options_dict = MaskingOption.getRandomMaskingOptionDict(video_info.person_identity_history)
        reader = XVideoReader(path_in_video)
        writer = XVideoWriter(reader.desc(True))
        writer.open(path_video_out)
        iterator_list = [(person, person.getInfoIterator()) for person in video_info.person_identity_history]
        for index_frame, bgr in enumerate(reader):
            for _, (person, it) in enumerate(iterator_list):
                try:
                    info: PersonFrameInfo = it.next()
                    if info.index_frame == index_frame:
                        masking_option = options_dict[person.identity]
                        bgr = LibMasking.maskingSingleFace(bgr, info.box, masking_option)
                        it.update()
                except IndexError as e:
                    pass
            writer.write(bgr)
        writer.release(reformat=True)

    """
    """
    @staticmethod
    def getFixedNumFromImage(cache:XPortrait, max_num):
        assert isinstance(max_num, int)
        return max_num if max_num > 0 else cache.number

    @staticmethod
    def scanningImage(path_image_or_bgr, path_out_json, **kwargs) -> typing.Tuple[VideoInfo, typing.Union[np.ndarray, None]]:
        cache = XPortrait.packageAsCache(path_image_or_bgr)
        path_out_json = path_out_json or Resource.createRandomCacheFileName('.json')
        max_num = LibMasking.getFixedNumFromImage(path_image_or_bgr, kwargs.pop('max_num', -1))
        video_info = LibScaner.inference([cache], path_out_json=path_out_json, fixed_num=max_num)
        visual_bgr = LibScaner.visualSingleFrame(cache.bgr, video_info) if bool(kwargs.pop('visual_scanning', False)) else None
        return video_info, visual_bgr

    @staticmethod
    def maskingImage(path_image_or_bgr: str, path_in_json: str, options_dict: typing.Dict[int, MaskingOption]) -> np.ndarray:
        video_info = VideoInfo.loadFromJson(path_in_json)
        if len(options_dict) == 0:
            options_dict = MaskingOption.getRandomMaskingOptionDict(video_info.person_identity_history)
        iterator_list = [(person, person.getInfoIterator()) for person in video_info.person_identity_history]
        bgr = cv2.imread(path_image_or_bgr) if isinstance(path_image_or_bgr, str) else np.array(path_image_or_bgr, dtype=np.uint8)
        for _, (person, it) in enumerate(iterator_list):
            info: PersonFrameInfo = it.next()
            masking_option = options_dict[person.identity]
            bgr = LibMasking.maskingSingleFace(bgr, info.box, masking_option)
        return bgr
