
import logging
import os
import random
import typing
import cv2
import numpy as np
import tqdm
from .libscaner import *
from .libmasking_blur import LibMasking_Blur
from .libmasking_mosaic import LibMasking_Mosaic
from .libmasking_sticker import LibMasking_Sticker
from ...base import XPortrait, XPortraitExceptionAssert
from ...utils.context import XContextTimer
from ...utils.video import XVideoReader, XVideoWriter
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
        if seed == 0:
            return MaskingOption(201, dict(mosaic_type='mosaic_pixel_square', focus_type='head'))
        if seed == 1:
            return MaskingOption(202, dict(mosaic_type='mosaic_pixel_polygon', focus_type='head'))

    @staticmethod
    def packageAsSticker():
        seed = random.randint(0, 10) % 2
        if seed == 0:
            path = '{}/resource/sticker/cartoon/00.png'.format(os.path.split(__file__)[0])
            sticker_bgr = Resource.loadImage(path)
            return MaskingOption(301, sticker_bgr)
        if seed == 1:
            sticker = cv2.imread('{}/resource/sticker/retro/01.png'.format(os.path.split(__file__)[0]), cv2.IMREAD_UNCHANGED)
            points = np.array([[88, 227], [190, 227]], dtype=np.int32)
            return MaskingOption(301, dict(bgr=sticker, eyes_center=points))

    @staticmethod
    def package():
        code = random.choice([0, 2])
        return [MaskingOption.packageAsBlur,
                MaskingOption.packageAsMosaic,
                MaskingOption.packageAsSticker][code]()

    @staticmethod
    def getRandomMaskingOptionDict(person_list):
        options_dict = dict()
        for n, person in enumerate(person_list):
            options_dict[person.identity] = MaskingOption.package()
            print(person.identity, options_dict[person.identity].option_code)
        return options_dict

    """
    """
    def __init__(self, option_code, parameters):
        self.option_code = option_code
        self.parameters = parameters

    def __str__(self):
        return '{} --> {}'.format(self.option_code, self.parameters)


class LibMasking:
    """
    """
    @staticmethod
    def benchmarkOnImage():
        path_in_image = R'N:\archive\2024\1202-mosaic\test\yyy.png'
        path_out_json = R'N:\archive\2024\1202-mosaic\test\yyy.json'
        path_out_image = R'N:\archive\2024\1202-mosaic\test\yyy-test.png'
        LibMasking.scanningImage(path_in_image, path_out_json=path_out_json)
        video_info = VideoInfo.loadVideoInfo(path_in_json=path_out_json)
        # options_dict = MaskingOption.getRandomMaskingOptionDict(video_info.person_identity_history)
        options_dict = {
            1: MaskingOption(301, dict(bgr=cv2.imread(R'N:\archive\2024\1202-mosaic\test\1.png', cv2.IMREAD_UNCHANGED), box=[[200, 84, 340, 224], [152, 0, 387, 229]])),
            2: MaskingOption(301, dict(bgr=cv2.imread(R'N:\archive\2024\1202-mosaic\test\2.png', cv2.IMREAD_UNCHANGED), box=[[516, 148, 596, 228], [488, 96, 623, 231]])),
        }
        result = LibMasking.maskingImage(path_in_image, options_dict, path_in_json=path_out_json)
        cv2.imwrite(path_out_image, result)

    @staticmethod
    def benchmarkOnVideo():
        # easy
        path_in_video = R'N:\archive\2024\1126-video\DanceShow2\01\input-01.mp4'
        path_out_json = R'N:\archive\2024\1126-video\DanceShow2\01\others.json'
        path_out_video_scanning = R'N:\archive\2024\1126-video\DanceShow2\01\input-01-scanning.mp4'
        path_out_video_masking = R'N:\archive\2024\1126-video\DanceShow2\01\input-01-others.mp4'
        # hard
        # path_in_video = R'N:\archive\2024\1126-video\DanceShow\03\input-03.mp4'
        # path_out_json = R'N:\archive\2024\1126-video\DanceShow\03\input-03.json'
        # path_out_video_scanning = R'N:\archive\2024\1126-video\DanceShow\03\input-03-scanning-face.mp4'
        # path_out_video_masking = R'N:\archive\2024\1126-video\DanceShow\03\input-03-masking-face.mp4'
        # pipeline
        # LibMasking.scanningVideo(path_in_video, path_out_json=path_out_json, path_out_video=path_out_video_scanning)
        video_info = VideoInfo.loadVideoInfo(path_in_json=path_out_json)
        options_dict = MaskingOption.getRandomMaskingOptionDict(video_info.person_identity_history)
        LibMasking.maskingVideo(path_in_video, options_dict, path_out_video_masking, path_in_json=path_out_json)

        # path_in_video = R'N:\archive\2024\1202-mosaic\test\input.mp4'
        # path_out_json = R'N:\archive\2024\1202-mosaic\test\input.json'
        # path_out_video_scanning = R'N:\archive\2024\1202-mosaic\test\input-scanning.mp4'
        # path_out_video_masking = R'N:\archive\2024\1202-mosaic\test\input-masking-align.mp4'
        # # LibMasking.scanningVideo(path_in_video, path_out_json=path_out_json, path_out_video=path_out_video_scanning)
        # video_info = VideoInfo.loadVideoInfo(path_in_json=path_out_json)
        # options_dict = {
        #     1: MaskingOption(301, dict(bgr=cv2.imread(R'N:\archive\2024\1202-mosaic\test\1.png', cv2.IMREAD_UNCHANGED), align=[[200, 84, 340, 224], [152, 0, 387, 229]])),
        #     2: MaskingOption(301, dict(bgr=cv2.imread(R'N:\archive\2024\1202-mosaic\test\2.png', cv2.IMREAD_UNCHANGED), align=[[516, 148, 596, 228], [488, 96, 623, 231]])),
        # }
        # LibMasking.maskingVideo(path_in_video, options_dict, path_out_video_masking, path_in_json=path_out_json)

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
    def getMaskingOption(option_code, parameters):
        return MaskingOption(option_code, parameters)

    """
    """
    @staticmethod
    def maskingSingleFace(bgr, box, masking_option: MaskingOption):
        if masking_option.option_code in MaskingOption.MaskingOption_Blur:
            return LibMasking_Blur.inferenceWithBox(bgr, box, masking_option.parameters)
        if masking_option.option_code in MaskingOption.MaskingOption_Mosaic:
            return LibMasking_Mosaic.inferenceWithBox(bgr, box, masking_option.parameters)
        if masking_option.option_code in MaskingOption.MaskingOption_Sticker:
            return LibMasking_Sticker.inferenceWithBox(bgr, box, masking_option.parameters)
        raise NotImplementedError(masking_option)

    """
    """
    @staticmethod
    def getFixedNumFromVideo(path_in_video, fixed_num=-1, num_preview=32):
        if fixed_num == -1:
            if 0 < num_preview <= 32:
                reader = XVideoReader(path_in_video)
                for n in range(num_preview):
                    ret, bgr = reader.read()
                    cache = XBody(bgr)
                    if cache.number > 0:
                        return int(cache.number)
            return -1
        else:
            return fixed_num

    @staticmethod
    def scanningVideo(path_in_video, **kwargs) -> VideoInfo:
        parameters = dict()
        parameters['path_out_json'] = kwargs.pop('path_out_json', None) or Resource.createRandomCacheFileName('.json')
        parameters['fixed_num'] = LibMasking.getFixedNumFromVideo(path_in_video, kwargs.pop('fixed_num', -1), kwargs.pop('num_preview', 32))
        parameters['sample_step'] = kwargs.pop('sample_step', 1)
        iterator = LibScaner.getCacheIterator(path_video=path_in_video)
        video_info = LibScaner.inference(iterator, **parameters)
        LibScaner.visualAllFrames(path_in_video, kwargs.pop('path_out_video', None), video_info)
        return video_info

    """
    """
    @staticmethod
    def maskingVideo(path_in_video: str, options_dict: typing.Dict[int, MaskingOption], path_video_out: str, **kwargs):
        parameters = dict(path_in_json=kwargs.pop('path_in_json', None), video_info_string=kwargs.pop('video_info_string', ''))
        video_info = VideoInfo.loadVideoInfo(**parameters)
        if len(options_dict) == 0:
            options_dict = MaskingOption.getRandomMaskingOptionDict(video_info.person_identity_history)
        reader = XVideoReader(path_in_video)
        writer = XVideoWriter(reader.desc(True))
        writer.open(path_video_out)
        min_seconds = int(kwargs.pop('min_seconds', 1) * reader.desc()['fps'])
        iterator_list = [(person, person.getInfoIterator()) for person in video_info.person_identity_history if len(person.frame_info_list) > min_seconds]
        with XContextTimer(True):
            with tqdm.tqdm(total=len(reader)) as bar:
                for index_frame, bgr in enumerate(reader):
                    for _, (person, it) in enumerate(iterator_list):
                        try:
                            info: PersonFrameInfo = it.next()
                            if info.index_frame == index_frame:
                                if person.identity in options_dict:
                                    masking_option = options_dict[person.identity]
                                    if np.sum(info.box_face) == 0:
                                        bgr = LibMasking.maskingSingleFace(bgr, it.previous().box_face, masking_option)
                                    else:
                                        bgr = LibMasking.maskingSingleFace(bgr, info.box_face, masking_option)
                                it.update()
                        except IndexError as e:
                            pass
                    writer.write(bgr)
                    bar.update(1)

        writer.release(reformat=True)

    """
    """
    @staticmethod
    def getFixedNumFromImage(cache: XPortrait, max_num):
        assert isinstance(max_num, int)
        if max_num == -1:
            XPortraitExceptionAssert.assertNoFace(cache.number)
            return cache.number
        return max_num

    @staticmethod
    def scanningImage(path_image_or_bgr, **kwargs) -> typing.Tuple[VideoInfo, typing.Union[np.ndarray, None]]:
        cache = LibScaner.packageAsCache(path_image_or_bgr)
        path_out_json = kwargs.pop('path_out_json', None) or Resource.createRandomCacheFileName('.json')
        max_num = LibMasking.getFixedNumFromImage(cache, kwargs.pop('max_num', -1))
        video_info = LibScaner.inference([cache], path_out_json=path_out_json, fixed_num=max_num)
        visual_bgr = LibScaner.visualSingleFrame(cache.bgr, video_info) if bool(kwargs.pop('visual_scanning', False)) else None
        return video_info, visual_bgr

    @staticmethod
    def maskingImage(path_image_or_bgr: str, options_dict: typing.Dict[int, MaskingOption], **kwargs) -> np.ndarray:
        parameters = dict(path_in_json=kwargs.pop('path_in_json', None), video_info_string=kwargs.pop('video_info_string', ''))
        video_info = VideoInfo.loadVideoInfo(**parameters)
        if len(options_dict) == 0:
            options_dict = MaskingOption.getRandomMaskingOptionDict(video_info.person_identity_history)
        iterator_list = [(person, person.getInfoIterator()) for person in video_info.person_identity_history]
        bgr = cv2.imread(path_image_or_bgr) if isinstance(path_image_or_bgr, str) else np.array(path_image_or_bgr, dtype=np.uint8)
        for _, (person, it) in enumerate(iterator_list):
            info: PersonFrameInfo = it.next()
            if person.identity in options_dict:
                masking_option = options_dict[person.identity]
                bgr = LibMasking.maskingSingleFace(bgr, info.box_face, masking_option)
        return bgr
