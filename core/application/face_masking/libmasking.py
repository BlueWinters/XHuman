
import logging
import os
import queue
import random
import threading
import typing
import cv2
import numpy as np
import tqdm
from .libscaner import *
from .masking_option import MaskingOption
from .libmasking_blur import LibMasking_Blur
from .libmasking_mosaic import LibMasking_Mosaic
from .libmasking_sticker import LibMasking_Sticker
from .libmasking_worker import MaskingVideoWorker
from ...base import XPortrait, XPortraitExceptionAssert
from ...utils.context import XContextTimer
from ...utils.video import XVideoReader, XVideoWriter
from ...utils.resource import Resource
from ... import XManager


class LibMasking:
    """
    """
    @staticmethod
    def benchmarkOnImage():
        from core.thirdparty.cartoon.libcartoon import LibCartoonWrapperQ
        # N:\archive\2024\1126-video\DanceShow2\01\test-image
        path_in_image = R'N:\archive\2024\1126-video\0117\4.jpg'
        path_out_json = R'N:\archive\2024\1126-video\0117\4.json'
        path_out_image = R'N:\archive\2024\1126-video\0117\4-output.png'
        video_info, _ = LibMasking.scanningImage(path_in_image, path_out_json=path_out_json)
        # video_info = VideoInfo.loadVideoInfo(path_in_json=path_out_json)
        # options_dict = MaskingOption.getRandomMaskingOptionDict(video_info.person_identity_history)
        options_dict = dict()
        for identity, data in video_info.getIdentityPreviewDict().items():
            bgr, box = data['image'], data['box']
            bgr_style, crop_box = LibCartoonWrapperQ.inference(np.copy(bgr), np.array(box, dtype=np.int32).tolist())
            options_dict[identity] = MaskingOption(301, dict(bgr=bgr_style, paste=crop_box))
            cv2.imwrite(R'N:\archive\2024\1126-video\0117\{}.png'.format(identity), bgr_style)
        result = LibMasking.maskingImage(path_in_image, options_dict, path_in_json=path_out_json)
        cv2.imwrite(path_out_image, result)

    @staticmethod
    def benchmarkOnVideo():
        from core.thirdparty.cartoon.libcartoon import LibCartoonWrapperQ
        # easy(obama)
        # path_in_video = R'N:\archive\2024\1126-video\Obama\02\input-02.mp4'
        # path_out_json = R'N:\archive\2024\1126-video\Obama\02\common\input-02.json'
        # path_out_video_scanning = R'N:\archive\2024\1126-video\Obama\02\common\input-02-scanning.mp4'
        # path_out_video_masking = R'N:\archive\2024\1126-video\Obama\02\common\input-02-masking-mosaic1.mp4'
        # easy
        # path_in_video = R'N:\archive\2024\1126-video\DanceShow2\01\input-01.mp4'
        # path_out_json = R'N:\archive\2024\1126-video\DanceShow2\01\common\input-01.json'
        # path_out_video_scanning = R'N:\archive\2024\1126-video\DanceShow2\01\common\input-01-scanning.mp4'
        # path_out_video_masking = R'N:\archive\2024\1126-video\DanceShow2\01\common\input-01-masking-mosaic.mp4'
        # hard
        path_in_video = R'N:\archive\2024\1126-video\DanceShow2\01\input-01.mp4'
        path_out_json = R'N:\archive\2024\1126-video\DanceShow2\01\input-01.json'
        # path_out_video_scanning = R'N:\archive\2024\1126-video\DanceShow2\01\test\input-01-scanning_yolo.mp4'
        path_out_video_masking = R'N:\archive\2024\1126-video\DanceShow2\01\masking\input-01-masking-cartoon-old.mp4'

        # pipeline
        # video_info = LibMasking.scanningVideo(path_in_video, path_out_json=path_out_json, path_out_video=None)
        video_info = VideoInfo.loadVideoInfo(path_in_json=path_out_json)
        # options_dict = MaskingOption.getRandomMaskingOptionDict(video_info.person_identity_history)
        options_dict = dict()
        for identity, data in video_info.getIdentityPreviewDict().items():
            bgr = data['image']
            box = data['box']
            bgr_style, crop_box = LibCartoonWrapperQ.inference(np.copy(bgr), np.array(box, dtype=np.int32).tolist())
            options_dict[identity] = MaskingOption(301, dict(bgr=bgr_style, box=crop_box))
        #     cv2.imwrite(R'N:\archive\2024\1126-video\DanceShow2\01\cache\{}.png'.format(identity), bgr_style)
        LibMasking.maskingVideo(path_in_video, options_dict, path_out_video_masking, path_in_json=path_out_json)

        # path_in_video = R'N:\archive\2024\1126-video\DanceShow4\input.mp4'
        # path_out_json = R'N:\archive\2024\1126-video\DanceShow4\input.json'
        # path_out_video_scanning = R'N:\archive\2024\1126-video\DanceShow4\input-scanning-xxxx.mp4'
        # path_out_video_masking = R'N:\archive\2024\1126-video\DanceShow4\input-masking.mp4'
        # video_info = LibMasking.scanningVideo(path_in_video, path_out_json=path_out_json, path_out_video=path_out_video_scanning, fixed_num=5)
        # video_info = VideoInfo.loadVideoInfo(path_in_json=path_out_json)
        # options_dict = dict()
        # for identity, data in video_info.getIdentityPreviewDict().items():
        #     bgr, box = data['image'], data['box']
        #     print(data['index_frame'], data['box'], data['box_score'])
        #     cv2.imwrite(R'N:\archive\2024\1126-video\DanceShow4\input-source-{}.png'.format(identity), bgr)
        #     try:
        #         bgr_style, crop_box = LibCartoonWrapperQ.inference(np.copy(bgr), np.array(box, dtype=np.int32).tolist())
        #         cv2.imwrite(R'N:\archive\2024\1126-video\DanceShow4\input-cartoon-{}.png'.format(identity), bgr_style)
        #         options_dict[identity] = MaskingOption(301, dict(bgr=bgr_style, box=crop_box))
        #     except Exception as e:
        #         print(e)
        # options_dict = MaskingOption.getRandomMaskingOptionDict(video_info.person_identity_history)
        # options_dict = {
        #     1: MaskingOption(301, dict(bgr=cv2.imread(R'N:\archive\2024\1126-video\Stuff\online\face\1.png', cv2.IMREAD_UNCHANGED), box=[[346, 290, 408, 354], [324, 250, 429, 355]])),
        #     2: MaskingOption(301, dict(bgr=cv2.imread(R'N:\archive\2024\1126-video\Stuff\online\face\2.png', cv2.IMREAD_UNCHANGED), box=[[158, 281, 207, 329], [141, 249, 223, 331]])),
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
    def maskingSingleFace(index_frame, bgr, box, masking_option: MaskingOption):
        if masking_option.option_code in MaskingOption.MaskingOption_Blur:
            return LibMasking_Blur.inferenceWithBox(bgr, box, masking_option)
        if masking_option.option_code in MaskingOption.MaskingOption_Mosaic:
            return LibMasking_Mosaic.inferenceWithBox(bgr, box, masking_option)
        if masking_option.option_code in MaskingOption.MaskingOption_Sticker:
            return LibMasking_Sticker.inferenceWithBox(bgr, box, masking_option)
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
                    cache = XBody(bgr, backend='ultralytics.yolo11m-pose')
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
        parameters['schedule_call'] = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
        iterator = LibScaner.getCacheIterator(path_video=path_in_video)
        video_info = LibScaner.inferenceOnVideo(iterator, **parameters)
        LibScaner.visualAllFrames(path_in_video, kwargs.pop('path_out_video', None), video_info)
        return video_info

    """
    """
    @staticmethod
    def maskingVideoOld(path_in_video: str, options_dict: typing.Dict[int, MaskingOption], path_video_out: str, **kwargs):
        parameters = dict(path_in_json=kwargs.pop('path_in_json', None), video_info_string=kwargs.pop('video_info_string', ''))
        video_info = VideoInfo.loadVideoInfo(**parameters)
        if len(options_dict) == 0:
            options_dict = MaskingOption.getRandomMaskingOptionDict(video_info.person_identity_history)
        reader = XVideoReader(path_in_video)
        writer = XVideoWriter(reader.desc(True))
        writer.open(path_video_out)
        min_seconds = int(kwargs.pop('min_seconds', 1) * reader.desc()['fps'])
        schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
        iterator_list = [(person, person.getInfoIterator()) for person in video_info.person_identity_history if len(person.frame_info_list) > min_seconds]
        with XContextTimer(True):
            with tqdm.tqdm(total=len(reader)) as bar:
                for index_frame, bgr in enumerate(reader):
                    for _, (person, it) in enumerate(iterator_list):
                        info: PersonFrameInfo = it.next()
                        if info.index_frame == index_frame:
                            if person.identity in options_dict:
                                masking_option = options_dict[person.identity]
                                bgr = LibMasking.maskingSingleFace(index_frame, bgr, info.box_face, masking_option)
                                # if np.sum(info.box_face) == 0:
                                #     bgr = LibMasking.maskingSingleFace(index_frame, bgr, it.previous().box_face, masking_option)
                                # else:
                                #     bgr = LibMasking.maskingSingleFace(index_frame, bgr, info.box_face, masking_option)
                            it.update()
                    writer.write(bgr)
                    # update schedule
                    bar.update(1)
                    schedule_call('打码视频', float((index_frame+1)/len(reader)))
        # reformat
        writer.release(reformat=True)

    @staticmethod
    def maskingVideo(path_in_video: str, options_dict: typing.Dict[int, MaskingOption], path_video_out: str, **kwargs):
        parameters = dict(path_in_json=kwargs.pop('path_in_json', None), video_info_string=kwargs.pop('video_info_string', ''))
        video_info = VideoInfo.loadVideoInfo(**parameters)
        if len(options_dict) == 0:
            options_dict = MaskingOption.getRandomMaskingOptionDict(video_info.person_identity_history)
        reader = XVideoReader(path_in_video)
        min_seconds = int(kwargs.pop('min_seconds', 1) * reader.desc()['fps'])
        schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
        num_workers = kwargs.pop('num_workers', 4)
        iterator_list = video_info.getSplitInfoIterator(len(reader), num_workers, min_seconds)
        with XContextTimer(True):
            # create worker
            worker_list = MaskingVideoWorker.createWorkers(num_workers, path_in_video,
                options_dict, iterator_list, LibMasking.maskingSingleFace, schedule_call=lambda v: schedule_call('打码视频', v))
            # do masking
            MaskingVideoWorker.doMaskingParallel(worker_list)
            # dump images
            MaskingVideoWorker.dumpAll(path_video_out, worker_list, reader.desc(True))

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
        schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
        video_info = LibScaner.inferenceOnImage(cache, path_out_json=path_out_json, schedule_call=schedule_call)
        visual_bgr = LibScaner.visualSingleFrame(cache.bgr, video_info) if bool(kwargs.pop('visual_scanning', False)) else None
        return video_info, visual_bgr

    @staticmethod
    def maskingImage(path_image_or_bgr: str, options_dict: typing.Dict[int, MaskingOption], **kwargs) -> np.ndarray:
        parameters = dict(path_in_json=kwargs.pop('path_in_json', None), video_info_string=kwargs.pop('video_info_string', ''))
        video_info = VideoInfo.loadVideoInfo(**parameters)
        schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
        if len(options_dict) == 0:
            options_dict = MaskingOption.getRandomMaskingOptionDict(video_info.person_identity_history)
        iterator_list = [(person, person.getInfoIterator()) for person in video_info.person_identity_history]
        bgr = cv2.imread(path_image_or_bgr) if isinstance(path_image_or_bgr, str) else np.array(path_image_or_bgr, dtype=np.uint8)
        for n, (person, it) in enumerate(iterator_list):
            info: PersonFrameInfo = it.next()
            if person.identity in options_dict:
                masking_option = options_dict[person.identity]
                bgr = LibMasking.maskingSingleFace(0, bgr, info.box_face, masking_option)
                schedule_call('打码图片', float((n + 1) / len(iterator_list)))
        return bgr

