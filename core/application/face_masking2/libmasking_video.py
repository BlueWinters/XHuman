
import logging
import os
import cv2
import tqdm
import numpy as np
from .masking_function import MaskingFunction
from .scanning.scanning_video import InfoVideo
from .helper.cursor import AsynchronousCursor
from .masking_mthread_worker import MaskingVideoSession
from .helper.masking_helper import MaskingHelper
from ...utils import XVideoReader, XVideoWriter, Resource, XContextTimer


class LibMaskingVideo:
    """
    video masking
    """
    @staticmethod
    def scanningVideo(path_in_video, **kwargs) -> InfoVideo:
        path_out_json = kwargs.pop('path_out_json', None) or Resource.createRandomCacheFileName('.json')
        schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
        path_out_video = kwargs.pop('path_out_video', None)
        # main pipeline
        info_video = InfoVideo()
        info_video.doScanning(path_in_video, schedule_call)
        info_video.saveAsJson(path_out_json, schedule_call)
        info_video.saveVisualScanning(path_in_video, path_out_video, schedule_call)
        return info_video

    @staticmethod
    def maskingVideo(path_in_video, options_dict, path_out_video, **kwargs):
        debug_mode = kwargs.pop('debug_mode', False)
        schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
        parameters = dict(
            path_in_json=kwargs.pop('path_in_json', None),
            video_info_string=kwargs.pop('video_info_string', ''))
        with_hair = kwargs.pop('with_hair', True)

        info_video = InfoVideo.createFromDict(**parameters)
        reader = XVideoReader(path_in_video)
        min_frames = min(int(kwargs.pop('min_seconds', 1) * reader.desc()['fps']), 30)
        num_workers = int(kwargs.pop('num_workers', 4))
        if num_workers == 0:
            # process with main thread
            with XContextTimer(True), tqdm.tqdm(total=len(reader)) as bar:
                writer = XVideoWriter(reader.desc(True))
                writer.open(path_out_video)
                cursor_list = info_video.getInfoCursorList(len(reader), 1, min_frames)[0]['cursor_list']
                preview_dict = info_video.getIdentityPreviewDict(size=0, is_bgr=None)
                for frame_index, frame_bgr in enumerate(reader):
                    canvas_bgr = np.copy(frame_bgr)
                    for _, (person, cursor) in enumerate(cursor_list):
                        assert isinstance(cursor, AsynchronousCursor)
                        info = cursor.current()
                        if info.frame_index == frame_index:
                            if person.identity in options_dict:
                                masking_option = options_dict[person.identity]
                                mask_info = MaskingHelper.getPortraitMaskingWithInfoVideo(
                                    frame_index, frame_bgr, person, info, options_dict, with_hair=with_hair)
                                canvas_bgr = MaskingFunction.maskingVideoFace(
                                    frame_bgr, canvas_bgr, info, masking_option,
                                    mask_info=mask_info, preview=preview_dict[person.identity])
                            cursor.next()
                    writer.write(canvas_bgr)
                    bar.update(1)
                # reformat
                writer.release(reformat=True)
        else:
            # process with multi-thread
            with XContextTimer(True):
                assert num_workers > 0, num_workers
                cursor_list = info_video.getInfoCursorList(len(reader), num_workers, min_frames)
                session = MaskingVideoSession(
                    num_workers, path_in_video, options_dict, cursor_list, lambda v: schedule_call('打码视频', v),
                    with_hair=with_hair, debug_mode=debug_mode)
                session.start()
                session.dump(path_out_video)

    # def maskingFunction(self, frame_index, frame_bgr, cursor_list, options_dict, with_hair, preview_dict):
    #     canvas_bgr = np.copy(frame_bgr)
    #     for _, (person, cursor) in enumerate(cursor_list):
    #         assert isinstance(cursor, AsynchronousCursor)
    #         info = cursor.current()
    #         if info.frame_index == frame_index:
    #             if person.identity in options_dict:
    #                 masking_option = options_dict[person.identity]
    #                 mask_info = MaskingHelper.getPortraitMaskingWithInfoVideo(
    #                     frame_index, frame_bgr, person, info, options_dict, with_hair=with_hair)
    #                 canvas_bgr = MaskingFunction.maskingVideoFace(
    #                     frame_bgr, canvas_bgr, info, masking_option,
    #                     mask_info=mask_info, preview=preview_dict[person.identity])
    #             cursor.next()
    #     return canvas_bgr
