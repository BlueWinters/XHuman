
import logging
import os
import cv2
import tqdm
import numpy as np
from .scanning import InfoVideo
from .masking_mthread_worker import MaskingVideoSession, MaskingVideoWorker
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
                preview_dict = info_video.getPreviewAsDict()
                for frame_index, frame_bgr in enumerate(reader):
                    canvas_bgr = MaskingVideoWorker.maskingFunction(
                        frame_index, frame_bgr, cursor_list, options_dict, with_hair, preview_dict)
                    writer.write(canvas_bgr)
                    bar.update(1)
                # reformat
                writer.release(reformat=True)
        else:
            # process with multi-thread
            with XContextTimer(True):
                assert num_workers > 0, num_workers
                cursor_list = info_video.getInfoCursorList(len(reader), num_workers, min_frames)
                preview_dict = info_video.getPreviewAsDict()
                session = MaskingVideoSession(
                    num_workers, path_in_video, options_dict, cursor_list, schedule_call,
                    with_hair=with_hair, preview_dict=preview_dict, debug_mode=debug_mode)
                session.start()
                session.dump(path_out_video)

