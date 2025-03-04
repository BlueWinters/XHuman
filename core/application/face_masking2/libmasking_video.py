
import logging
import os
import cv2
import tqdm
import json
import pickle
from .scanning import *
from .masking_mthread_worker import MaskingVideoSession, MaskingVideoWorker
from ...utils import XVideoReader, XVideoWriter, Resource, XContextTimer


class LibMaskingVideo:
    """
    """
    ClassDict = dict(person=ScanningVideo_Person, plate=ScanningVideo_Plate)

    """
    """
    def __init__(self, *args, **kwargs):
        self.scanning_dict = dict()
        self.info_string = None

    """
    scanning
    """
    def trackingObjects(self, category_list, path_in_video, schedule_call) -> dict:
        objects_counter = 0
        for category in category_list:
            scanning = LibMaskingVideo.ClassDict[category](object_identity_seq=objects_counter)
            objects_counter += scanning.doScanning(path_in_video, schedule_call)
            self.scanning_dict[category] = scanning
        return self.scanning_dict

    def formatAsJson(self, path_out_json=None):
        if isinstance(self.info_string, str) is False:
            format_list = list()
            for category, scanning in self.scanning_dict.items():
                format_list.append(dict(category=category, objects=scanning.formatObjectsAsDict(with_frame_info=True)))
            self.info_string = json.dumps(format_list, indent=4)
            if isinstance(path_out_json, str) and path_out_json.endswith('.json'):
                open(path_out_json, 'w').write(self.info_string)

    def visualScanning(self, path_in_video, path_out_video, schedule_call):
        for category, scanning in self.scanning_dict.items():
            prefix, suffix = path_out_video.split('.')
            path_out_video_new = '{}.{}.{}'.format(prefix, category, suffix)
            scanning.saveVisualScanning(path_in_video, path_out_video_new, schedule_call)
        return None

    def scanningVideo(self, path_in_video, **kwargs):
        category_list = kwargs.pop('category_list', ['person'])
        path_out_json = kwargs.pop('path_out_json', None) or Resource.createRandomCacheFileName('.json')
        schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
        path_out_video = kwargs.pop('path_out_video', None)
        # tracking
        self.trackingObjects(category_list, path_in_video, schedule_call)
        # format
        self.formatAsJson(path_out_json)
        # visual
        self.visualScanning(path_in_video, path_out_video, schedule_call)

    """
    preview
    """
    def getSummaryAsDict(self):
        summary_dict = dict()
        for category, scanning in self.scanning_dict.items():
            summary_dict.update(scanning.getPreviewSummaryAsDict())
        return summary_dict

    def savePreviewAsPickle(self, path_out_pkl):
        preview_dict = self.getSummaryAsDict()
        pickle.dump(preview_dict, open(path_out_pkl, 'wb'))

    @staticmethod
    def loadPreviewFromPickle(path_in_pkl):
        return pickle.load(open(path_in_pkl, 'rb'))

    """
    json
    """
    def getInfoAsJson(self, *args, **kwargs) -> str:
        assert isinstance(self.info_string, str), self.info_string
        return self.info_string

    """
    masking
    """
    @staticmethod
    def createFromScanningDict(**kwargs):
        if 'path_in_json' in kwargs and isinstance(kwargs['path_in_json'], str):
            masking_video = LibMaskingVideo()
            for info in json.load(open(kwargs.pop('path_in_json'), 'r')):
                category = info['category']
                info_objects = info['objects']
                masking_video.scanning_dict[category] = LibMaskingVideo.ClassDict[category].createFromDict(objects_list=info_objects)
            return masking_video
        if 'video_info_string' in kwargs and isinstance(kwargs['video_info_string'], str):
            masking_video = LibMaskingVideo()
            for info in json.loads(kwargs.pop('video_info_string')):
                category = info['category']
                info_objects = info['objects']
                masking_video.scanning_dict[category] = LibMaskingVideo.ClassDict[category].createFromDict(objects_list=info_objects)
            return masking_video
        raise NotImplementedError('both "path_in_json" and "video_info_string" not in kwargs')

    def getAllObjects(self):
        objects_list = []
        for category, scanning in self.scanning_dict.items():
            objects_list.extend(scanning.object_identity_history)
        return objects_list

    def maskingVideo(self, path_in_video, options_dict, path_out_video, **kwargs):
        debug_mode = kwargs.pop('debug_mode', False)
        schedule_call = kwargs.pop('schedule_call', lambda *_args, **_kwargs: None)
        # parameters = dict(
        #     path_in_json=kwargs.pop('path_in_json', None),
        #     video_info_string=kwargs.pop('video_info_string', ''))
        with_hair = kwargs.pop('with_hair', True)

        reader = XVideoReader(path_in_video)
        min_frames = min(int(kwargs.pop('min_seconds', 1) * reader.desc()['fps']), 0)
        num_workers = int(kwargs.pop('num_workers', 4))
        num_frames = len(reader)
        objects_list = self.getAllObjects()
        if num_workers == 0:
            # process with main thread
            with XContextTimer(True), tqdm.tqdm(total=len(reader)) as bar:
                writer = XVideoWriter(reader.desc(True))
                writer.open(path_out_video)
                # cursor list: [AsynchronousCursor,]
                cursor_list = ScanningVideo.getInfoCursorList(
                    objects_list, num_frames, 1, min_frames)[0]['cursor_list']
                # preview dict: {identity: preview,}
                preview_dict = ScanningVideo.getPreviewAsDict(objects_list)
                # masking pipeline
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
                # cursor list: [AsynchronousCursor,]
                cursor_list = ScanningVideo.getInfoCursorList(
                        objects_list, num_frames, num_workers, min_frames)
                # preview dict: {identity: preview,}
                preview_dict = ScanningVideo.getPreviewAsDict(objects_list)
                # masking pipeline
                session = MaskingVideoSession(
                    num_workers, path_in_video, options_dict, cursor_list, schedule_call,
                    with_hair=with_hair, preview_dict=preview_dict, debug_mode=debug_mode)
                session.start()
                session.dump(path_out_video)

