
import os
import logging
import cv2
import numpy as np
import tqdm
from .video_reader import *
from .video_writer import *



class XVideoHelper:
    """
    """
    @staticmethod
    def saveReformat(path_video_source, path_video_target, **kwargs):
        def replaceSuffix(path, suffix):
            return '{}.{}'.format(os.path.splitext(path)[0], suffix)

        assert os.path.exists(path_video_source), path_video_source
        reader_source = XVideoReader(path_video_source)
        config = reader_source.desc(True)
        config['fourcc'] = XVideoWriter.default_fourcc()
        path_video_target = replaceSuffix(path_video_target, 'mp4')
        if 'path_video_format' in kwargs:
            path_video_format = kwargs['path_video_format']
            assert os.path.exists(path_video_format), path_video_format
            config['fourcc'] = XVideoReader(path_video_format).fourcc
            path_video_target = replaceSuffix(path_video_target, os.path.splitext(path_video_format)[-1][1:])
        writer_target = XVideoWriter(config)
        writer_target.open(path_video_target)
        desc = kwargs.pop('desc', 'reformat video: {} --> {}'.format(reader_source.fourcc, config['fourcc']))
        # print('reformat video: {} --> {}'.format(reader_source.fourcc, reader_format.fourcc))
        with tqdm.tqdm(total=reader_source.num_frame, desc=desc, unit='image') as bar:
            for bgr in reader_source:
                writer_target.write(bgr)
                bar.update(1)

    @staticmethod
    def captureVideoPart(path_video_or_reader, time_beg, time_end, pos_lft, pos_rig, pos_top, pos_bot, **kwargs):
        assert isinstance(path_video_or_reader, (XVideoReader, str)), type(path_video_or_reader)
        reader = XVideoReader(path_video_or_reader) if isinstance(path_video_or_reader, str) else path_video_or_reader
        if reader.isOpen():
            assert 0 <= time_beg < time_end <= reader.num_frame, \
                (time_beg, time_end, reader.num_frame)
            assert 0 <= pos_lft < pos_rig <= reader.w, (pos_lft, pos_rig, reader.w)
            assert 0 <= pos_top < pos_bot <= reader.h, (pos_top, pos_bot, reader.h)
            reader.resetPositionByIndex(time_beg)
            num_cap_frames = time_end - time_beg
            if 'path_save_images' in kwargs:
                path_save_images = kwargs['path_save_images']
                assert os.path.isdir(path_save_images), path_save_images
                # function = lambda n, bgr: cv2.imwrite('{}/{:04d}.png'.format(path_save_images, n), bgr[pos_top:pos_bot, pos_lft:pos_rig, :])
                # map(function, range(num_cap_frames), reader)
                desc = kwargs.pop('desc', 'capture video into images')
                with tqdm.tqdm(total=num_cap_frames, desc=desc, unit='image') as bar:
                    for n in range(num_cap_frames):
                        _, bgr = reader.read()
                        cv2.imwrite('{}/{:04d}.png'.format(path_save_images, n), bgr[pos_top:pos_bot, pos_lft:pos_rig, :])
                        bar.update(1)
                return None
            if 'path_save_video' in kwargs:
                path_save_video = kwargs['path_save_video']
                writer = XVideoWriter(reader.desc(False))
                writer.open(path_save_video)
                # function = lambda n, bgr: writer.write(bgr[pos_top:pos_bot, pos_lft:pos_rig, :])
                # iter = map(function, range(num_cap_frames), reader)
                desc = kwargs.pop('desc', 'capture video into video')
                with tqdm.tqdm(total=num_cap_frames, desc=desc, unit='image') as bar:
                    for n in range(num_cap_frames):
                        _, bgr = reader.read()
                        writer.write(bgr[pos_top:pos_bot, pos_lft:pos_rig, :])
                        bar.update(1)
                return None
            raise ValueError('path_save_images or path_save_video not in kwargs')
        else:
            raise IOError('source video do not open successful: {}'.format(path_video_or_reader))

    @staticmethod
    def splitVideoByTime(path_video, path_save, **kwargs):
        assert os.path.exists(path_video), path_video
        reader = XVideoReader(path_video)
        if reader.isOpen():
            num_frames = kwargs.pop('num_frames', 0)
            if not (0 < num_frames < reader.num_frame):
                if 'num_seconds' in kwargs:
                    num_seconds = kwargs['num_seconds']
                    num_seconds_all = reader.num_frame / reader.fps
                    if 0 < num_seconds < int(num_seconds_all):
                        num_frames = reader.fps * num_seconds
                if 'num_videos' in kwargs:
                    num_videos = kwargs['num_videos']
                    if 0 < num_videos < reader.num_frame:
                        num_frames = int(reader.num_frame / num_videos)  # the last video is bigger
            # split video
            num_frames_per = num_frames
            num_videos = int(np.ceil(reader.num_frame / num_frames_per))
            assert os.path.isdir(path_save), path_save
            prefix, suffix = reader.prefix(), reader.suffix()
            desc = kwargs.pop('desc', 'split video by time')
            with tqdm.tqdm(total=num_videos, desc=desc, unit='image') as bar:
                for n in range(0, num_videos):
                    beg = (n + 0) * num_frames_per
                    end = (n + 1) * num_frames_per
                    end = min(end, reader.num_frame)
                    path = '{}/{}-{:02d}.{}'.format(path_save, prefix, n+1, suffix)
                    XVideoHelper.captureVideoPart(reader, beg, end, 0, reader.w, 0, reader.h, path_save_video=path)
                    bar.update(1)
        else:
            raise IOError('source video do not open successful: {}'.format(path_video))

    @staticmethod
    def splitVideoByImage(path_video_in: str, path_video_out: str, num_split: int, axis: int = 0, **kwargs):
        assert os.path.isfile(path_video_in), path_video_in
        assert os.path.isdir(path_video_out), path_video_out
        assert 1 < num_split < 32, num_split
        assert axis == 0 or axis == 1, axis
        reader = XVideoReader(path_video_in)
        writer_list = []
        for n in range(num_split):
            writer = XVideoWriter(reader.desc(with_hw=False))
            full_name = os.path.split(path_video_in)[1]
            pure_name, ext = os.path.splitext(full_name)
            writer.open('{}/{}-{}.{}'.format(path_video_out, pure_name, n, ext[1:]))
            writer_list.append(writer)
        desc = kwargs.pop('desc', 'concatenate videos')
        with tqdm.tqdm(total=reader.num_frame, desc=desc, unit='image') as bar:
            for frame in reader:
                parts = np.split(frame, num_split, axis=axis)
                for n, each in enumerate(parts):
                    writer_list[n].write(each)
                bar.update(1)

    @staticmethod
    def concatenateVideosByImage(path_in_list, path_out, axis:int=0, **kwargs):
        assert axis == 0 or axis == 1
        reader_list = list()
        config = None
        for path_in in path_in_list:
            reader = XVideoReader(path_in)
            assert reader.isOpen()
            if config is not None:
                current = reader.desc(True)
                assert config['h'] == current['h']
                assert config['w'] == current['w']
                assert config['num_frames'] == current['num_frames'], (config['num_frames'], current['num_frames'])
            else:
                config = reader.desc(True) if config is None else config
            reader_list.append(reader)
        axis_str = 'h' if axis == 0 else 'w'
        config[axis_str] *= len(reader_list)
        writer = XVideoWriter(config)
        writer.open(path_out)
        desc = kwargs.pop('desc', 'concatenate videos')
        with tqdm.tqdm(total=config['num_frames'], desc=desc, unit='image') as bar:
            for n, frames in enumerate(zip(*reader_list)):
                image = np.concatenate(frames, axis=axis)
                writer.write(np.ascontiguousarray(image))
                bar.update(1)

    @staticmethod
    def concatenateVideosByTime(path_in_list, path_out, **kwargs):
        reader_list = list()
        config = None
        sum_frames = 0
        for path_in in path_in_list:
            reader = XVideoReader(path_in)
            assert reader.isOpen()
            config = reader.desc() if config is None else config
            assert config == reader.desc(), (config, reader.desc())
            reader_list.append(reader)
            sum_frames += reader.desc()['num_frames']
        writer = XVideoWriter(config)
        writer.open(path_out)
        desc = kwargs.pop('desc', 'concatenate videos by time')
        with tqdm.tqdm(total=sum_frames, desc=desc, unit='image') as bar:
            for reader in reader_list:
                for bgr in reader:
                    writer.write(bgr)
                    bar.update(1)

    """
    merge some images into one video
    """
    @staticmethod
    def mergeFromImages(path_images:str, path_video:str, config=dict(), **kwargs):
        h, w = cv2.imdecode(np.fromfile(path_images[0], dtype=np.uint8), cv2.IMREAD_COLOR).shape[:2]
        writer = XVideoWriter(dict(h=h, w=w, **config))
        writer.open(path_video)
        desc = kwargs.pop('desc', 'merge images into video')
        with tqdm.tqdm(total=len(path_images), desc=desc, unit='image') as bar:
            for n, path in enumerate(path_images):
                assert os.path.exists(path), path
                image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                writer.write(image)
                bar.update(1)

    """
    """
    @staticmethod
    def resizeResolution(path_video_in:str, path_video_out:str, new_height=0.5, new_width=0.5, **kwargs):
        assert os.path.isfile(path_video_in)
        assert new_height > 0 and new_width > 0, (new_height, new_width)
        reader = XVideoReader(path_video_in)
        if reader.isOpen() is False:
            logging.warning('open video fail...')
        config = reader.desc()
        h = config['h'] = int(round(new_height))
        w = config['w'] = int(round(new_width))
        writer = XVideoWriter(config)
        writer.open(path_video_out)
        desc = kwargs.pop('desc', 'size video')
        with tqdm.tqdm(total=reader.num_frame, desc=desc, unit='image') as bar:
            for bgr in reader:
                writer.write(cv2.resize(bgr, (w, h)))
                bar.update(1)
