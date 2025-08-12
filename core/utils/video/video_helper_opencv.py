
import os
import logging
import cv2
import numpy as np
import tqdm
from .video_reader_opencv import XVideoReaderOpenCV
from .video_writer_opencv import XVideoWriterOpenCV


class XVideoHelperOpenCV:
    """
    """
    @staticmethod
    def saveReformat(path_video_source, path_video_target, **kwargs):
        def replaceSuffix(path, suffix):
            return '{}.{}'.format(os.path.splitext(path)[0], suffix)

        assert os.path.exists(path_video_source), path_video_source
        reader_source = XVideoReaderOpenCV(path_video_source)
        config = reader_source.desc(True)
        path_video_target = replaceSuffix(path_video_target, 'mp4')
        if 'path_video_format' in kwargs:
            path_video_format = kwargs['path_video_format']
            assert os.path.exists(path_video_format), path_video_format
            config['fourcc'] = XVideoReaderOpenCV(path_video_format).fourcc
            path_video_target = replaceSuffix(path_video_target, os.path.splitext(path_video_format)[-1][1:])
        writer_target = XVideoWriterOpenCV(config)
        writer_target.open(path_video_target)
        desc = kwargs.pop('desc', 'reformat video: {} --> {}'.format(reader_source.fourcc, config['fourcc']))
        # print('reformat video: {} --> {}'.format(reader_source.fourcc, reader_format.fourcc))
        with tqdm.tqdm(total=reader_source.num_frame, desc=desc, unit='image') as bar:
            for bgr in reader_source:
                writer_target.write(bgr)
                bar.update(1)

    @staticmethod
    def captureVideoPart(path_video_or_reader, time_beg, time_end, pos_lft, pos_rig, pos_top, pos_bot, **kwargs):
        assert isinstance(path_video_or_reader, (XVideoReaderOpenCV, str)), type(path_video_or_reader)
        reader = XVideoReaderOpenCV(path_video_or_reader) if isinstance(path_video_or_reader, str) else path_video_or_reader
        if reader.isOpen():
            assert 0 <= time_beg < time_end <= reader.num_frame, \
                (time_beg, time_end, reader.num_frame)
            assert 0 <= pos_lft < pos_rig <= reader.w, (pos_lft, pos_rig, reader.w)
            assert 0 <= pos_top < pos_bot <= reader.h, (pos_top, pos_bot, reader.h)
            reader.resetPositionByIndex(time_beg)
            num_cap_frames = time_end - time_beg  # do not include the last
            path_list = []
            if 'path_save_images' in kwargs:
                path_save_images = kwargs['path_save_images']
                assert os.path.isdir(path_save_images), path_save_images
                # function = lambda n, bgr: cv2.imwrite('{}/{:04d}.png'.format(path_save_images, n), bgr[pos_top:pos_bot, pos_lft:pos_rig, :])
                # map(function, range(num_cap_frames), reader)
                desc = kwargs.pop('desc', 'capture video into images')
                with tqdm.tqdm(total=num_cap_frames, desc=desc, unit='image') as bar:
                    for n in range(num_cap_frames):
                        flag, bgr = reader.read()
                        if flag is False:
                            break
                        path = '{}/{:04d}.png'.format(path_save_images, n)
                        cv2.imwrite(path, bgr[pos_top:pos_bot, pos_lft:pos_rig, :])
                        path_list.append(path)
                        bar.update(1)
                return path_list, len(path_list)
            if 'path_save_video' in kwargs:
                path_save_video = kwargs['path_save_video']
                writer = XVideoWriterOpenCV(reader.desc(False))
                writer.open(path_save_video)
                # function = lambda n, bgr: writer.write(bgr[pos_top:pos_bot, pos_lft:pos_rig, :])
                # iter = map(function, range(num_cap_frames), reader)
                desc = kwargs.pop('desc', 'capture video into video')
                with tqdm.tqdm(total=num_cap_frames, desc=desc, unit='image') as bar:
                    count = 0
                    for n in range(num_cap_frames):
                        flag, bgr = reader.read()
                        if flag is False:
                            break
                        writer.write(bgr[pos_top:pos_bot, pos_lft:pos_rig, :])
                        count += 1
                        bar.update(1)
                    path_list.append(writer.path)
                return path_list[0], count
            raise ValueError('path_save_images or path_save_video not in kwargs')
        else:
            raise IOError('source video do not open successful: {}'.format(path_video_or_reader))

    @staticmethod
    def splitVideoByTime(path_video, path_save, **kwargs) -> list:
        assert os.path.exists(path_video), path_video
        reader = XVideoReaderOpenCV(path_video)
        if reader.isOpen():
            num_videos = 0
            index_pair_list = []
            if 'num_frames' in kwargs:
                num_frames = kwargs.pop('num_frames')
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
                num_frames_per = num_frames
                num_videos = int(np.ceil(reader.num_frame / num_frames_per))
                for n in range(0, num_videos):
                    beg = (n + 0) * num_frames_per
                    end = (n + 1) * num_frames_per
                    end = min(end, len(reader))
                    index_pair_list.append((beg, end))
            if 'index_pair_list' in kwargs:
                index_list = kwargs.pop('index_pair_list')
                for index_pair in index_list:
                    beg, end = index_pair
                    assert isinstance(beg, int), beg
                    assert isinstance(end, int), end
                    # index: index beg, exclude end
                    assert 0 <= beg < end <= len(reader), (beg, end, len(reader))
                    index_pair_list.append((beg, end))
                num_videos = len(index_pair_list)
            # split video
            assert os.path.isdir(path_save), path_save
            prefix, suffix = reader.prefix(), reader.suffix()
            desc = kwargs.pop('desc', 'split video by time')
            result_list = []
            with tqdm.tqdm(total=num_videos, desc=desc, unit='image') as bar:
                for n, (beg, end) in enumerate(index_pair_list):
                    path = '{}/{}-{:02d}.{}'.format(path_save, prefix, n+1, suffix)
                    path, counter = XVideoHelper.captureVideoPart(
                        reader, beg, end, 0, reader.w, 0, reader.h, path_save_video=path)
                    result_list.append((path, counter))
                    bar.update(1)
            return result_list
        else:
            raise IOError('source video do not open successful: {}'.format(path_video))

    @staticmethod
    def splitVideoByImage(path_video_in: str, path_video_out: str, num_split: int, axis: int = 0, **kwargs):
        assert os.path.isfile(path_video_in), path_video_in
        assert os.path.isdir(path_video_out), path_video_out
        assert 1 < num_split < 32, num_split
        assert axis == 0 or axis == 1, axis
        reader = XVideoReaderOpenCV(path_video_in)
        writer_list = []
        for n in range(num_split):
            writer = XVideoWriterOpenCV(reader.desc(with_hw=False))
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
        for n, path_in in enumerate(path_in_list):
            reader = XVideoReaderOpenCV(path_in)
            assert reader.isOpen(), 'open video fail: {}'.format(path_in)
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
        writer = XVideoWriterOpenCV(config)
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
        config_global = None
        sum_frames = 0
        for path_in in path_in_list:
            reader = XVideoReaderOpenCV(path_in)
            assert reader.isOpen()
            config_global = reader.desc() if config_global is None else config_global
            config_current = reader.desc()
            assert config_global['w'] == config_current['w'] and \
                   config_global['h'] == config_current['h'], \
                (config_global, config_current)
            reader_list.append(reader)
            sum_frames += reader.desc()['num_frames']
        writer = XVideoWriterOpenCV(config_global)
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
        writer = XVideoWriterOpenCV(dict(h=h, w=w, **config))
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
        reader = XVideoReaderOpenCV(path_video_in)
        if reader.isOpen() is False:
            logging.warning('open video fail...')
        config = reader.desc()
        h = config['h'] = int(round(new_height))
        w = config['w'] = int(round(new_width))
        writer = XVideoWriterOpenCV(config)
        writer.open(path_video_out)
        desc = kwargs.pop('desc', 'size video')
        with tqdm.tqdm(total=reader.num_frame, desc=desc, unit='image') as bar:
            for bgr in reader:
                writer.write(cv2.resize(bgr, (w, h)))
                bar.update(1)

