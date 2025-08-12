
import logging
import os
import numpy as np
import ffmpeg


class XVideoReaderFFMpeg:
    @staticmethod
    def getVideoInfo(path_video):
        # if probe fail, it will raise error(ffmpeg._run.Error)
        assert os.path.exists(path_video), path_video
        probe = ffmpeg.probe(path_video)
        video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        audio_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'audio')
        return video_info, audio_info

    @staticmethod
    def checkFPS(video_info, diff_max, verbose=False):
        list_r_frame_rate = video_info['r_frame_rate'].split('/')
        list_avg_frame_rate = video_info['avg_frame_rate'].split('/')
        r_frame_rate = float(list_r_frame_rate[0]) / float(list_r_frame_rate[1])
        avg_frame_rate = float(list_avg_frame_rate[0]) / float(list_avg_frame_rate[1])
        if verbose is True:
            logging.info('r_frame_rate: {}={}'.format(video_info['r_frame_rate'], r_frame_rate))
            logging.info('avg_frame_rate: {}={}'.format(video_info['avg_frame_rate'], avg_frame_rate))
        return bool(abs(r_frame_rate - avg_frame_rate) <= diff_max)

    """
    """
    def __init__(self, path_video):
        self.path = path_video
        self.video_info, self.audio_info = self.getVideoInfo(self.path)
        self.w = int(self.video_info['width'])
        self.h = int(self.video_info['height'])
        self.num_frames = int(self.video_info['nb_frames'])
        self.num_sec = float(self.video_info['duration'])
        # average frame rate
        self.avg_frame_rate = eval(self.video_info['avg_frame_rate'])
        # raw frame rate
        self.r_frame_rate = eval(self.video_info['r_frame_rate'])
        # fps
        self.fps = int(self.avg_frame_rate)
        # fourcc
        self.fourcc = str(self.video_info['codec_name'])
        # reader pipe
        self.process = (
            ffmpeg.input(self.path)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run_async(pipe_stdout=True, pipe_stderr=True))
        # reader index
        self.cur = 0

    def suffix(self):
        return os.path.splitext(self.path)[1][1:]

    def prefix(self):
        file_name = os.path.split(self.path)[1]
        return os.path.splitext(file_name)[0]

    def desc(self, with_hw: bool = True) -> dict:
        desc = dict(
            fps=self.fps,
            num_frames=self.num_frames,
            fourcc=self.fourcc,
            num_sec=self.num_sec
        )
        if with_hw is True:
            desc['h'] = self.h
            desc['w'] = self.w
        return desc

    def __len__(self):
        assert self.isOpen() is True
        return self.num_frames

    def isOpen(self) -> bool:
        return self.process is not None

    def release(self):
        if self.process:
            self.process.stdout.close()
            self.process.wait()
            self.process = None

    def resetPositionByIndex(self, index: int):
        if not (0 <= index < self.num_frames):
            raise IndexError('index out of range: {} --> (0,{})'.format(index, self.num_frames))
        raise NotImplementedError

    def read(self):
        frame_size = self.w * self.h * 3  # 3 channels (BGR)
        raw_frame = self.process.stdout.read(frame_size)
        if len(raw_frame) < frame_size:
            self.release()
            return False, None
        frame = np.frombuffer(raw_frame, np.uint8).reshape((self.h, self.w, 3))
        self.cur += 1
        return True, frame

    def __iter__(self):
        return self

    def __next__(self):
        success, frame = self.read()
        if not success:
            raise StopIteration
        return frame



