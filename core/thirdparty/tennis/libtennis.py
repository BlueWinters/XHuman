
import logging
import cv2
import os
import json
import numpy as np
from .libtennis_ball_tracking import LibTennisBallTracking
from .libtennis_ball_bounce import LibTennisBallBounceDetector
from .libtennis_court import LibTennisCourtTracker
from .libtennis_minicourt import LibTennisMiniCourt
from ...utils.video.video_helper_opencv import XVideoWriterOpenCV, XVideoReaderOpenCV


class LibTennis:
    @staticmethod
    def packageAsStream(frames_stream):
        if isinstance(frames_stream, list):
            assert isinstance(frames_stream[0], np.ndarray), 'frames_stream should be list of numpy.ndarray'
            return frames_stream
        if isinstance(frames_stream, str):
            assert os.path.exists(frames_stream), 'frames_stream should be path to video file'
            from core.utils.video import XVideoReaderOpenCVCache
            return XVideoReaderOpenCVCache(frames_stream)
        raise TypeError('frames_stream should be path to video file or list of numpy.ndarray')

    def __init__(self, *args, **kwargs):
        self.ball_tracker = LibTennisBallTracking()
        self.court_tracker = LibTennisCourtTracker()
        self.ball_bounce = LibTennisBallBounceDetector()
        self.mini_court = LibTennisMiniCourt()

    def __del__(self):
        # logging.warning('delete module {}'.format(self.__class__.__name__))
        pass

    def initialize(self, *args, **kwargs):
        self.ball_tracker.initialize(*args, **kwargs)
        self.court_tracker.initialize(*args, **kwargs)
        self.ball_bounce.initialize(*args, **kwargs)

    def visBallTracking(self, path_video, path_video_out, path_json_out=None, trace=7):
        ball_track, json_string = self.ball_tracker(path_video, targets=['source', 'json'])
        # write to json
        if path_json_out is not None:
            with open(path_json_out, 'w') as file:
                file.write(json_string)
        # visual to video
        reader = XVideoReaderOpenCV(path_video)
        writer = XVideoWriterOpenCV(reader.desc(True))
        writer.open(path_video_out)
        for num, frame in enumerate(reader):
            for i in range(trace):
                if num - i > 0:
                    if ball_track[num - i][0]:
                        x = int(ball_track[num - i][0])
                        y = int(ball_track[num - i][1])
                        frame = cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness=10 - i)
                    else:
                        break
            writer.write(frame)

    def analysisGame(self, path_video, path_json_out=None):
        # ball tracking
        ball_track = self.ball_tracker(path_video, targets='source')
        # court tracking
        court_points, inv_mat = self.court_tracker(path_video, targets='mean')
        # detect bounce
        bounces = self.ball_bounce(ball_track)
        # writer to json
        if path_json_out is not None:
            ball_trace = [dict(index=n, x=pos[0], y=pos[1]) for n, pos in enumerate(ball_track)]
            data = dict(
                ball_trace=ball_trace,
                court_points=court_points.tolist(),
                court_inv_mat=inv_mat.tolist(),
                bounces=list(bounces),
            )
            with open(path_json_out, 'w') as file:
                file.write(json.dumps(data, indent=4))
        return ball_track, court_points, inv_mat, bounces

    def visGame(self, path_video, path_video_out, path_json_out=None, trace=7):
        if path_json_out is not None and os.path.exists(path_json_out):
            data = json.load(open(path_json_out, 'r'))
            ball_track = [[pos['x'], pos['y']] for pos in data['ball_trace']]
            court_points = np.array(data['court_points'], dtype=np.float32)
            court_inv_mat = np.array(data['court_inv_mat'], dtype=np.float32)
            bounces = data['bounces']
        else:
            ball_track, court_points, court_inv_mat, bounces = self.analysisGame(path_video)

        map_w = 166
        map_h = 350
        court_image = self.mini_court.build_court_reference()
        reader = XVideoReaderOpenCV(path_video)
        writer = XVideoWriterOpenCV(reader.desc(True))
        writer.open(path_video_out)
        height, width = reader.h, reader.w
        for num, frame in enumerate(reader):
            # ball tracking
            for i in range(trace):
                if num - i > 0:
                    if ball_track[num - i][0]:
                        x = int(ball_track[num - i][0])
                        y = int(ball_track[num - i][1])
                        frame = cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness=10 - i)
                    else:
                        break
            # court tracking
            for i in range(14):
                x = int(round(court_points[i][0]))
                y = int(round(court_points[i][1]))
                frame = cv2.circle(frame, (x, y), radius=0, color=(255, 0, 0), thickness=10)
            # minimap
            if num in bounces and court_inv_mat is not None:
                ball_point = ball_track[num]
                ball_point = np.reshape(np.array(ball_point, dtype=np.float32), (1, 1, 2))
                ball_point = cv2.perspectiveTransform(ball_point, court_inv_mat)
                x, y = int(round(ball_point[0][0][0])), int(round(ball_point[0][0][1]))
                court_image = cv2.circle(court_image, (x, y), radius=0, color=(0, 255, 255), thickness=50)
            mini_map = cv2.resize(court_image, (map_w, map_h))
            frame_part = np.copy(frame[30:(30 + map_h), (width - 30 - map_w):(width - 30), :])
            frame[30:(30 + map_h), (width - 30 - map_w):(width - 30), :] = cv2.addWeighted(frame_part, 0.5, mini_map, 0.5, 0)
            # show & write
            cv2.imshow('show', frame)
            cv2.waitKey(1)
            writer.write(frame)
