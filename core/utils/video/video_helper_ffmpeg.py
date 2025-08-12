
import logging
import traceback
import os
import subprocess


class XVideoHelperFfmpeg:
    """
    """

    @staticmethod
    def isExist():
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        finally:
            return False

    @staticmethod
    def hasAudioStream(video_path: str) -> bool:
        if os.path.exists(video_path) is False:
            return False

        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path,
        ]
        params = dict(
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        try:
            # result = subprocess.run(cmd, capture_output=True, text=True)
            result = subprocess.run(' '.join(cmd), **params)
            if result.returncode != 0:
                logging.error('error occurred while probing video: {}'.format(result.stderr))
                return False
            # check if there is any output from ffprobe command
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            logging.error('error occurred while probing video: {}'.format(video_path))
        return False
    
    @staticmethod
    def extractAudioFromVideo(path_video: str, path_audio: str, loglevel='warning', verbose=False):
        """
        ffmpeg -i input.mp4 -map 0:a -c:a copy output.aac
        """
        assert os.path.exists(path_video), path_video
        cmd = [
            'ffmpeg',
            '-i', path_video,
            '-map', '0:a',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',
            '-loglevel', loglevel,
            path_audio,
        ]
        params = dict(
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        if verbose is True:
            logging.info(' '.join(cmd))
        try:
            subprocess.run(' '.join(cmd), **params)
            return True
        except subprocess.CalledProcessError as e:
            logging.error('error occurred while extracting audio: {}'.format(path_video))
            return False

    @staticmethod
    def addAudioFromVideo(silent_video_path: str, audio_video_path: str, output_video_path: str, loglevel='warning'):
        """
        ffmpeg -i silent_video_path -i audio_video_path -c:v copy -map 0:v -map 1:a -shortest output_video_path
        """
        assert os.path.exists(silent_video_path), silent_video_path
        assert os.path.exists(audio_video_path), audio_video_path
        cmd = [
            'ffmpeg',
            '-loglevel', loglevel,
            '-y',  # auto over-write
            '-i', silent_video_path,
            '-i', audio_video_path,
            '-map', '0:v',
            '-map', '1:a',
            '-c:v', 'copy',
            '-shortest',
            output_video_path,
        ]
        params = dict(
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        try:
            subprocess.run(' '.join(cmd), **params)
            return True
        except subprocess.CalledProcessError as e:
            logging.error('error occurred: {}'.format(e))
            return False

    @staticmethod
    def getVideoPropertyNumFrames(path_video, loglevel='warning'):
        # ffprobe -loglevel warning -count_frames -select_streams v:0 -show_entries stream=nb_read_frames
        #   -of default=nokey=1:noprint_wrappers=1 input.mp4
        assert os.path.exists(path_video), path_video
        cmd = [
            'ffprobe',
            '-loglevel', loglevel,
            '-count_frames',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_read_frames',
            '-of', 'default=nokey=1:noprint_wrappers=1',
            '-i', path_video,
        ]
        params = dict(
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        try:
            output = subprocess.run(' '.join(cmd), **params)
            result_line = output.stdout.decode('utf-8').split('\n')
            num_frames = int(result_line[0])
            return num_frames
        except Exception as e:
            logging.warning('error occurred: {}'.format(e))
            logging.warning(traceback.print_exc())
            return -1

    @staticmethod
    def getVideoPropertyFPS(path_video, loglevel='warning'):
        # ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,avg_frame_rate -of default=noprint_wrappers=1 input.mp4
        cmd = [
            'ffprobe',
            '-loglevel', loglevel,
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=r_frame_rate,avg_frame_rate',
            '-of',
            'default=noprint_wrappers=1',
            path_video,
        ]
        params = dict(
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        try:
            output = subprocess.run(' '.join(cmd), **params)
            result_list = output.stdout.decode('utf-8').split('\n')
            r_f_line = [line for line in result_list if 'r_frame_rate' in line]
            a_f_line = [line for line in result_list if 'avg_frame_rate' in line]
            assert len(r_f_line) == 1 and len(a_f_line) == 1, output
            list_r_frame_rate = r_f_line[0].split('=')[1].split('/')
            list_avg_frame_rate = a_f_line[0].split('=')[1].split('/')
            return True, list_r_frame_rate, list_avg_frame_rate
        except Exception as e:
            logging.error('error occurred: {}'.format(e))
            logging.error(traceback.print_exc())
            return False, None, None

    @staticmethod
    def checkVideoFPS(path_video, diff_max=3, loglevel='warning') -> bool:
        is_success, list_r_frame_rate, list_avg_frame_rate = XVideoHelperFfmpeg.getVideoPropertyFPS(path_video, loglevel)
        if is_success is True:
            r_frame_rate = float(list_r_frame_rate[0]) / float(list_r_frame_rate[1])
            avg_frame_rate = float(list_avg_frame_rate[0]) / float(list_avg_frame_rate[1])
            return bool(abs(r_frame_rate - avg_frame_rate) <= diff_max)
        else:
            return False

    @staticmethod
    def changeVideoFPS(path_video_in, path_video_out, fps=20, codec='libx264', crf=12, loglevel='warning'):
        if os.path.exists(path_video_in):
            cmd = [
                'ffmpeg',
                '-loglevel', loglevel,
                '-i', path_video_in,
                '-c:v', codec,
                '-crf', str(crf),
                '-r', str(fps),
                path_video_out,
                '-y'
            ]
            params = dict(
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            try:
                subprocess.run(' '.join(cmd), **params)
                return True
            except subprocess.CalledProcessError as e:
                logging.error('error occurred: {}'.format(e))
                return False
        else:
            logging.error('input video do not exist: {}'.format(path_video_in))
            return False

    @staticmethod
    def concatVideoLowQuality(input_file_list, output_file, audio_file=None, loglevel='warning', verbose=False):
        content = ''
        for name in input_file_list:
            content += "file '{}'\n".format(name)
        video_string = '|'.join(input_file_list)
        if isinstance(audio_file, str) and os.path.exists(audio_file):
            cmd = [
                'ffmpeg',
                '-loglevel', loglevel,
                '-i', '"concat:{}"'.format(video_string),
                '-i', audio_file,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-y',
                output_file,
            ]
        else:
            cmd = [
                'ffmpeg',
                '-loglevel', loglevel,
                '-i', '"concat:{}"'.format(video_string),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-y',
                output_file,
            ]
        params = dict(
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        if verbose is True:
            logging.info(' '.join(cmd))
        try:
            subprocess.run(' '.join(cmd), **params)
            return True
        except subprocess.CalledProcessError as e:
            logging.error('error occurred: {}'.format(e))
            return False

    @staticmethod
    def concatVideo(input_file_list, output_file, audio_file=None, loglevel='warning', verbose=False):
        content = ''
        for name in input_file_list:
            content += "file '{}'\n".format(name)
        video_string = '|'.join(input_file_list)
        if isinstance(audio_file, str) and os.path.exists(audio_file):
            cmd = [
                'ffmpeg',
                '-loglevel', loglevel,
                '-i', '"concat:{}"'.format(video_string),
                '-i', audio_file,
                '-map', '0:v', '-map', '1:a',
                '-c:v', 'libx264',
                '-crf', '20',
                '-preset', 'ultrafast',
                '-pix_fmt', 'yuv420p',
                '-async', '1',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-strict', 'experimental',
                '-shortest',
                '-y',
                output_file
            ]
        else:
            cmd = [
                'ffmpeg',
                '-loglevel', loglevel,
                '-i', '"concat:{}"'.format(video_string),
                '-c:v', 'libx264',
                '-crf', '20',
                '-preset', 'ultrafast',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'copy',
                '-b:a', 'copy',
                '-y',
                '-strict', 'experimental',
                '-vsync', '0', 
                output_file,
            ]
        params = dict(
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        if verbose is True:
            logging.info(' '.join(cmd))
        try:
            subprocess.run(' '.join(cmd), **params)
            return True
        except subprocess.CalledProcessError as e:
            logging.error('error occurred: {}'.format(e))
            return False

    @staticmethod
    def splitVideo(path_video_in, path_video_out, index_beg, index_end, loglevel='warning'):
        # index start from 0, eg. 20-->21-th
        # ffmpeg -i ./input.mp4 -vf "select=between(n\,20\,200)" -y -acodec copy ./output.mp4
        if os.path.exists(path_video_in):
            cmd = [
                'ffmpeg',
                '-loglevel', loglevel,
                '-i', path_video_in,
                '-vf', '"select=between(n\\,{}\\,{})"'.format(index_beg, index_end),
                '-y',
                '-acodec', 'copy',
                path_video_out,
            ]
            params = dict(
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            try:
                subprocess.run(' '.join(cmd), **params)
                return True
            except subprocess.CalledProcessError as e:
                logging.error('error occurred: {}'.format(e))
                return False
        else:
            logging.error('input video do not exist: {}'.format(path_video_in))
            return False

    @staticmethod
    def reformatVideo(path_video_in, path_video_out, auto_remove=False, loglevel='warning', verbose=True):
        assert os.path.exists(path_video_in), path_video_in
        suffix_in = os.path.splitext(path_video_in)
        suffix_out = os.path.splitext(path_video_out)
        if suffix_in == suffix_out:
            logging.warning('skip reformat: {} --> {}'.format(path_video_in, path_video_out))
            return None
        cmd = [
            'ffmpeg',
            '-loglevel', loglevel,
            '-y',
            '-i', path_video_in,
            path_video_out,
        ]
        params = dict(
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        try:
            subprocess.run(' '.join(cmd), **params)
            if os.path.exists(path_video_out) and auto_remove is True:
                os.remove(path_video_in)
            if verbose is True:
                logging.warning('reformat video: {}'.format(cmd))
            return True
        except subprocess.CalledProcessError as e:
            logging.error('error occurred: {}'.format(e))
            return False


