
import logging
import os
import subprocess


class FFMPEGHelper:
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
            f'"{video_path}"'
        ]

        try:
            # result = subprocess.run(cmd, capture_output=True, text=True)
            result = subprocess.run(' '.join(cmd), shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                logging.error('Error occurred while probing video: {}'.format(result.stderr))
                return False
            # Check if there is any output from ffprobe command
            return bool(result.stdout.strip())
        except Exception as e:
            logging.error('Error occurred while probing video: {}'.format(video_path))
        return False

    @staticmethod
    def addAudioToVideo(silent_video_path: str, audio_video_path: str, output_video_path: str):
        cmd = [
            'ffmpeg',
            '-y',
            '-i', f'"{silent_video_path}"',
            '-i', f'"{audio_video_path}"',
            '-map', '0:v',
            '-map', '1:a',
            '-c:v', 'copy',
            '-shortest',
            f'"{output_video_path}"'
        ]

        try:
            subprocess.run(' '.join(cmd), shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logging.error('error occurred: {}'.format(e))

    @staticmethod
    def changeVideoFPS(input_file, output_file, fps=20, codec='libx264', crf=12):
        assert os.path.exists(input_file), input_file
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-c:v', codec,
            '-crf', crf,
            '-r', fps,
            output_file,
            '-y'
        ]
        subprocess.run(' '.join(cmd), shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
