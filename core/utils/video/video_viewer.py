
import os
import gradio
import numpy as np
from . import XVideoReaderOpenCV, XVideoReaderFFMpeg


class XVideoViewer:
    """
    """
    @staticmethod
    def constructUI():
        interface = gradio.Blocks(title='Video Tools', theme=gradio.themes.Default())
        with interface:
            XVideoViewer.constructUITab()

    @staticmethod
    def constructUITab():
        with gradio.Tab('Video Viewer'):
            with gradio.Row():
                with gradio.Column():
                    input_video = gradio.Video(label='input video', height=512)
                    input_video_path = gradio.Textbox('', interactive=True, show_label=False)
                    with gradio.Row():
                        action_load = gradio.Button('load')
                        action_check = gradio.Button('check')
                        action_get = gradio.Button('get')
                        input_auto_check = gradio.Checkbox(False, label='auto-check')
                        input_number_index = gradio.Number(0, interactive=True, show_label=False, minimum=0)
                with gradio.Tab('json'):
                    output_json = gradio.JSON(dict(data=None))
                with gradio.Tab('image'):
                    output_image = gradio.Image(height=512, interactive=False)

        action_load.click(
            fn=XVideoViewer.actionLoad,
            inputs=input_video_path,
            outputs=input_video,
        )

        action_check.click(
            fn=XVideoViewer.actionCheck,
            inputs=input_video,
            outputs=output_json,
        )

        action_get.click(
            fn=XVideoViewer.actionGet,
            inputs=[input_video, input_number_index, input_auto_check],
            outputs=output_image,
        )

        input_number_index.change(
            fn=XVideoViewer.actionGet,
            inputs=[input_video, input_number_index],
            outputs=output_image,
        )

    @staticmethod
    def actionLoad(path_video):
        assert os.path.exists(path_video), path_video
        return path_video

    @staticmethod
    def actionCheck(path_video, fps_check=True):
        assert os.path.exists(path_video), path_video
        info_video_opencv = XVideoReaderOpenCV(path_video).desc()
        info_video_ffmpeg = XVideoReaderFFMpeg.getVideoInfo(path_video)[0]
        if fps_check is True:
            r_rate, a_rate = info_video_ffmpeg['r_frame_rate'], info_video_ffmpeg['avg_frame_rate']
            info_video_ffmpeg['fps_r_rate'] = '{}/{}'.format(r_rate[0], r_rate[1])
            info_video_ffmpeg['fps_a_rate'] = '{}/{}'.format(a_rate[0], a_rate[1])
            info_video_ffmpeg['fps_r_rate_f'] = float(r_rate[0]) / float(r_rate[1])
            info_video_ffmpeg['fps_a_rate_f'] = float(a_rate[0]) / float(a_rate[1])
        meta_info = dict(opencv=info_video_opencv, ffmpeg=info_video_ffmpeg)
        return meta_info

    @staticmethod
    def actionGet(path_video, n_index, auto_check):
        assert os.path.exists(path_video), path_video
        reader = XVideoReaderOpenCV(path_video)
        if auto_check is True:
            info_video_ffmpeg = XVideoReaderFFMpeg.getVideoInfo(path_video)[0]
            if XVideoReaderFFMpeg.checkFPS(info_video_ffmpeg, 0) is True:
                reader.resetPositionByIndex(n_index)
                ret, bgr = reader.read()
                if ret is True:
                    return np.copy(bgr[:, :, ::-1])
            return np.zeros(shape=(512, 512, 3), dtype=np.uint8)
        else:
            reader.resetPositionByIndex(n_index)
            ret, bgr = reader.read()
            return np.copy(bgr[:, :, ::-1]) if ret is True \
                else np.zeros(shape=(512, 512, 3), dtype=np.uint8)
