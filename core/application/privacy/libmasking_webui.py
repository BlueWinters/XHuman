
import shutil
import gradio as gr
from .libmasking import *


class LibMaskingWebUI:
    """
    """
    @staticmethod
    def constructUI():
        interface = gr.Blocks(title='devkit online GUI', theme=gr.themes.Default())
        with interface:
            with gr.Tab('privacy'):
                with gr.Column():
                    with gr.Row(equal_height=True):
                        video_input = gr.Video(label='input video', height=512)
                        with gr.Column():
                            textbox_output_uuid = gr.Textbox(label='process uuid')
                            with gr.TabItem('scanning video'):
                                video_output_scanning = gr.Video()
                            with gr.TabItem('scanning json'):
                                textarea_output_json = gr.TextArea()
                            with gr.TabItem('scanning faces'):
                                gallery_face = gr.Gallery()
                            with gr.TabItem('masking video'):
                                video_output_masking = gr.Video()
                    with gr.Row():
                        num_preview = gr.Textbox('1', label='number of preview frames')
                        fix_num = gr.Textbox('-1', 'specify the number of maximum person')
                    with gr.Row():
                        button_scanning = gr.Button('Scanning')
                        button_masking = gr.Button('Masking')

            button_scanning.click(
                fn=LibMaskingWebUI.interface_Scanning,
                inputs=[video_input, num_preview, fix_num],
                outputs=[textbox_output_uuid, video_output_scanning, textarea_output_json, gallery_face],
            )

            button_masking.click(
                fn=LibMaskingWebUI.interface_Masking,
                inputs=[textbox_output_uuid],
                outputs=[video_output_masking],
            )

    @staticmethod
    def interface_Scanning(path_video, num_preview, fix_num):
        uuid_name, (path_uuid, path_out_json, path_in_video, path_out_video) = Resource.createRandomCacheFile(
            ['', '.json', '-in.mp4', '-out_scan.mp4'])
        shutil.copyfile(path_video, path_in_video)
        parameters = dict(path_out_video=path_out_video, num_preview=int(num_preview), fix_num=int(fix_num))
        video_info = LibMasking.scanning(path_in_video, path_out_json, **parameters)
        return path_uuid, path_out_video, video_info.getInfoJson(False), video_info.getIdentityPreviewList(is_bgr=False)

    @staticmethod
    def interface_Masking(path_uuid):
        path_in_video = '{}-in.mp4'.format(path_uuid)
        path_in_json = '{}.json'.format(path_uuid)
        path_out_video = '{}-out_mask.mp4'.format(path_uuid)
        LibMasking.masking(path_in_video, path_in_json, dict(), path_out_video)
        return path_out_video

