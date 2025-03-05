# API-Document

## application

### masking_object

```python
import numpy as np
import cv2
from core.api import getModules
module = getModules('face_masking')

# 1.图片打码
# 1.1.获取人脸数量和人脸的小头像
bgr_input = cv2.imread('path_to_image', cv2.IMREAD_COLOR)
options_for_scanning_image = dict(
    category_list=['person', 'plate'],  # 扫描类别
    schedule_call=lambda *_args, **_kwargs: None)  # 进度回调函数
masking_image = module.scanningImage(bgr_input, **options_for_scanning_image)
# summary_dict: 
#   - 类型dict，key是目标id(int)，value是具体内容(dict)
#   - {int: dict(category=str, cartoon_bgr=np.ndarray, cartoon_box=list, preview=np.ndarray)}
#   - 其中，category是类别，preview用于前端展示，cartoon_bgr与cartoon_box卡通的输入（仅category为'person'时有效），text为车牌号（当前不可用）
summary_parameters = dict(person=dict(...), plate=dict(...))  # 缺省即可
summary_dict: dict = masking_image.getImageSummaryAsDict(**summary_parameters)
# 返回中间信息(str)用于保存
data_string_to_save: str = masking_image.getImageInfoAsJson()

# 1.2.交互获得用户打码选择（给每个人赋一个打码类型）
sticker_base_bgr, sticker_base_points = np.ndarray(...), np.ndarray(...)  # 一般贴纸
sticker_cartoon_bgr, sticker_cartoon_box = np.ndarray(...), list(...)  # 卡通贴纸
sticker_custom_bgr = np.ndarray(...)  # 自定义贴纸
masking_options = dict()
for identity in list(summary_dict.keys()):
    # 人脸
    masking_options[identity] = module.getMaskingOption('person_blur_gaussian')
    masking_options[identity] = module.getMaskingOption('person_blur_motion')
    masking_options[identity] = module.getMaskingOption('person_blur_water')
    masking_options[identity] = module.getMaskingOption('person_mosaic_square')
    masking_options[identity] = module.getMaskingOption('person_mosaic_polygon_small')
    masking_options[identity] = module.getMaskingOption('person_mosaic_polygon_small_line')
    masking_options[identity] = module.getMaskingOption('person_mosaic_polygon_big')
    masking_options[identity] = module.getMaskingOption('person_mosaic_polygon_big_line')
    # 卡通类型贴纸
    masking_options[identity] = module.getMaskingOption('person_sticker_cartoon', sticker=sticker_cartoon_bgr, box_tuple=sticker_cartoon_box)
    # 眼罩类型贴纸
    masking_options[identity] = module.getMaskingOption('person_sticker_align_points', sticker=sticker_base_bgr, eyes_center_similarity=sticker_base_points)
    # 口罩类型贴纸
    masking_options[identity] = module.getMaskingOption('person_sticker_align_points', sticker=sticker_base_bgr, mouth_corners_similarity=sticker_base_points)
    # 自定义贴纸
    masking_options[identity] = module.getMaskingOption('person_sticker_custom', sticker=sticker_custom_bgr)
    # 车牌
    masking_options[identity] = module.getMaskingOption('plate_blur_gaussian')
    masking_options[identity] = module.getMaskingOption('plate_mosaic_square')

# 1.3.获得结果
info_string = data_string_to_save  # 加载vido_info_string_to_save保存的信息
bgr_output: np.ndarray = module.maskingImage(bgr_input, masking_options, info_string=info_string, with_hair=True)


# 2.视频人脸打码
# 2.1.获取人脸数量和人脸的预览头像
path_video_input = 'path_video_input'
options_for_scanning_video = dict(
    category_list=['person', 'plate'],  # 扫描类别
    schedule_call=lambda *_args, **_kwargs: None) # 进度回调函数
masking_video = module.scanningVideo(path_video_input, **options_for_scanning_video)
# summary_dict: 
#   - 类型dict，key是人脸id(int)，value是具体内容(dict)
#   - {int: dict(cartoon_bgr=np.ndarray, cartoon_box=list, preview=np.ndarray)}
#   - 其中cartoon_bgr与cartoon_box卡通的输入，preview用于前端展示
summary_parameters = dict(person=dict(...), plate=dict(...))  # 缺省即可
summary_dict: dict = masking_video.getVideoSummaryAsDict(**summary_parameters)
# 返回中间信息(str)用于保存
data_string_to_save: str = masking_video.getVideoInfoAsJson()

# 2.2.交互获得用户打码选择
# 方法同1.2

# 2.3.获得结果
path_video_out = 'path_video_out'
info_string = data_string_to_save  # 加载vido_info_string_to_save保存的信息
module.maskingVideo(bgr_input, masking_options, path_video_out, info_string=info_string, num_workers=4, with_hair=False)
```