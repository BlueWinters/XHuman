
import logging
import functools
from typing import Union
from . import XManager


"""
interface
"""
@functools.lru_cache(maxsize=1)
def configLogging():
    format = '%(asctime)s - Lv%(levelno)s - %(filename)s:%(lineno)d - %(message)s'
    logging.basicConfig(level=logging.INFO, format=format, datefmt='%Y-%m-%d %H:%M:%S')

def printModuleInfo():
    """
    输出当前锁包含的所有模块名称
    """
    XManager.printModuleInfo()


def checkImportAuto():
    """
    创建并初始化所有的模型，用于自动检查模型配置是否正确
    """
    XManager.checkImportAuto()


def setRootParameters(root:str):
    """
    设置模型参数的路径
    """
    XManager.setRootParameters(root)


def setCommonDevice(device:str):
    """
    设置模块运行设备，仅限cuda:n或者cpu（其中n表示GPU编号：例如0）
    """
    XManager.setCommonDevice(device)


def releaseModules(name:str=''):
    """
    释放指定的模块，当name为空时，释放所有的模块（用于清理GPU缓存）
    """
    XManager.releaseModules(name)


def getModules(names:Union[str, list]):
    """
    获取指定的模块
    1.基本方法：
        >>> module = XManager.getModules('name')
    2.特别方法:
        该方法用于指定非缺省初始化时模块的获取方法
        >>> modules_with_options = [
        >>>    ('name1', {'root': 'path_to_root', 'device':'cpu'}),
        >>>    ('name2', {'parameters': 'path_to_param', 'device':'cuda:0'}),
        >>> ]
        >>> module1, module2 = XManager.getModules(modules_with_options)
    """
    return XManager.getModules(names)


def getResultsFromCaller(name:str, *args, **kwargs):
    """
    算法调用接口
    """

    configLogging()
    module = getModules(name)
    return module(*args, **kwargs)


def getResultsFromFunctions(name, function, *args, **kwargs):
    """
    算法调用接口
    - App算法：图片人脸打码
        >>> import cv2, random, numpy
        >>> # 1.获取人脸数量和人脸的小头像
        >>> bgr_input = cv2.imread('path_to_image', cv2.IMREAD_COLOR)
        >>> # path_out_json = 'path_to_cache_json'
        >>> options_for_scanning_image = dict(max_num=-1)  # max_num表示最大人脸数量
        >>> video_info, _ = getResultsFromFunctions('face_masking', 'scanningImage', bgr_input, **options_for_scanning_image)
        >>> preview_dict = video_info.getIdentityPreviewDict(size=256, is_bgr=True)  # 返回dict，key是人脸id(int)，value是一张人脸图片(array)，即{int: np.ndarray}
        >>> vido_info_string_to_save = video_info.getInfoJson(True)  # 返回中间信息(str)用于保存
        >>> # 2.交互获得用户打码选择
        >>> masking_options_dict = dict()
        >>> # focus_type只能是head或者box，head表示只头部，box表示
        >>> options_blur_gaussian = (101, dict(blur_type='blur_gaussian', focus_type='head'))  # 其他：blur_kernel=15是表示模糊核大小，越大模糊程度越严重，越大速度越慢
        >>> options_blur_motion = (102, dict(blur_type='blur_motion', focus_type='head'))  # 其他：blur_kernel=15是表示模糊核大小，越大模糊程度越严重，越大速度越慢
        >>> options_blur_water = (103, dict(blur_type='blur_water', focus_type='head'))  # 其他：rotation_degree=2表示水波纹旋转度，water_length=8表示水波纹的长度
        >>> options_blur_pencil = (104, dict(blur_type='blur_pencil', focus_type='head'))  # 其他：k_neigh=17表示扩散程度, pre_k=3表示模糊度，post_k=3表示模糊度
        >>> options_blur_diffuse = (105, dict(blur_type='blur_diffuse', focus_type='head'))  # 其他：k_neigh=17表示扩散程度
        >>> options_mosaic_square = (201, dict(mosaic_type='mosaic_pixel_square', focus_type='head'))  # 其他：num_pixel=16表示像素个数，越小越模糊
        >>> options_mosaic_polygon = (202, dict(mosaic_type='mosaic_pixel_polygon', focus_type='head'))  # 其他：n_div=16表示像素个数，越小越模糊，vis_boundary=False表示不显示边界线
        >>> # options_mosaic_polygon一些特别的设置；focus_type可设置为'head'或者'face'或者'box'
        >>> options_mosaic_polygon_small = (202, dict(mosaic_type='mosaic_pixel_polygon_small', focus_type='head'))  # 很少像素+没有边界线
        >>> options_mosaic_polygon_small_line = (202, dict(mosaic_type='mosaic_pixel_polygon_small_line', focus_type='head'))  # 很少像素+有边界线
        >>> options_mosaic_polygon_big = (202, dict(mosaic_type='mosaic_pixel_polygon_big', focus_type='head'))  # 较多像素+没有边界线
        >>> options_mosaic_polygon_big_line = (202, dict(mosaic_type='mosaic_pixel_polygon_big_line', focus_type='head'))  # 较多像素+有边界线
        >>> options_sticker_pure = (301, cv2.imread('path_to_sticker'))  # 纯粹的贴纸素材，4通道图
        >>> options_sticker_align1 = (301, dict(bgr=cv2.imread('path_to_sticker'), eyes_center=np.array(...)))  # 贴纸素材+对应的点
        >>> options_sticker_align2 = (301, dict(bgr=cv2.imread('path_to_sticker'), box=...))  # 定制卡通：贴纸素材+定制卡通裁剪框Box
        >>> options_sticker_align3 = (301, dict(bgr=cv2.imread('path_to_sticker'), align=True))  # 定制卡通：贴纸素材+定制卡通（对齐方法）
        >>> # 给每个人随机赋一个打码类型
        >>> for identity in list(preview_dict.keys()):
        >>>     options = random.choice([options_blur_gaussian, options_mosaic_square, options_sticker_pure])  # 根据用户选择一个
        >>>     code, parameters = options
        >>>     masking_options_dict[identity] = getResultsFromFunctions('face_masking', 'getMaskingOption', code, parameters)
        >>> # 3.获得结果
        >>> video_info_string = vido_info_string_to_save  # 加载vido_info_string_to_save保存的信息
        >>> bgr_output:numpy.ndarray = getResultsFromFunctions('face_masking', 'maskingImage', bgr_input, masking_options_dict, video_info_string=video_info_string)
    - App算法：视频人脸打码
        >>> # 1.获取人脸数量和人脸的小头像
        >>> path_video_input = 'path_video_input'
        >>> options_for_scanning_video = dict()
        >>> options_for_scanning_video['fixed_num'] = 5  # fixed_num表示固定最大人脸数量
        >>> options_for_scanning_video['num_preview'] = 1  # num_preview针对fixed_num等于-1时候，通过预览1帧来确定fixed_num；0<num_previe<=8
        >>> video_info = getResultsFromFunctions('face_masking', 'scanningVideo', path_video_input, path_out_json, **options_for_scanning_image)
        >>> preview_dict = video_info.getIdentityPreviewDict(size=256, is_bgr=True)  # 返回dict，key是人脸id(int)，value是一个dict:包含box和对应帧的图片，
        >>>                                                                          # 即{int: dict(box=box, image=np.ndarray, face=np.ndarray)}
        >>> vido_info_string_to_save = video_info.getInfoJson(True)  # 返回中间信息(str)用于保存
        >>> # 2.交互获得用户打码选择（方法同上）
        >>> # 3.获得结果
        >>> path_video_out = 'path_video_out'
        >>> video_info_string = vido_info_string_to_save  # 加载vido_info_string_to_save保存的信息
        >>> getResultsFromFunctions('face_masking', 'maskingVideo', bgr_input, masking_options_dict, path_video_out, video_info_string=video_info_string)
    """

    configLogging()
    module = getModules(name)
    return getattr(module, function)(*args, **kwargs)

