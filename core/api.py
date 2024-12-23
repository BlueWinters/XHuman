
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
        >>> blur_options = (0, 15)  # 15是表示模糊核大小，越大模糊程度越严重，越大速度越慢
        >>> mosaic_options = (1, 7)  # 7是表示人脸的像素个数，越小模糊程度越严重，对速度影响不大
        >>> sticker_options = (2, cv2.imread('path_to_sticker'))  # sticker贴纸素材
        >>> for identity in list(preview_dict.keys()):
        >>>     options = random.choice([blur_options, mosaic_options, sticker_options])  # 根据用户选择一个
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
        >>> preview_dict = video_info.getIdentityPreviewDict(size=256, is_bgr=True)  # 返回dict，key是人脸id(int)，value是一张人脸图片(array)，即{int: np.ndarray}
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

