
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


def getModules(names: Union[str, list]):
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


def getModuleFunction(name: str, function):
    """
    算法调用接口
    """
    module = getModules(name)
    handle = getattr(module, function)
    if isinstance(handle, staticmethod):
        logging.warning('{} is not staticmethod'.format(handle))
    return handle


def getResultsFromFunctions(name, function, *args, **kwargs):
    """
    算法调用接口
    """

    configLogging()
    module = getModules(name)
    return getattr(module, function)(*args, **kwargs)

