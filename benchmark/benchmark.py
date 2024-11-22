
import os
import numpy as np
import cv2


def configLogging():
    import logging
    format = '%(asctime)s - Lv%(levelno)s - %(filename)s:%(lineno)d - %(message)s'
    logging.basicConfig(level=logging.INFO, format=format, datefmt='%Y-%m-%d %H:%M:%S')
# config logging format
configLogging()



class Benchmark_Runtime:
    @staticmethod
    def benchmark_checkFiles():
        def getFolder(path):
            return os.path.split(path)[0]

        from core.checkpoints import checkModelFiles
        path = getFolder(getFolder(__file__))
        checkModelFiles()


class Benchmark_Base:
    @staticmethod
    def benchmark_XPortrait():
        from core.base import XPortrait
        XPortrait.benchmark_property()


class Benchmark_Thirdparty:
    @staticmethod
    def benchmark_YoloX():
        from core.thirdparty import LibYoloX
        LibYoloX.benchmark()

    @staticmethod
    def benchmark_Sapiens():
        from core.thirdparty import LibSapiens
        LibSapiens.benchmark()

    @staticmethod
    def benchmark_RTMPose():
        from core.thirdparty import LibRTMPose
        LibRTMPose.benchmark()

    @staticmethod
    def benchmark_XBody():
        from core.thirdparty import XBody
        XBody.benchmark_property()


if __name__ == '__main__':
    # Benchmark_Runtime.benchmark_checkFiles()

    # Benchmark_Base.benchmark_XPortrait()

    # Benchmark_Thirdparty.benchmark_YoloX()
    # Benchmark_Thirdparty.benchmark_Sapiens()
    # Benchmark_Thirdparty.benchmark_RTMPose()
    Benchmark_Thirdparty.benchmark_XBody()