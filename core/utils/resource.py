
import logging
import os
import time
import cv2
import numpy as np
import uuid


class Resource:
    """
    """
    ResourcesDict = dict()
    CacheFolder = 'cache'

    """
    """
    def __init__(self):
        pass

    @staticmethod
    def loadImage(path):
        if path not in Resource.ResourcesDict:
            assert os.path.exists(path), path
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            image.setflags(write=False)
            Resource.ResourcesDict[path] = image
        return Resource.ResourcesDict[path]

    @staticmethod
    def createTimeStampFolder():
        path_exec = os.getcwd()
        time_stamp = int(round(time.time() * 1000))
        time_day_string = time.strftime('%Y-%m-%d', time.localtime(time_stamp / 1000))
        path_folder = '{}/{}/{}'.format(path_exec, Resource.CacheFolder, time_day_string)
        os.makedirs(path_folder, exist_ok=True)
        return path_folder

    @staticmethod
    def createRandomCacheFile(suffixes=''):
        path_folder_time = Resource.createTimeStampFolder()
        uuid_name = str(uuid.uuid4().hex)
        if isinstance(suffixes, str):
            return uuid_name, '{}/{}{}'.format(path_folder_time, uuid_name, suffixes)
        if isinstance(suffixes, list):
            return uuid_name, ['{}/{}{}'.format(path_folder_time, uuid_name, suffix) for suffix in suffixes]
        raise NotImplementedError(suffixes)

    @staticmethod
    def createRandomCacheFolder():
        path_folder_time = Resource.createTimeStampFolder()
        path_folder_random = '{}/{}'.format(path_folder_time, str(uuid.uuid4().hex))
        os.makedirs(path_folder_random, exist_ok=True)
        return path_folder_random
