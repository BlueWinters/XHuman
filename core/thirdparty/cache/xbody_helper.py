
import logging
import os
import cv2
import numpy as np
import typing
from .xbody import XBody


class XBodyHelper:
    @staticmethod
    def dumpXBodyFromFolder(path_dir_in, path_dir_out, suffix='.png'):
        for name in sorted(os.listdir(path_dir_in)):
            if name.endswith(suffix):
                bgr = cv2.imread('{}/{}'.format(path_dir_in, name))
                cache = XBody(bgr)
                path_pkl = '{}/{}.pkl'.format(path_dir_out, os.path.splitext(name)[0])
                cache.save(path_pkl, name_list=['bgr', 'number', 'scores', 'boxes', 'scores26', 'points26'])
            else:
                logging.warning('skip file: {}'.format(name))

