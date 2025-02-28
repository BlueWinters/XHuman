
import os
import numpy as np
from ....utils import Resource


def packageStickerAlignPoints(align_type):
    path_sticker = '{}/align_points/{}/'.format(os.path.split(__file__)[0], align_type)
    data = []
    for name in sorted(os.listdir(path_sticker)):
        bgr = Resource.loadImage('{}/{}'.format(path_sticker, name))
        int_list = [int(v) for v in name[:-4].split('-')[1][1:-1].split(',')]
        points = np.reshape(np.array(int_list, dtype=np.int32), (2, 2))
        data.append({'sticker': bgr, align_type: points})
    return data


def getResourceStickerAlignPoints(align_type, prefix):
    path_sticker = '{}/align_points/{}/'.format(os.path.split(__file__)[0], align_type)
    file_name_list = sorted(os.listdir(path_sticker))
    prefix_name_list = [name.split('-')[0] for name in file_name_list]
    # assert prefix in prefix_name_list, (prefix, prefix_name_list)
    index = prefix_name_list.index(prefix)
    name = file_name_list[index]
    bgr = Resource.loadImage('{}/{}'.format(path_sticker, name))
    int_list = [int(v) for v in name[:-4].split('-')[1][1:-1].split(',')]
    points = np.reshape(np.array(int_list, dtype=np.int32), (2, 2))
    return bgr, points


def packageStickerAlignBox(align_type):
    path_sticker = '{}/align_box/{}/'.format(os.path.split(__file__)[0], align_type)
    data = []
    for name in sorted(os.listdir(path_sticker)):
        bgr = Resource.loadImage('{}/{}'.format(path_sticker, name))
        data.append({'sticker': bgr})
    return data


def getResourceStickerCustom(prefix):
    path_sticker = '{}/custom'.format(os.path.split(__file__)[0])
    file_name_list = sorted(os.listdir(path_sticker))
    prefix_name_list = [name.split('.')[0] for name in file_name_list]
    # assert prefix in prefix_name_list, (prefix, prefix_name_list)
    index = prefix_name_list.index(prefix)
    name = file_name_list[index]
    bgr = Resource.loadImage('{}/{}'.format(path_sticker, name))
    return bgr
