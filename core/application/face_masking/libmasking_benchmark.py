
import logging
import os
import cv2
import numpy as np
import tqdm
from ...base.cache import XPortrait
from ...utils.resource import Resource
from .libmasking_blur import LibMasking_Blur
from .libmasking_mosaic import LibMasking_Mosaic
from .libmasking_sticker import LibMasking_Sticker


def maskingAllWithBlur(cache, path_out, name):
    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Blur.inferenceOnBox_WithBlurGaussian(canvas, cache.box[i, :], 13, 'box')
    cv2.imwrite('{}/{}-blur-1.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Blur.inferenceOnBox_WithBlurGaussian(canvas, cache.box[i, :], 13, 'head')
    cv2.imwrite('{}/{}-blur-2.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Blur.inferenceOnBox_WithBlurMotion(canvas, cache.box[i, :], 13, 'box')
    cv2.imwrite('{}/{}-blur-3.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Blur.inferenceOnBox_WithBlurMotion(canvas, cache.box[i, :], 13, 'head')
    cv2.imwrite('{}/{}-blur-4.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Blur.inferenceOnBox_WithBlurWater(canvas, cache.box[i, :], 4, 4, 'box')
    cv2.imwrite('{}/{}-blur-5.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Blur.inferenceOnBox_WithBlurWater(canvas, cache.box[i, :], 4, 4, 'head')
    cv2.imwrite('{}/{}-blur-6.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Blur.inferenceOnBox_WithBlurDiffusion(canvas, cache.box[i, :], 17, 3, 3, 'box')
    cv2.imwrite('{}/{}-blur-7.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Blur.inferenceOnBox_WithBlurDiffusion(canvas, cache.box[i, :], 17, 3, 3, 'head')
    cv2.imwrite('{}/{}-blur-8.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Blur.inferenceOnBox_WithBlurDiffusion(canvas, cache.box[i, :], 17, 0, 0, 'box')
    cv2.imwrite('{}/{}-blur-9.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Blur.inferenceOnBox_WithBlurDiffusion(canvas, cache.box[i, :], 17, 0, 0, 'head')
    cv2.imwrite('{}/{}-blur-10.png'.format(path_out, name[:-4]), canvas)

def maskingAllWithMosaic(cache, path_out, name):
    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Mosaic.inferenceBoxWithMosaicSquare(canvas, cache.box[i, :], 24, 'box')
    cv2.imwrite('{}/{}-mosaic-1.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Mosaic.inferenceBoxWithMosaicSquare(canvas, cache.box[i, :], 24, 'head')
    cv2.imwrite('{}/{}-mosaic-2.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 8, True, 'box')
    cv2.imwrite('{}/{}-mosaic-3.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 8, True, 'head')
    cv2.imwrite('{}/{}-mosaic-4.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 16, True, 'box')
    cv2.imwrite('{}/{}-mosaic-5.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 16, True, 'head')
    cv2.imwrite('{}/{}-mosaic-6.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 8, False, 'box')
    cv2.imwrite('{}/{}-mosaic-7.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 8, False, 'head')
    cv2.imwrite('{}/{}-mosaic-8.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 16, False, 'box')
    cv2.imwrite('{}/{}-mosaic-9.png'.format(path_out, name[:-4]), canvas)

    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 16, False, 'head')
    cv2.imwrite('{}/{}-mosaic-10.png'.format(path_out, name[:-4]), canvas)

def maskingAllWithSticker(cache, path_out, name):
    sticker = cv2.imread('{}/document/sticker_retro.png'.format(os.path.split(__file__)[0]), cv2.IMREAD_UNCHANGED)
    points = np.array([[857, 1735], [1683, 1735]], dtype=np.int32)
    canvas = np.copy(cache.bgr)
    for i in range(cache.number):
        canvas = LibMasking_Sticker.inferenceWithBox(canvas, cache.box[i, :], dict(bgr=sticker, eyes_center=points))
    cv2.imwrite('{}/{}-sticker-1.png'.format(path_out, name[:-4]), canvas)
    
def benchmark():
    name = 'example.png'
    path_in = '{}/document'.format(os.path.split(__file__)[0])
    path_out = '{}/document/example'.format(os.path.split(__file__)[0])
    bgr = cv2.imread('{}/{}'.format(path_in, name), cv2.IMREAD_COLOR)
    cache = XPortrait(bgr)
    # blur
    maskingAllWithBlur(cache, path_out, name)
    # mosaic
    maskingAllWithMosaic(cache, path_out, name)
    # sticker
    maskingAllWithSticker(cache, path_out, name)
