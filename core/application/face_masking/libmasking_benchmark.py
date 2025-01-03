
import logging
import os
import cv2
import numpy as np
import tqdm
from ...base.cache import XPortrait
from ...thirdparty.cache import XBody
from ...utils.context import XContextTimer
from ...utils.resource import Resource
from .libmasking_blur import LibMasking_Blur
from .libmasking_mosaic import LibMasking_Mosaic
from .libmasking_sticker import LibMasking_Sticker


def maskingWithBlur(canvas, cache, i):
    result = []
    result.append(np.concatenate([LibMasking_Blur.inferenceOnBox_WithBlurGaussian(canvas, cache.box[i, :], 13),
                                  LibMasking_Blur.inferenceOnBox_WithBlurGaussian(canvas, None, 13)], axis=0))
    result.append(np.concatenate([LibMasking_Blur.inferenceOnBox_WithBlurMotion(canvas, cache.box[i, :], 13),
                                  LibMasking_Blur.inferenceOnBox_WithBlurMotion(canvas, None, 13)], axis=0))
    result.append(np.concatenate([LibMasking_Blur.inferenceOnBox_WithBlurWater(canvas, cache.box[i, :], 4, 4),
                                  LibMasking_Blur.inferenceOnBox_WithBlurWater(canvas, None, 4, 4)], axis=0))
    result.append(np.concatenate([LibMasking_Blur.inferenceOnBox_WithBlurDiffusion(canvas, cache.box[i, :], 17, 3, 3),
                                  LibMasking_Blur.inferenceOnBox_WithBlurDiffusion(canvas, None, 17, 3, 3)], axis=0))
    result.append(np.concatenate([LibMasking_Blur.inferenceOnBox_WithBlurDiffusion(canvas, cache.box[i, :], 17, 0, 0),
                                  LibMasking_Blur.inferenceOnBox_WithBlurDiffusion(canvas, None, 17, 0, 0)], axis=0))
    return np.concatenate(result, axis=1)

def maskingWithMosaic(canvas, cache, i):
    result = []
    result.append(np.concatenate([LibMasking_Mosaic.inferenceBoxWithMosaicSquare(canvas, cache.box[i, :], 24, 'box'),
                                  LibMasking_Mosaic.inferenceBoxWithMosaicSquare(canvas, cache.box[i, :], 24, 'head')], axis=0))
    result.append(np.concatenate([LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 8, True, 'box'),
                                  LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 8, True, 'head')], axis=0))
    result.append(np.concatenate([LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 16, True, 'box'),
                                  LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 16, True, 'head')], axis=0))
    result.append(np.concatenate([LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 8, False, 'box'),
                                  LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 8, False, 'head')], axis=0))
    result.append(np.concatenate([LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 16, False, 'box'),
                                  LibMasking_Mosaic.inferenceBoxWithMosaicPolygon(canvas, cache.box[i, :], 16, False, 'head')], axis=0))
    return np.concatenate(result, axis=1)

def maskingWithSticker(canvas, cache, i):
    # path = '{}/resource/sticker/retro'.format(os.path.split(__file__)[0])
    sticker = cv2.imread('{}/resource/sticker/retro/01.png'.format(os.path.split(__file__)[0]), cv2.IMREAD_UNCHANGED)
    points = np.array([[857, 1735], [1683, 1735]], dtype=np.int32)
    return LibMasking_Sticker.inferenceWithBox(canvas, cache.box[i, :], dict(bgr=sticker, eyes_center=points))

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
    
def benchmark(path_in, path_out):
    # assert os.path.isdir(path_in), path_in
    # assert os.path.isdir(path_out), path_out
    # with XContextTimer(True) as context:
    #     name_list = sorted(os.listdir(path_in))[:1]
    #     with tqdm.tqdm(total=len(name_list)) as bar:
    #         for n, name in enumerate(name_list):
    #             bgr = cv2.imread('{}/{}'.format(path_in, name))
    #             cache = LibMasking_Mosaic.toCache(bgr)
    #             # blur
    #             # canvas_blur = np.copy(cache.bgr)
    #             # for i in range(cache.number):
    #             #     canvas_blur = maskingWithBlur(canvas_blur, cache, i)
    #             # cv2.imwrite('{}/{}-blur.png'.format(path_out, name[:-4]), canvas_blur)
    #             # mosaic
    #             # canvas_mosaic = np.copy(cache.bgr)
    #             # for i in range(cache.number):
    #             #     canvas_mosaic = maskingWithMosaic(canvas_mosaic, cache, i)
    #             # cv2.imwrite('{}/{}-mosaic.png'.format(path_out, name[:-4]), canvas_mosaic)
    #             # sticker
    #             canvas_sticker = np.copy(cache.bgr)
    #             for i in range(cache.number):
    #                 canvas_sticker = maskingWithSticker(canvas_sticker, cache, i)
    #             cv2.imwrite('{}/{}-sticker.png'.format(path_out, name[:-4]), canvas_sticker)
    #             bar.update(1)
    #     logging.warning('blur finish...')

    name = 'obama.png'
    path_in = '{}/document'.format(os.path.split(__file__)[0])
    path_out = '{}/document/obama'.format(os.path.split(__file__)[0])
    bgr = cv2.imread('{}/{}'.format(path_in, name), cv2.IMREAD_COLOR)
    cache = XPortrait(bgr)
    # blur
    # maskingAllWithBlur(cache, path_out, name)
    # mosaic
    maskingAllWithMosaic(cache, path_out, name)
    # sticker
    # maskingAllWithSticker(cache, path_out, name)

    # cache = XBody(bgr)
    # for n in range(cache.number):
    #     cv2.imwrite('{}/{}-parsing-{}.png'.format(path_out, name[:-4], n), cache.visual_portrait_parsing[n, :, :, :])
