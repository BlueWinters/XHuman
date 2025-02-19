
import logging
import os
import cv2
import numpy as np
from skimage import segmentation
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from .masking_mosaic import MaskingMosaic


class MaskingMosaicPolygon(MaskingMosaic):
    """
    """
    NameEN = 'mosaic_polygon'
    NameCN = '多边形马赛克'

    @staticmethod
    def benchmark():
        pass

    """
    """
    def __init__(
            self, n_div=12, visual_boundary=False, segment_reuse=True, segment_mask=None,
            align_type='head', segment_handle='voronoi', *args, **kwargs):
        super(MaskingMosaicPolygon, self).__init__(*args, **kwargs)
        self.n_div = int(n_div)
        self.visual_boundary = bool(visual_boundary)
        self.segment_reuse = bool(segment_reuse)
        self.segment_mask = segment_mask
        self.align_type = align_type
        self.segment_handle = self.getSegmentationHandle(segment_handle)

    def __str__(self):
        return '{}(n_div={}, visual_boundary={}, segment_reuse={}, align_type={})'.format(
            self.NameEN, self.n_div, self.visual_boundary, self.segment_reuse, self.align_type)

    """
    """
    @staticmethod
    def doSuperPixelSegmentation(bgr, n_div=32, **kwargs):
        kernel = 11
        bgr = cv2.GaussianBlur(bgr, (kernel, kernel), sigmaX=kernel // 2, sigmaY=kernel // 2)
        mesh_size = kwargs.pop('mesh_size', 32)
        compactness = kwargs.pop('compactness', 10)
        mask = kwargs['mask'] if 'mask' in kwargs else None
        n_div = n_div if n_div > 0 else int((bgr.shape[0] * bgr.shape[1]) / (mesh_size * mesh_size))
        h, w, c = bgr.shape
        n_segments = n_div * int(n_div * max(h, w) / min(h, w))
        segments = segmentation.slic(bgr, n_segments=n_segments, slic_zero=True, compactness=compactness, start_label=1, mask=mask)
        return segments, n_segments

    @staticmethod
    def generateVoronoiPentagons(h, w, cell_size, seed=None):
        # 随机生成点，作为五边形的中心
        np.random.seed(seed)
        step = cell_size // 2
        # 1.source method
        # points = []
        # for x in range(0, w, cell_size):
        #     for y in range(0, h, cell_size):
        #         jitter_x = np.random.randint(-step, step)*0
        #         jitter_y = np.random.randint(-step, step)*0
        #         points.append([x + step + jitter_x, y + step + jitter_y])
        # 2.fast method
        x = np.arange(0, h, cell_size) + step
        y = np.arange(0, w, cell_size) + step
        xv, yv = np.meshgrid(x, y)
        points = np.stack([np.reshape(xv, (-1,)), np.reshape(yv, (-1,))], axis=1)
        points = points + np.random.randint(-step, step, size=(len(points), 2))
        # 使用 Voronoi 图生成区域
        vor = Voronoi(points)
        polygons = []
        for region_idx in vor.point_region:
            region = vor.regions[region_idx]
            if -1 not in region:  # 忽略无限区域
                polygon = [vor.vertices[i] for i in region]
                polygons.append(Polygon(polygon))
        return polygons

    @staticmethod
    def doSegmentationWithVoronoi(bgr, n_div=32, **kwargs):
        h, w, c = bgr.shape
        cell_size = int((h + w) / n_div / 2)
        polygons = MaskingMosaicPolygon.generateVoronoiPentagons(
            h+2*cell_size+2, w+2*cell_size+2, cell_size)
        canvas = np.zeros((h+2*cell_size+2, w+2*cell_size+2), dtype=np.int32)
        for n, poly in enumerate(polygons):
            points = np.array(poly.exterior.coords, dtype=np.int32)
            cv2.fillPoly(canvas, [points], n + 1)
        mask = canvas[cell_size+1:-cell_size-1, cell_size+1:-cell_size-1]
        return mask, None

    @staticmethod
    def getSegmentationHandle(segment_type='super_pixel'):
        if segment_type == 'voronoi':
            return MaskingMosaicPolygon.doSegmentationWithVoronoi
        if segment_type == 'super_pixel':
            return MaskingMosaicPolygon.doSuperPixelSegmentation
        raise NotImplementedError(segment_type)

    @staticmethod
    def visualAsMean(label_field, bgr, bg_label=0, bg_color=(255, 255, 255)):
        out = np.zeros(label_field.shape + (3,), dtype=bgr.dtype)
        labels = np.unique(label_field)
        if (labels == bg_label).any():
            labels = labels[labels != bg_label]
            mask = (label_field == bg_label).nonzero()
            out[mask] = bg_color
        for label in labels:
            mask = (label_field == label).nonzero()
            color = bgr[mask].mean(axis=0).astype(np.uint8)
            out[mask] = color
        return out

    @staticmethod
    def doPostprocess(bgr, seg, vis_boundary):
        bgr_copy = np.copy(bgr)
        bgr_copy = MaskingMosaicPolygon.visualAsMean(seg, bgr_copy, 0)
        if vis_boundary is True:
            # bgr_copy = segmentation.mark_boundaries(bgr_copy, seg, color=(1, 1, 1), mode='outer')
            # bgr_copy = np.round(bgr_copy * 255).astype(np.uint8)
            boundary = segmentation.find_boundaries(seg, mode='thick').astype(np.uint8) * 255
            boundary = cv2.resize(boundary, bgr.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            multi = boundary.astype(np.float32)[:, :, None] / 255.
            fusion = bgr_copy.astype(np.float32) * (1-multi) + np.ones_like(bgr_copy) * 255 * multi
            bgr_copy = np.round(fusion).astype(np.uint8)
        return bgr_copy

    def getSegmentationMask(self, bgr, **kwargs):
        if self.segment_reuse is True:
            if isinstance(self.segment_mask,  np.ndarray) is False:
                self.segment_mask = self.segment_handle(bgr, self.n_div, **kwargs)[0]
            return self.segment_mask
        return self.segment_handle(bgr, self.n_div, **kwargs)[0]

    def inference(self, bgr, **kwargs):
        segment_mask = self.getSegmentationMask(bgr, **kwargs)
        return MaskingMosaicPolygon.doPostprocess(bgr, segment_mask, self.visual_boundary)

