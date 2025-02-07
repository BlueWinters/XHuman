
import numpy as np
import skimage


def getMaxRegion(mask):
    label_image = skimage.measure.label(mask.astype(np.uint8), connectivity=2, return_num=False)
    regions = skimage.measure.regionprops(label_image)
    if not regions:
        return np.zeros_like(mask, dtype=bool)
    max_region = max(regions, key=lambda r: r.area)
    max_mask = (label_image == max_region.label).astype(np.uint8)
    return max_mask
