# distutils: language = c++
# cython: language_level=3

import numpy as np
import cv2
from openmp cimport omp_get_thread_num, omp_get_num_threads
cimport numpy as np
cimport openmp


np.import_array()  # initialize numpy C-API

cdef extern from "mesh_render.h":
    void render_rasterize(
        const float* pos, int N,
        const int* tri, int M,
        const float* proj,
        int h, int w,
        float* rast_out
    );

    void render_interpolate(
        const float* attr, int N, int num_attr,
        const float* rast, int h, int w,
        const int* tri, int M,
        float* output
    );

    void render_texture(
        const float* texture, int tex_h, int tex_w, int tex_c,
        const float* uv, int uv_h, int uv_w,
        float* output
    );


"""
"""
def testOpenMP():
    cdef int num_threads = 0
    num_threads = omp_get_num_threads()
    return num_threads > 0


"""
"""
def rasterize(
    np.ndarray[np.float32_t, ndim=2] pos,
    np.ndarray[np.int32_t, ndim=2] tri,
    np.ndarray[np.float32_t, ndim=2] proj,
    int h, int w):
    cdef int N = pos.shape[0]
    cdef int M = tri.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] pos_c = np.ascontiguousarray(pos)
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] tri_c = np.ascontiguousarray(tri)
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] proj_c = np.ascontiguousarray(proj)
    cdef np.ndarray[np.float32_t, ndim=3] output = np.zeros((h, w, 4), dtype=np.float32)
    render_rasterize(
        <const float*>pos_c.data, N,
        <const int*>tri_c.data, M,
        <const float*>proj_c.data,
        h, w,
        <float*>output.data)
    return output

def interpolate(
    np.ndarray[np.float32_t, ndim=2] attr,
    np.ndarray[np.float32_t, ndim=3] rast,
    np.ndarray[np.int32_t, ndim=2] tri):
    h, w = rast.shape[:2]
    N = attr.shape[0]
    num_attr = attr.shape[1]
    M = tri.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] attr_c = np.ascontiguousarray(attr)
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] rast_c = np.ascontiguousarray(rast)
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] tri_c = np.ascontiguousarray(tri)
    cdef np.ndarray[np.float32_t, ndim=3] output = np.zeros((h, w, num_attr), dtype=np.float32)
    render_interpolate(
        <const float*>attr_c.data, N, num_attr,
        <const float*>rast_c.data, h, w,
        <const int*>tri_c.data, M,
        <float*>output.data)
    return output

def texture(
    np.ndarray[np.float32_t, ndim=3] texture,
    np.ndarray[np.float32_t, ndim=3] uv):
    tex_h, tex_w, tex_c = texture.shape[:3]
    uv_h, uv_w = uv.shape[:2]
    cdef np.ndarray[np.float32_t, ndim=3] texture_c = np.ascontiguousarray(texture)
    cdef np.ndarray[np.float32_t, ndim=3] uv_c = np.ascontiguousarray(uv)
    cdef np.ndarray[np.float32_t, ndim=3] output = np.zeros((uv_h, uv_w, tex_c), dtype=np.float32)
    render_texture(
        <const float*>texture_c.data, tex_h, tex_w, tex_c,
        <const float*>uv_c.data, uv_h, uv_w,
        <float*>output.data)
    return output