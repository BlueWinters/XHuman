
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <limits>
#include <vector>
#include "mesh_render.h"


// calculate barycentric coordinates
inline void computeBaryCentric(
    float px, float py,
    float x0, float y0, float x1, float y1, float x2, float y2,
    float* w0, float* w1, float* w2)
{
    float v0x = x1 - x0, v0y = y1 - y0;
    float v1x = x2 - x0, v1y = y2 - y0;
    float v2x = px - x0, v2y = py - y0;
    float d00 = v0x * v0x + v0y * v0y;
    float d01 = v0x * v1x + v0y * v1y;
    float d11 = v1x * v1x + v1y * v1y;
    float d20 = v2x * v0x + v2y * v0y;
    float d21 = v2x * v1x + v2y * v1y;
    float denom = d00 * d11 - d01 * d01;
    if (fabs(denom) < 1e-12f) {
        *w0 = *w1 = *w2 = -1.0f;
        return;
    }
    *w1 = (d11 * d20 - d01 * d21) / denom;
    *w2 = (d00 * d21 - d01 * d20) / denom;
    *w0 = 1.0f - *w1 - *w2;
}

void render_rasterize(
    const float* pos, int N,
    const int* tri, int M,
    const float* proj, // 4x4
    int h, int w,
    float* output // h*w*4
)
{
    // z-buffer
    float* z_buffer = new float[h * w];
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < h * w; ++i) {
        z_buffer[i] = std::numeric_limits<float>::infinity();
    }

    float* pos_ndc = new float[N * 3];
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < N; ++i) {
        // world -> clip space
        float x = pos[i * 3 + 0];
        float y = pos[i * 3 + 1];
        float z = pos[i * 3 + 2];
        float clip0 = x * proj[0 * 4 + 0] + y * proj[1 * 4 + 0] + z * proj[2 * 4 + 0] + proj[3 * 4 + 0];
        float clip1 = x * proj[0 * 4 + 1] + y * proj[1 * 4 + 1] + z * proj[2 * 4 + 1] + proj[3 * 4 + 1];
        float clip2 = x * proj[0 * 4 + 2] + y * proj[1 * 4 + 2] + z * proj[2 * 4 + 2] + proj[3 * 4 + 2];
        float clip3 = x * proj[0 * 4 + 3] + y * proj[1 * 4 + 3] + z * proj[2 * 4 + 3] + proj[3 * 4 + 3];
        // clip space -> NDC
        float wv = clip3;
        pos_ndc[i * 3 + 0] = clip0 / wv;
        pos_ndc[i * 3 + 1] = clip1 / wv;
        pos_ndc[i * 3 + 2] = clip2 / wv;
    }

    float eps = 1e-5f;
    #pragma omp parallel for num_threads(2)
    for (int t_idx = 0; t_idx < M; ++t_idx) {
        int i0 = tri[t_idx * 3 + 0];
        int i1 = tri[t_idx * 3 + 1];
        int i2 = tri[t_idx * 3 + 2];
        float x0 = pos_ndc[i0 * 3 + 0], y0 = pos_ndc[i0 * 3 + 1], z0 = pos_ndc[i0 * 3 + 2];
        float x1 = pos_ndc[i1 * 3 + 0], y1 = pos_ndc[i1 * 3 + 1], z1 = pos_ndc[i1 * 3 + 2];
        float x2 = pos_ndc[i2 * 3 + 0], y2 = pos_ndc[i2 * 3 + 1], z2 = pos_ndc[i2 * 3 + 2];

        // skip triangles completely outside the view frustum
        if (std::max(std::max(z0, z1), z2) < -1.0f || std::min(std::min(z0, z1), z2) > 1.0f)
            continue;

        // NDC -> pixel coordinates
        auto ndc_to_pixel = [&](float v, int size) {
            return (v + 1.0f) * 0.5f * (size - 1);
        };
        float px0 = ndc_to_pixel(x0, w);
        float px1 = ndc_to_pixel(x1, w);
        float px2 = ndc_to_pixel(x2, w);
        float py0 = ndc_to_pixel(y0, h);
        float py1 = ndc_to_pixel(y1, h);
        float py2 = ndc_to_pixel(y2, h);

        // world/clip space triangle vertices
        float vx0 = pos[i0 * 3 + 0], vy0 = pos[i0 * 3 + 1], vz0 = pos[i0 * 3 + 2];
        float vx1 = pos[i1 * 3 + 0], vy1 = pos[i1 * 3 + 1], vz1 = pos[i1 * 3 + 2];
        float vx2 = pos[i2 * 3 + 0], vy2 = pos[i2 * 3 + 1], vz2 = pos[i2 * 3 + 2];

        // compute triangle normal
        float nx = (vy1 - vy0) * (vz2 - vz0) - (vz1 - vz0) * (vy2 - vy0);
        float ny = (vz1 - vz0) * (vx2 - vx0) - (vx1 - vx0) * (vz2 - vz0);
        float nz = (vx1 - vx0) * (vy2 - vy0) - (vy1 - vy0) * (vx2 - vx0);

        // view direction (assuming the viewer is at the origin)
        float view_dir_x = vx0;
        float view_dir_y = vy0;
        float view_dir_z = vz0;

        // dot product test
        float dot = nx * view_dir_x + ny * view_dir_y + nz * view_dir_z;
        // backface culling
        if (dot >= 0.0f) continue;

        int min_x = std::max(0, static_cast<int>(std::floor(std::min(std::min(px0, px1), px2)) - 1));
        int max_x = std::min(w - 1, static_cast<int>(std::ceil(std::max(std::max(px0, px1), px2)) + 1));
        int min_y = std::max(0, static_cast<int>(std::floor(std::min(std::min(py0, py1), py2)) - 1));
        int max_y = std::min(h - 1, static_cast<int>(std::ceil(std::max(std::max(py0, py1), py2)) + 1));

        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                // map pixel center back to NDC
                float ndc_x = (x + 0.5f) / static_cast<float>(w) * 2.0f - 1.0f;
                float ndc_y = (y + 0.5f) / static_cast<float>(h) * 2.0f - 1.0f;
                // bounding box test
                float min_xx = std::min(std::min(x0, x1), x2);
                float max_xx = std::max(std::max(x0, x1), x2);
                float min_yy = std::min(std::min(y0, y1), y2);
                float max_yy = std::max(std::max(y0, y1), y2);
                if (!(min_xx <= ndc_x && ndc_x <= max_xx && min_yy <= ndc_y && ndc_y <= max_yy))
                    continue;
                // compute barycentric coordinates
                float w0, w1, w2;
                computeBaryCentric(
                    ndc_x, ndc_y, x0, y0, x1, y1, x2, y2,
                    &w0, &w1, &w2);
                if (w0 >= -eps && w1 >= -eps && w2 >= -eps) {
                    float w0_clamp = std::max(w0, 0.0f);
                    float w1_clamp = std::max(w1, 0.0f);
                    float z_over_w = w0 * z0 + w1 * z1 + w2 * z2;
                    int idx = y * w + x;
                    #pragma omp critical
                    {
                        if (z_over_w < z_buffer[idx]) {
                            z_buffer[idx] = z_over_w;
                            output[idx * 4 + 0] = w0_clamp;
                            output[idx * 4 + 1] = w1_clamp;
                            output[idx * 4 + 2] = z_over_w;
                            output[idx * 4 + 3] = static_cast<float>(t_idx + 1);
                        }
                    }
                }
            }
        }
    }

    delete[] z_buffer;
    delete[] pos_ndc;
}

void render_interpolate(
    const float* attr, int N, int num_attr,
    const float* rast, int h, int w,
    const int* tri, int M,
    float* output // h * w * num_attr
)
{
    int total_pixels = h * w;
    #pragma omp parallel for num_threads(2)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            float u = rast[idx * 4 + 0];
            float v = rast[idx * 4 + 1];
            float triangle_id = rast[idx * 4 + 3];
            int t_idx = static_cast<int>(triangle_id - 1);
            if (triangle_id == 0 || t_idx < 0 || t_idx >= M)
                continue;
            int i0 = tri[t_idx * 3 + 0];
            int i1 = tri[t_idx * 3 + 1];
            int i2 = tri[t_idx * 3 + 2];
            float w0 = u;
            float w1 = v;
            float w2 = 1.0f - w0 - w1;
            if (w0 < 0 || w1 < 0 || w2 < 0) {
                w0 = std::max(w0, 0.0f);
                w1 = std::max(w1, 0.0f);
                w2 = std::max(1.0f - w0 - w1, 0.0f);
            }
            if (i0 < N && i1 < N && i2 < N) {
                for (int k = 0; k < num_attr; ++k) {
                    float attr0 = attr[i0 * num_attr + k];
                    float attr1 = attr[i1 * num_attr + k];
                    float attr2 = attr[i2 * num_attr + k];
                    output[idx * num_attr + k] = w0 * attr0 + w1 * attr1 + w2 * attr2;
                }
            }
        }
    }
}

void render_texture(
    const float* texture, int tex_h, int tex_w, int tex_c,
    const float* uv, int uv_h, int uv_w,
    float* output // uv_h * uv_w * tex_c
)
{
    int total_pixels = uv_h * uv_w;
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < uv_h; ++i) {
        for (int j = 0; j < uv_w; ++j) {
            float u = uv[(i * uv_w + j) * 2 + 0];
            float v = uv[(i * uv_w + j) * 2 + 1];
            if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f)
                continue;
            float u_scaled = u * (tex_w - 1);
            float v_scaled = v * (tex_h - 1);
            int x0 = static_cast<int>(floor(u_scaled));
            int y0 = static_cast<int>(floor(v_scaled));
            int x1 = std::min(x0 + 1, tex_w - 1);
            int y1 = std::min(y0 + 1, tex_h - 1);
            float wx = u_scaled - x0;
            float wy = v_scaled - y0;
            for (int c = 0; c < tex_c; ++c) {
                float tl = texture[(y0 * tex_w + x0) * tex_c + c];
                float tr = texture[(y0 * tex_w + x1) * tex_c + c];
                float bl = texture[(y1 * tex_w + x0) * tex_c + c];
                float br = texture[(y1 * tex_w + x1) * tex_c + c];
                float top = (1.0f - wx) * tl + wx * tr;
                float bot = (1.0f - wx) * bl + wx * br;
                float interp = (1.0f - wy) * top + wy * bot;
                output[(i * uv_w + j) * tex_c + c] = interp;
            }
        }
    }
}