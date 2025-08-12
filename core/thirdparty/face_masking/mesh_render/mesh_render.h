#ifndef __Mesh_Render__
#define __Mesh_Render__


void render_rasterize(
    const float* pos, int N,
    const int* tri, int M,
    const float* proj,
    int h, int w,
    float* output
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

#endif