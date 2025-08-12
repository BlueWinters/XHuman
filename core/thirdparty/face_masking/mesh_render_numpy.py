
import logging
import cv2
import numpy as np
import numba


class MeshRendererNumpy:
    def __init__(self, rasterize_fov, znear=0.1, zfar=10, rasterize_size=224):
        super(MeshRendererNumpy, self).__init__()
        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        self.ndc_proj = self.ndc_projection(x=x, n=znear, f=zfar)
        self.rast_h = rasterize_size
        self.rast_w = rasterize_size

    @staticmethod
    def ndc_projection(x=0.1, n=1.0, f=50.0):
        array = np.array([
            [n / x, 0, 0, 0],
            [0, n / -x, 0, 0],
            [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
            [0, 0, -1, 0]
        ]).astype(np.float32)
        diagonal = np.diag(np.array([1., -1, -1, 1], dtype=np.float32))
        projection = np.matmul(array, diagonal).T
        return np.ascontiguousarray(projection)

    @staticmethod
    def compute_barycentric(p_x, p_y, x0, y0, x1, y1, x2, y2):
        """
        计算点 (p_x, p_y) 相对于三角形 (v0, v1, v2) 的重心坐标 (w0, w1, w2)
        """
        # 向量
        v0 = np.array([x1 - x0, y1 - y0])  # edge v0->v1
        v1 = np.array([x2 - x0, y2 - y0])  # edge v0->v2
        v2 = np.array([p_x - x0, p_y - y0])  # point - v0
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            return -1, -1, -1  # 退化三角形
        w1 = (d11 * d20 - d01 * d21) / denom  # v1 权重
        w2 = (d00 * d21 - d01 * d20) / denom  # v2 权重
        w0 = 1.0 - w1 - w2  # v0 权重
        return w0, w1, w2

    @staticmethod
    def rasterize(pos, tri, proj, h, w):
        output = np.zeros((h, w, 4), dtype=np.float64)
        # 修复：初始化为 +inf，因为我们希望保留 z/w 更小的（更近）
        z_buffer = np.full((h, w), np.inf, dtype=np.float64)
        eps = 1e-5

        # world -> clip space
        pos_homo = np.hstack([pos, np.ones((pos.shape[0], 1))])  # (N, 4)
        # pos_clip = (proj @ pos_homo.T).T  # (N, 4)
        pos_clip = pos_homo @ proj
        # clip -> NDC
        weight = pos_clip[:, 3:4]
        pos_ndc = pos_clip[:, :3] / weight  # (N, 3): x/w, y/w, z/w
        for t_idx in range(tri.shape[0]):
            i0, i1, i2 = tri[t_idx]
            x0, y0, z0_over_w = pos_ndc[i0]
            x1, y1, z1_over_w = pos_ndc[i1]
            x2, y2, z2_over_w = pos_ndc[i2]
            # 跳过完全在视锥外的三角形
            if max(z0_over_w, z1_over_w, z2_over_w) < -1.0 or min(z0_over_w, z1_over_w, z2_over_w) > 1.0:
                continue

            # NDC -> 像素坐标
            def ndc_to_pixel(v):
                return (v + 1.0) * 0.5 * w

            px0 = ndc_to_pixel(x0)
            px1 = ndc_to_pixel(x1)
            px2 = ndc_to_pixel(x2)
            py0 = ndc_to_pixel(y0)
            py1 = ndc_to_pixel(y1)
            py2 = ndc_to_pixel(y2)
            min_x = max(0, int(np.floor(min(px0, px1, px2))) - 1)
            max_x = min(w - 1, int(np.ceil(max(px0, px1, px2))) + 1)
            min_y = max(0, int(np.floor(min(py0, py1, py2))) - 1)
            max_y = min(h - 1, int(np.ceil(max(py0, py1, py2))) + 1)
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    ndc_x = (x + 0.5) / w * 2.0 - 1.0
                    ndc_y = (y + 0.5) / h * 2.0 - 1.0
                    # 包围盒判断
                    min_xx = min(x0, x1, x2)
                    max_xx = max(x0, x1, x2)
                    min_yy = min(y0, y1, y2)
                    max_yy = max(y0, y1, y2)
                    if not (min_xx <= ndc_x <= max_xx and min_yy <= ndc_y <= max_yy):
                        continue  # 直接跳过
                    w0, w1, w2 = MeshRendererNumpy.compute_barycentric(
                        ndc_x * 100, ndc_y * 100, x0 * 100, y0 * 100, x1 * 100, y1 * 100, x2 * 100, y2 * 100)
                    if w0 >= -eps and w1 >= -eps and w2 >= -eps:
                        w0_clamp = max(w0, 0.0)
                        w1_clamp = max(w1, 0.0)
                        z_over_w = w0 * z0_over_w + w1 * z1_over_w + w2 * z2_over_w
                        # 修复：z/w 越小表示越近！
                        if z_over_w < z_buffer[y, x]:
                            z_buffer[y, x] = z_over_w
                            output[y, x, 0] = w0_clamp
                            output[y, x, 1] = w1_clamp
                            output[y, x, 2] = z_over_w
                            output[y, x, 3] = t_idx + 1
        return output

    @staticmethod
    @numba.jit(nopython=True)
    def interpolate(attr, rast, tri):
        """
        插值顶点属性，参照nvdiffrast.torch.interpolate。
        参数:
            attr (np.ndarray): (N, num_attributes), 顶点属性
            rast (np.ndarray): (H, W, 4), rast 输出 (u, v, z/w, triangle_id)
            tri (np.ndarray): (M, 3), 三角形索引
        返回:
            output (np.ndarray): (H, W, num_attributes), 插值后的属性
        """
        h, w = rast.shape[:2]
        num_attributes = attr.shape[1]
        output = np.zeros((h, w, num_attributes), dtype=np.float32)

        # 遍历每个像素
        for y in range(h):
            for x in range(w):
                r = rast[y, x]
                u, v, _, triangle_id = r  # u=w0, v=w1
                t_idx = int(triangle_id - 1)  # 还原三角形索引
                # 如果未被覆盖
                if triangle_id == 0:
                    continue  # output 保持 0
                # 检查 t_idx 是否越界
                if t_idx < 0 or t_idx >= tri.shape[0]:
                    continue
                # 获取三角形三个顶点索引
                i0, i1, i2 = tri[t_idx]
                # 获取重心坐标
                w0 = u
                w1 = v
                w2 = 1.0 - w0 - w1
                # 边界保护：防止数值误差导致负权重
                if w0 < 0 or w1 < 0 or w2 < 0:
                    # 可选：clamp 或跳过
                    w0 = max(w0, 0)
                    w1 = max(w1, 0)
                    w2 = max(1.0 - w0 - w1, 0)
                # 插值属性
                # 使用 np.clip 防止索引越界
                if i0 < attr.shape[0] and i1 < attr.shape[0] and i2 < attr.shape[0]:
                    attr0 = attr[i0]  # (num_attributes,)
                    attr1 = attr[i1]
                    attr2 = attr[i2]
                    interp_attr = w0 * attr0 + w1 * attr1 + w2 * attr2
                    output[y, x] = interp_attr

        return output

    @staticmethod
    @numba.jit(nopython=True)
    def texture(texture, uv):
        """
        纹理采样函数：对texture进行bilinear采样，基于 uv 坐标
        参数:
            texture: (tex_h, tex_w, C), 纹理图
            uv: (uv_h, uv_w, 2), UV 坐标，范围 [0, 1]
        返回:
            output: (uv_h, uv_w, C), 采样结果
        """
        tex_h, tex_w, tex_c = texture.shape
        uv_h, uv_w, _ = uv.shape
        output = np.zeros((uv_h, uv_w, tex_c), dtype=np.float32)

        # 遍历每个 UV 坐标
        for i in range(uv_h):
            for j in range(uv_w):
                u, v = uv[i, j]
                # 检查是否在 [0,1] 范围内
                if u < 0 or u > 1 or v < 0 or v > 1:
                    continue  # 输出为 0
                # 映射到纹理坐标
                u_scaled = u * (tex_w - 1)
                v_scaled = v * (tex_h - 1)
                # 找到四个最近的像素
                x0 = int(np.floor(u_scaled))
                y0 = int(np.floor(v_scaled))
                x1 = min(x0 + 1, tex_w - 1)
                y1 = min(y0 + 1, tex_h - 1)
                # 双线性插值权重
                wx = u_scaled - x0
                wy = v_scaled - y0
                # 获取四个角的像素值
                tl = texture[y0, x0]  # top-left
                tr = texture[y0, x1]  # top-right
                bl = texture[y1, x0]  # bottom-left
                br = texture[y1, x1]  # bottom-right
                # 双线性插值
                top = (1 - wx) * tl + wx * tr
                bot = (1 - wx) * bl + wx * br
                interp = (1 - wy) * top + wy * bot
                output[i, j] = interp
        return output

    def render_uv_texture(self, vertex, tri, uv, uv_texture):
        """
        渲染三角网格的纹理图像。
        参数:
            vertex          -- (N, 3)，人脸顶点坐标
            tri             -- (M, 3)，三角形索引
            uv              -- (N, 2)，每个顶点的UV坐标
            uv_texture      -- (H, W, C)，纹理图像
        返回:
            image           -- (H, W, 3)，渲染后的图片
            mask            -- (H, W)，人脸区域掩码
            depth           -- (H, W)，深度图
        """
        # 顶点预处理：y轴取反，齐次坐标
        vertex = np.reshape(vertex.astype(np.float32), (-1, 3))
        vertex[:, 1] = -vertex[:, 1]
        tri = np.reshape(tri.astype(np.int32), (-1, 3))
        uv = np.reshape(uv.astype(np.float32), (-1, 2))
        # V方向反向，适配纹理坐标系
        uv[..., -1] = 1.0 - uv[..., -1]
        # 光栅化：输出每个像素的重心坐标、深度、三角形id
        rast_out = MeshRendererNumpy.rasterize(vertex, tri, self.ndc_proj, self.rast_h, self.rast_w)
        # 插值UV坐标
        interp_out = MeshRendererNumpy.interpolate(uv, rast_out, tri)
        # 纹理采样，得到渲染图片
        image = MeshRendererNumpy.texture(uv_texture.astype(np.float32), interp_out)
        # 生成mask：被三角形覆盖的像素为1，否则为0
        mask = np.where(rast_out[..., 3] > 0, 1, 0).astype(np.uint8)
        # 插值深度
        depth = MeshRendererNumpy.interpolate(vertex[:, 2:3], rast_out, tri)[:, :, 0]
        # 返回渲染结果
        return np.round(image).astype(np.uint8), (mask * 255), depth

    def render_feature_texture(self, vertex, tri, feat):
        # 顶点预处理：y轴取反，齐次坐标
        vertex = np.reshape(vertex.astype(np.float32), (-1, 3))
        vertex[:, 1] = -vertex[:, 1]
        tri = np.reshape(tri.astype(np.int32), (-1, 3))
        feat = np.reshape(feat.astype(np.float32), (-1, 3))
        # 光栅化：输出每个像素的重心坐标、深度、三角形id
        rast_out = MeshRendererNumpy.rasterize(vertex, tri, self.ndc_proj, self.rast_h, self.rast_w)
        # 对feat进行插值
        image = MeshRendererNumpy.interpolate(feat, rast_out, tri)
        # 生成mask：被三角形覆盖的像素为1，否则为0
        mask = np.where(rast_out[..., 3] > 0, 255, 0).astype(np.uint8)
        # 插值深度
        depth = MeshRendererNumpy.interpolate(vertex[:, 2:3], rast_out, tri)[:, :, 0]
        return mask, depth, np.round(image * 255).astype(np.uint8)

    def render_uv_texture_cython(self, vertex, tri, uv, uv_texture):
        """
        渲染三角网格的纹理图像。
        参数:
            vertex          -- (N, 3)，人脸顶点坐标
            tri             -- (M, 3)，三角形索引
            uv              -- (N, 2)，每个顶点的UV坐标
            uv_texture      -- (H, W, C)，纹理图像
        返回:
            image           -- (H, W, 3)，渲染后的图片
            mask            -- (H, W)，人脸区域掩码
            depth           -- (H, W)，深度图
        """
        from . import mesh_render_cython
        # 顶点预处理：y轴取反，齐次坐标
        vertex = np.reshape(vertex.astype(np.float32), (-1, 3))
        vertex[:, 1] = -vertex[:, 1]
        tri = np.reshape(tri.astype(np.int32), (-1, 3))
        uv = np.reshape(uv.astype(np.float32), (-1, 2))
        # V方向反向，适配纹理坐标系
        uv[..., -1] = 1.0 - uv[..., -1]
        # 光栅化：输出每个像素的重心坐标、深度、三角形id
        rast_out = mesh_render_cython.rasterize(vertex, tri, self.ndc_proj, self.rast_h, self.rast_w)
        # 插值UV坐标
        interp_out = mesh_render_cython.interpolate(uv, rast_out, tri)
        # 纹理采样，得到渲染图片
        image = mesh_render_cython.texture(uv_texture.astype(np.float32), interp_out)
        # 生成mask：被三角形覆盖的像素为1，否则为0
        mask = np.where(rast_out[..., 3] > 0, 1, 0).astype(np.uint8)
        # 插值深度
        depth = mesh_render_cython.interpolate(vertex[:, 2:3], rast_out, tri)[:, :, 0]
        # 返回渲染结果
        return np.round(image).astype(np.uint8), (mask * 255), depth
