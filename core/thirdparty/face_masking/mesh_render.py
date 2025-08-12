
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr


class MeshRenderer(nn.Module):
    def  __init__(self, rasterize_fov, znear=0.1, zfar=10, rasterize_size=224):
        super(MeshRenderer, self).__init__()
        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        self.ndc_proj = torch.tensor(self.ndc_projection(x=x, n=znear, f=zfar)).matmul(
            torch.diag(torch.tensor([1., -1, -1, 1])))
        self.rasterize_size = rasterize_size
        self.ctx = None

    @staticmethod
    def ndc_projection(x=0.1, n=1.0, f=50.0):
        return np.array([
            [n / x, 0, 0, 0],
            [0, n / -x, 0, 0],
            [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
            [0, 0, -1, 0]
        ]).astype(np.float32)

    def render_uv_texture(self, vertex, tri, uv, uv_texture, reverse_uv=False):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None
            uv(optional)       -- torch.tensor, size (B, C, H, W) if reverse_uv is not None
        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (M, 3), triangles
            uv              -- torch.tensor, size (B, N, 2),  uv mapping
            uv_texture      -- torch.tensor, size (B, C, H, W),  texture map
        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1]

        vertex_ndc = vertex @ ndc_proj.t()
        if self.ctx is None:
            self.ctx = dr.RasterizeGLContext(device=device)

        # vertex_ndc: Size([1, 35709, 4])
        # Size([70789, 3])
        ranges = None
        if isinstance(tri, list) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device)
            fstartidx = torch.cumsum(fnum, dim=0) - fnum
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i * vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vertex: [B*N, 4], tri: [B*M, 3], for instance_mode vertex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.ctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize], ranges=ranges)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)

        vertex_z = vertex.reshape([-1, 4])[..., 2].unsqueeze(1).contiguous()
        interp_vert_z, _ = dr.interpolate(vertex_z, rast_out, tri)
        interp_vert_z = interp_vert_z.permute(0, 3, 1, 2)  # B,H,W,C --> B,C,H,W
        depth = mask * interp_vert_z

        uv[..., -1] = 1.0 - uv[..., -1]
        interp_uv, _ = dr.interpolate(uv, rast_out, tri)
        uv_texture = uv_texture.permute(0, 2, 3, 1).contiguous()  # B,C,H,W --> B,H,W,C
        sample_tex = dr.texture(uv_texture, interp_uv, filter_mode='linear')
        sample_tex = sample_tex * torch.clamp(rast_out[..., -1:], 0, 1)  # mask out background.
        image = sample_tex.permute(0, 3, 1, 2)  # B,H,W,C --> B,C,H,W
        return (mask, depth, image) if reverse_uv is False \
            else (mask, depth, image, interp_uv.permute(0, 3, 1, 2))

    def render_feature_texture(self, vertex, tri, feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None
        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, C), features
        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1]

        vertex_ndc = vertex @ ndc_proj.t()
        if self.ctx is None:
            self.ctx = dr.RasterizeCudaContext(device=device)

        ranges = None
        if isinstance(tri, list) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device)
            fstartidx = torch.cumsum(fnum, dim=0) - fnum
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i * vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vertex: [B*N, 4], tri: [B*M, 3], for instance_mode vertex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.ctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize], ranges=ranges)

        depth, _ = dr.interpolate(vertex.reshape([-1, 4])[..., 2].unsqueeze(1).contiguous(), rast_out, tri)
        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth

        image = None
        if feat is not None:
            image, _ = dr.interpolate(feat, rast_out, tri)
            image = image.permute(0, 3, 1, 2)
            image = mask * image
        return mask, depth, image

