import torch
import torch.nn as nn
import os
import imageio
import time
from .run_nerf_helpers import *
import numpy as np

def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        # nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)



class DSKnet(nn.Module):
    def __init__(self, num_img, H=None, W=None, poses=None, K=None, num_pt=5, kernel_hwindow=10, random_hwindow=0.25,
                 in_embed=3, random_mode='input', img_embed=32, spatial_embed=3,
                 num_hidden=3, num_wide=64, short_cut=False, pattern_init_radius=0.1,
                 isglobal=True, optim_trans=False, optim_spatialvariant_trans=False):
        """
        num_img: number of image, used for deciding the view embedding
        poses: the original poses, used for generating new rays, len(poses) == num_img
        num_pt: number of sparse point, we use 5 in the paper
        kernel_hwindow: the size of physically equivalent blur kernel, the sparse points are bounded inside the blur kernel. 
                        Can be a very big number
        
        random_hwindow: in training, we randomly perturb the sparse point to model a smooth manifold
        random_mode: 'input' or 'output', it controls whether the random perturb is added to the input of DSK or output of DSK
        // the above two parameters do not have big impact on the results

        in_embed: embedding for the canonical kernel location
        img_embed: the length of the view embedding
        spatial_embed: embedding for the pixel location of the blur kernel inside an image
        
        num_hidden, num_wide, short_cut: control the structure of the MLP
        pattern_init_radius: the little gain add to the deform location described in Sec. 4.4
        isglobal: control whether the canonical kernel should be shared by all the input views or not, does not have big impact on the results
        optim_trans: whether to optimize the ray origin described in Sec. 4.3
        optim_spatialvariant_trans: whether to optimize the ray origin for each view or each kernel point. 
        """
        super().__init__()
        self.num_pt = num_pt
        self.num_img = num_img
        self.short_cut = short_cut
        self.kernel_hwindow = kernel_hwindow
        self.random_hwindow = random_hwindow  # about 1 pix
        self.random_mode = random_mode
        self.isglobal = isglobal
        self.K = K
        self.H = H
        self.W = W
        pattern_num = 1 if isglobal else num_img
        assert self.random_mode in ['input', 'output'], f"DSKNet::random_mode {self.random_mode} unrecognized, " \
                                                        f"should be input/output"
        self.register_buffer("poses", poses)
        self.register_parameter("pattern_pos",
                                nn.Parameter(torch.randn(pattern_num, num_pt, 2)
                                             .type(torch.float32) * pattern_init_radius, True))
        self.optim_trans = optim_trans
        self.optim_sv_trans = optim_spatialvariant_trans

        if optim_trans:
            self.register_parameter("pattern_trans",
                                    nn.Parameter(torch.zeros(pattern_num, num_pt, 2)
                                                 .type(torch.float32), True))

        if in_embed > 0:
            self.in_embed_fn, self.in_embed_cnl = get_embedder(in_embed, input_dim=2)
        else:
            self.in_embed_fn, self.in_embed_cnl = None, 0

        self.img_embed_cnl = img_embed

        if spatial_embed > 0:
            self.spatial_embed_fn, self.spatial_embed_cnl = get_embedder(spatial_embed, input_dim=2)
        else:
            self.spatial_embed_fn, self.spatial_embed_cnl = None, 0

        in_cnl = self.in_embed_cnl + self.img_embed_cnl + self.spatial_embed_cnl
        out_cnl = 1 + 2 + 2 if self.optim_sv_trans else 1 + 2  # u, v, w or u, v, w, dx, dy
        hiddens = [nn.Linear(num_wide, num_wide) if i % 2 == 0 else nn.ReLU()
                   for i in range((num_hidden - 1) * 2)]
        # hiddens = [nn.Linear(num_wide, num_wide), nn.ReLU()] * num_hidden
        self.linears = nn.Sequential(
            nn.Linear(in_cnl, num_wide), nn.ReLU(),
            *hiddens,
        )
        self.linears1 = nn.Sequential(
            nn.Linear((num_wide + in_cnl) if short_cut else num_wide, num_wide), nn.ReLU(),
            nn.Linear(num_wide, out_cnl)
        )
        self.linears.apply(init_linear_weights)
        self.linears1.apply(init_linear_weights)
        if img_embed > 0:
            self.register_parameter("img_embed",
                                    nn.Parameter(torch.zeros(num_img, img_embed).type(torch.float32), True))
        else:
            self.img_embed = None

    def forward(self, uv, img_idx):
        """
        inputs: all input has shape (ray_num, cnl), ??????????????????
        outputs: output shape (ray_num, ptnum, 3, 2)  last two dim: [ray_o, ray_d]
        """
        # img_idx = rays_info['images_idx'].squeeze(-1)

        point_num = uv.size(0)
        # ???????????????????????? [point_num, 32]
        img_embed = self.img_embed[img_idx].expand(point_num, -1) if self.img_embed is not None else \
            torch.tensor([]).reshape(point_num, self.img_embed_cnl)

        # ??????5????????????????????????????????? [point_num, 5, 2]
        pt_pos = self.pattern_pos.expand(point_num, -1, -1) if self.isglobal \
            else self.pattern_pos[img_idx]
        pt_pos = torch.tanh(pt_pos) * self.kernel_hwindow

        if self.random_hwindow > 0 and self.random_mode == "input":
            random_pos = torch.randn_like(pt_pos) * self.random_hwindow
            pt_pos = pt_pos + random_pos

        input_pos = pt_pos  # the first point is the reference point  [point_num, 5, 2]
        # ????????????5????????????????????????????????????????????? [point_num, 5, 3]
        if self.in_embed_fn is not None:
            pt_pos = pt_pos * (np.pi / self.kernel_hwindow)
            pt_pos = self.in_embed_fn(pt_pos)

        # ?????????????????????????????????5????????????????????? [point_num, 32]-->[point_num, 5, 32]
        # ?????????????????????????????????????????????
        img_embed_expand = img_embed[:, None].expand(point_num, self.num_pt, self.img_embed_cnl)
        x = torch.cat([pt_pos, img_embed_expand], dim=-1)

        # rays_x, rays_y???????????????????????????????????????????????????????????????????????????????????????
        rays_x, rays_y = uv[..., 0:1], uv[..., 1:]   # [point_num, 1]
        if self.spatial_embed_fn is not None:
            spatialx = rays_x / (self.W / 2 / np.pi) - np.pi
            spatialy = rays_y / (self.H / 2 / np.pi) - np.pi  # scale 2pi to match the freq in the embedder
            spatial = torch.cat([spatialx, spatialy], dim=-1)
            spatial = self.spatial_embed_fn(spatial)
            spatial = spatial[:, None, :].expand(-1, self.num_pt, -1)   # [point_num, 5, 3]
            x = torch.cat([x, spatial], dim=-1)

        # x: [point_num, 5, 3+32+3]
        # forward
        x1 = self.linears(x)
        x1 = torch.cat([x, x1], dim=-1) if self.short_cut else x1
        x1 = self.linears1(x1)

        delta_trans = None
        if self.optim_sv_trans:
            delta_trans, delta_pos, weight = torch.split(x1, [2, 2, 1], dim=-1)
        else:
            delta_pos, weight = torch.split(x1, [2, 1], dim=-1)

        if self.optim_trans:
            delta_trans = self.pattern_trans.expand(point_num, -1, -1) if self.isglobal \
                else self.pattern_trans[img_idx]

        if delta_trans is None:
            delta_trans = torch.zeros_like(delta_pos)

        # delta_trans, delta_pos    [point_num, 5, 2]
        # delta_trans?????????????????????????????????????????????xy???????????????????????????
        # delta_pos????????????????????????5??????????????????????????????????????????xy?????????
        delta_trans = delta_trans * 0.01
        # ?????????????????????5??????????????????dsk??????????????????????????????????????????????????????(?????????)???????????????
        new_rays_xy = delta_pos + input_pos
        weight = torch.softmax(weight[..., 0], dim=-1)

        if self.random_hwindow > 0 and self.random_mode == 'output':
            raise NotImplementedError(f"{self.random_mode} for self.random_mode is not implemented")

        poses = self.poses[img_idx]
        # get rays from offsetted pt position
        rays_x = (rays_x - self.K[0, 2] + new_rays_xy[..., 0]) / self.K[0, 0]
        rays_y = -(rays_y - self.K[1, 2] + new_rays_xy[..., 1]) / self.K[1, 1]
        dirs = torch.stack([rays_x - delta_trans[..., 0],
                            rays_y - delta_trans[..., 1],
                            -torch.ones_like(rays_x)], -1)   # [point_num, 5, 3]

        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., None, :] * poses[..., None, :3, :3],
                           -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        # ????????????????????????????????????????????????????????????xy?????????????????????????????????o????????????????????????????????????
        translation = torch.stack([
            delta_trans[..., 0],
            delta_trans[..., 1],
            torch.zeros_like(rays_x),
            torch.ones_like(rays_x)
        ], dim=-1)

        rays_o = torch.sum(translation[..., None, :] * poses, dim=-1)
        # rays_o = poses[..., None, :3, -1].expand_as(rays_d)

        align = new_rays_xy[:, 0, :].abs().mean()
        align += (delta_trans[:, 0, :].abs().mean() * 10)
        return torch.stack([rays_o[..., 0:3], rays_d], dim=-1), weight, align

