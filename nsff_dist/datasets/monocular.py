import cv2
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from collections import defaultdict
from scipy.stats import linregress
from PIL import Image
from torchvision import transforms as T

from . import ray_utils, depth_utils, colmap_utils, flowlib

'''
MonocularDataset:
    1. center操作的直觉理解
    2. 距离计算 
    3. projection
'''
class MonocularDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(512, 288),
                 start_end=(0, 30), cache_dir=None, hard_sampling=False):
        """
        split options:
            train - training mode (used in train.py) rays are from all images
            val - validation mode (used in val.py) validate on the middle frame
            test - test on the training poses and times
            test_spiral - create spiral poses around the whole trajectory,
                          time is gradually advanced (only integers for now)
            test_spiralX - create spiral poses (fixed time) around training pose X
            test_fixviewX_interpY - fix view to training pose X and interpolate Y frames
                                    between each integer timestamps, from start to end
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.cam_train = [0]
        self.cam_test = 1
        self.start_frame = start_end[0]
        self.end_frame = start_end[1]
        self.cache_dir = cache_dir
        self.hard_sampling = hard_sampling
        self.define_transforms()
        self.read_meta()

    def read_meta(self):
        # 1. read inputs
        self.image_paths = \
            sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))[self.start_frame:self.end_frame]
        self.disp_paths = \
            sorted(glob.glob(os.path.join(self.root_dir, 'disps/*')))[self.start_frame:self.end_frame]
        self.mask_paths = \
            sorted(glob.glob(os.path.join(self.root_dir, 'masks/*')))[self.start_frame:self.end_frame]
        self.flow_fw_paths = \
            sorted(glob.glob(os.path.join(self.root_dir, 'flow_fw/*.flo')))[self.start_frame:self.end_frame] + ['dummy']
        self.flow_bw_paths = \
            ['dummy'] + sorted(glob.glob(os.path.join(self.root_dir, 'flow_bw/*.flo')))[self.start_frame:self.end_frame]
        self.N_frames = len(self.image_paths)

        # 2. read intrinsics
        camdata = colmap_utils.read_cameras_binary(os.path.join(self.root_dir,
                                                                'sparse/0/cameras.bin'))
        H = camdata[1].height
        W = camdata[1].width
        f, cx, cy, _ = camdata[1].params

        self.K = np.array([[f, 0, W/2],
                           [0, f, H/2],
                           [0,  0,  1]], dtype=np.float32)
        self.K[0] *= self.img_wh[0]/W
        self.K[1] *= self.img_wh[1]/H

        # 3. read extrinsics
        # 这里读取的colmap计算的外参数据应该不是按照序号排列的，因此后面才会通过perm对重组的外参矩阵数组进行重排序后再取范围
        imdata = colmap_utils.read_images_binary(os.path.join(self.root_dir,
                                                              'sparse/0/images.bin'))
        perm = np.argsort([imdata[k].name for k in imdata])
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
        # 每一张图片都对应一个外参矩阵
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)[perm]
        w2c_mats = w2c_mats[self.start_frame:self.end_frame] # (N_frames, 4, 4)
        poses = np.linalg.inv(w2c_mats)[:, :3] # (N_frames, 3, 4)

        # 3. read bounds
        pts3d = colmap_utils.read_points3d_binary(os.path.join(self.root_dir,
                                                               'sparse/0/points3D.bin'))
        # 最后一维存放的是同一个轴上的坐标（x,y,z） 坐标值是世界坐标系下的
        # pts_w[0][0]存放的是所有点的x坐标
        # pts_w[0][1]存放的是所有点的y坐标
        # pts_w[0][2]存放的是所有点的z坐标
        pts_w = np.zeros((1, 3, len(pts3d))) # (1, 3, N_points)
        visibilities = np.zeros((len(poses), len(pts3d))) # (N_frames, N_points)
        for i, k in enumerate(pts3d):
            # pts3d[k].xyz 表示的是第k个点的xyz坐标
            # pts3d[k].image_ids 表示的是第k个点都在哪些帧的画面中
            pts_w[0, :, i] = pts3d[k].xyz
            for j in pts3d[k].image_ids: # 对于表示的是第k个点所在的第j帧来讲
                if self.start_frame <= j-1 < self.end_frame: # 如果这一帧在可见帧的范围内
                    visibilities[j-1-self.start_frame, i] = 1 # 则将可见矩阵中该帧(j)所对应的改点(k)初标1

        min_depth = 1e8
        for i in range(self.N_frames):
            # for each image, compute the nearest depth according to real depth from COLMAP
            # and the disparity estimated by monodepth.
            # (using linear regression to find the best scale and shift)
            disp = cv2.imread(self.disp_paths[i], cv2.IMREAD_ANYDEPTH).astype(np.float32)
            disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)
            pts_w_ = np.concatenate([pts_w[0], np.ones((1, len(pts3d)))], 0) # (4, N_points): (x,y,z,1)^T
            visibility_i = visibilities[i] # (N_points) 1 if visible
            # 取出当前帧中所有可见点的 homogeneous coordiante: [x,y,z,1]
            pts_w_v = pts_w_[:, visibility_i==1] # (4, N_points_v)
            # 世界坐标系(pts_w_v)转化到相机坐标系(pts_c_v)
            pts_c_v = (w2c_mats[i] @ pts_w_v)[:3] # (3, N_points_v)
            # 相机坐标系(pts_c_v)转化到图像像素坐标系(pts_uvd_v)
            pts_uvd_v = self.K @ pts_c_v
            # 图像像素坐标系(pts_uvd_v)处以homogeneous coordiante weight，获得统一后的uv值
            pts_uv_v = (pts_uvd_v[:2]/pts_uvd_v[2:]).T # (N_points_v, 2)
            # uv值离散化，用于取深度模型预测值disp中对应像素点上的深度值
            pts_uv_v = pts_uv_v.astype(int) # to integer pixel coordinates
            # 只取设定图像范围内的点
            pts_uv_v[:, 0] = np.clip(pts_uv_v[:, 0], 0, self.img_wh[0]-1)
            pts_uv_v[:, 1] = np.clip(pts_uv_v[:, 1], 0, self.img_wh[1]-1)
            # pts_d_v: 使用先验模型预测的深度值，colmap计算的当前帧中可见点在相机坐标系下z轴上的距离
            pts_d_v = pts_uvd_v[2]
            reg = linregress(1/pts_d_v, disp[pts_uv_v[:, 1], pts_uv_v[:, 0]])
            # 这部分认为colmap计算的距离值最准，如果预训练模型的预测距离值准确度较高的话就用预训练模型的输出距离
            # 如果不准的话就用colmap计算的距离值，那为什么不直接用colmap计算的距离值呢
            if reg.rvalue**2 > 0.9: # if the regression is authentic
                min_depth = min(min_depth, reg.slope/(np.percentile(disp, 95)-reg.intercept))
            else:
                min_depth = min(min_depth, np.percentile(pts_d_v, 5))
                # 这一套不直接取最小值而是先取百分之五分位数，然后再*0.75的操作感觉安全性不是很高
                # (万一百分之五分位数与最小值间差的较大导致*0.75并调整scale后百分之五分位数处的z轴坐标略大于1，但原本最小距离处调整z轴scale后的距离小于1怎么办呢)

        self.nearest_depth = min_depth * 0.75

        # Step 2: correct poses
        # change "right down front" of COLMAP to "right up back"
        self.poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
        self.poses = colmap_utils.center_poses(self.poses)

        # Step 3: correct scale
        self.scale_factor = self.nearest_depth
        self.poses[..., 3] /= self.scale_factor

        # create projection matrix, used to compute optical flow
        bottom = np.zeros((self.N_frames, 1, 4))
        bottom[..., -1] = 1
        rt = np.linalg.inv(np.concatenate([self.poses, bottom], 1))[:, :3]
        rt[:, 1:] *= -1 # "right up back" to "right down forward" for cam projection
        self.Ps = self.K @ rt
        self.Ps = torch.FloatTensor(self.Ps).unsqueeze(0) # (1, N_frames, 3, 4)
        self.Ks = torch.FloatTensor(self.K).unsqueeze(0) # (1, 3, 3)

        # Step 4: create ray buffers
        if self.split == 'train':
            self.last_t = -1
            # 图像坐标系与相机坐标系与图片/帧/视角无关, 单目没有深度可言
            directions, uv = ray_utils.get_ray_directions(
                                self.img_wh[1], self.img_wh[0], self.K, return_uv=True)
            if self.cache_dir:
                self.rays_dict = torch.load(os.path.join(self.cache_dir, 'rays_dict.pt'))
            else:
                self.rays_dict = {}
                for t in range(self.N_frames):
                    img = Image.open(self.image_paths[t]).convert('RGB')
                    img = img.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img).view(3, -1).T # (h*w, 3) RGB

                    # 世界坐标系与图片/帧/视角有关，针对每个图片/帧/视角都要单独生成世界坐标点
                    # self.poses[t]是第t张图片/视角的3*4 c2w矩阵
                    # self.poses[t, 2, 3]是第t张图片/视角在世界坐标系下z方向偏移量
                    c2w = torch.FloatTensor(self.poses[t])
                    rays_o, rays_d = ray_utils.get_rays(directions, c2w) # both (h*w, 3)
                    # shift_near = max(1.0, -self.poses[t, 2, 3]) 不能比1小
                    shift_near = -min(-1.0, self.poses[t, 2, 3])
                    rays_o, rays_d = ray_utils.get_ndc_rays(self.K, 1.0, 
                                                            shift_near, rays_o, rays_d)

                    rays_t = t * torch.ones(len(rays_o), 1) # (h*w, 1)

                    disp = cv2.imread(self.disp_paths[t], cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)
                    disp = torch.FloatTensor(disp).reshape(-1, 1) # (h*w, 1)

                    mask = Image.open(self.mask_paths[t]).convert('L')
                    mask = mask.resize(self.img_wh, Image.NEAREST)
                    mask = self.transform(mask).flatten() # (h*w)
                    rays_mask = mask.unsqueeze(-1) # 0:static, 1:dynamic

                    if t < self.N_frames-1:
                        flow_fw = flowlib.read_flow(self.flow_fw_paths[t])
                        flow_fw = flowlib.resize_flow(flow_fw, self.img_wh[0], self.img_wh[1])
                        flow_fw = torch.FloatTensor(flow_fw.reshape(-1, 2))
                    else:
                        # 最后一帧的后向光流为0
                        flow_fw = torch.zeros(len(rays_o), 2)

                    if t >= 1:
                        flow_bw = flowlib.read_flow(self.flow_bw_paths[t])
                        flow_bw = flowlib.resize_flow(flow_bw, self.img_wh[0], self.img_wh[1])
                        flow_bw = torch.FloatTensor(flow_bw.reshape(-1, 2))
                    else:
                        # 第一帧的前向光流为0
                        flow_bw = torch.zeros(len(rays_o), 2)

                    rays = [rays_o, rays_d, img, rays_t,
                            disp, rays_mask,
                            uv+flow_fw, uv+flow_bw]
                    self.rays_dict[t] = torch.cat(rays, 1) # (h*w, 3+3+3+1+1+1+2+2=16)

            if self.hard_sampling:
                self.weights = [np.ones(self.img_wh[0]*self.img_wh[1])
                                for _ in range(self.N_frames)]
    ############################################################################################
        elif self.split == 'test':
            self.poses_test = self.poses.copy()
            self.image_paths_test = self.image_paths

        elif self.split.startswith('test_fixview'):
            # fix to target view and change time
            target_idx = int(self.split.split('_')[1][7:])
            self.poses_test = np.tile(self.poses[target_idx], (self.N_frames, 1, 1))

        elif self.split.startswith('test_spiral'):
            if self.split == 'test_spiral': # spiral on the whole sequence
                max_trans = np.percentile(np.abs(np.diff(self.poses[:, 0, 3])), 10)
                radii = np.array([max_trans, max_trans, 0])
                self.poses_test = colmap_utils.create_spiral_poses(
                                    self.poses, radii, n_poses=6*self.N_frames)
            else: # spiral on the target idx
                target_idx = int(self.split.split('_')[1][6:])
                max_trans = np.abs(self.poses[0, 0, 3]-self.poses[-1, 0, 3])/5
                self.poses_test = colmap_utils.create_wander_path(
                                    self.poses[target_idx], max_trans=max_trans, n_poses=60)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            # //1000 ???
            return self.img_wh[0]*self.img_wh[1]*self.N_frames//1000
        if self.split == 'val': return 1
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train':
            # first select t (which image)
            if self.last_t == -1: # the very first sample, uniformly random
                t = np.random.choice(self.N_frames)
            else:
                # for EACH worker, sample t outside some window of the last sampled t
                # to avoid bad static net convergence (dynamic explained by static)
                w_size = 5
                valid_t = list(set(range(self.N_frames))-
                               set(range(self.last_t-w_size, self.last_t+w_size+1)))
                t = np.random.choice(valid_t)
            self.last_t = t
            # then select the rays
            if self.hard_sampling: # random rays according to weights
                #TODO: when loading checkpoints, the weights must be loaded or recalculated too!
                rand_idx = np.random.choice(np.arange(self.img_wh[0]*self.img_wh[1]), 
                                            self.batch_size,
                                            p=self.weights[t]/self.weights[t].sum())
            else: # uniformly random
                rand_idx = np.random.choice(len(self.rays_dict[t]), self.batch_size)
            rays = self.rays_dict[t][rand_idx]
            sample = {'rays': rays[:, :6],
                      'rgbs': rays[:, 6:9],
                      'ts': rays[:, 9].long(),
                      'cam_ids': 0*rays[:, 9].long(),
                      'disps': rays[:, 10],
                      'rays_mask': rays[:, 11],
                      'uv_fw': rays[:, 12:14],
                      'uv_bw': rays[:, 14:16]}
            if self.hard_sampling: sample['rand_idx'] = torch.LongTensor(rand_idx)
        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.poses[self.N_frames//2])
                t = self.N_frames//2
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])
                if self.split == 'test':
                    t = idx
                elif self.split.startswith('test_spiral'):
                    if self.split == 'test_spiral': 
                        t = int(idx/len(self.poses_test)*self.N_frames)
                    else:
                        t = int(self.split.split('_')[1][6:])
                elif self.split.startswith('test_fixview'):
                    t = idx
                else: t = 0

            directions = ray_utils.get_ray_directions(self.img_wh[1], self.img_wh[0], self.K)
            rays_o, rays_d = ray_utils.get_rays(directions, c2w)
            shift_near = -min(-1.0, c2w[2, 3])
            rays_o, rays_d = ray_utils.get_ndc_rays(self.K, 1.0, 
                                                    shift_near, rays_o, rays_d)

            rays_t = t * torch.ones(len(rays_o), dtype=torch.long) # (h*w)

            rays = torch.cat([rays_o, rays_d], 1) # (h*w, 6)

            sample = {'rays': rays, 'ts': rays_t, 'c2w': c2w}

            sample['cam_ids'] = 0
            img = Image.open(self.image_paths[t]).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).T # (h*w, 3)
            sample['rgbs'] = img

            disp = cv2.imread(self.disp_paths[t], cv2.IMREAD_ANYDEPTH).astype(np.float32)
            disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)
            sample['disp'] = torch.FloatTensor(disp.flatten())

            mask = Image.open(self.mask_paths[t]).convert('L')
            mask = mask.resize(self.img_wh, Image.NEAREST)
            mask = self.transform(mask).flatten() # (h*w)
            sample['mask'] = mask

            if t < self.N_frames-1:
                flow_fw = flowlib.read_flow(self.flow_fw_paths[t])
                flow_fw = flowlib.resize_flow(flow_fw, self.img_wh[0], self.img_wh[1])
                sample['flow_fw'] = flow_fw
            else:
                sample['flow_fw'] = torch.zeros(self.img_wh[1], self.img_wh[0], 2)

            if t >= 1:
                flow_bw = flowlib.read_flow(self.flow_bw_paths[t])
                flow_bw = flowlib.resize_flow(flow_bw, self.img_wh[0], self.img_wh[1])
                sample['flow_bw'] = flow_bw
            else:
                sample['flow_bw'] = torch.zeros(self.img_wh[1], self.img_wh[0], 2)

        return sample