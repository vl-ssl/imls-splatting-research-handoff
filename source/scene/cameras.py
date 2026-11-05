import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 gt_normal=None, gt_disparity=None, image_background=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.gt_alpha_mask = gt_alpha_mask
        self.image_background = image_background
        
        self.original_normal = None
        if gt_normal is not None:
            # 1. 转换为 Tensor 并上传 GPU
            if isinstance(gt_normal, np.ndarray):
                self.original_normal = torch.from_numpy(gt_normal).float().to(self.data_device)
            elif isinstance(gt_normal, torch.Tensor):
                self.original_normal = gt_normal.to(self.data_device)
            
            # 2. 归一化 [0, 255] -> [0, 1]
            if self.original_normal.max() > 1.1:
                self.original_normal /= 255.0
            
            self.original_normal = self.original_normal.clamp(0.0, 1.0)

            # 3. 维度调整: (H, W, 3) -> (3, H, W)
            # dataset_readers 读取的一般是 H,W,C，而 renderer 需要 C,H,W
            if len(self.original_normal.shape) == 3 and self.original_normal.shape[2] == 3:
                self.original_normal = self.original_normal.permute(2, 0, 1)

            # 4. 应用 Alpha Mask (Mask 背景置黑)
            # 这对于 Ref-NeRF 数据集非常重要，否则物体边缘会因为背景的法线噪声产生伪影
            if self.gt_alpha_mask is not None:
                self.original_normal *= self.gt_alpha_mask.to(self.data_device)

        self.original_disparity = None
        if gt_disparity is not None:
            self.original_disparity = torch.as_tensor(
                gt_disparity,
                dtype=torch.float32,
                device=self.data_device,
            )

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

