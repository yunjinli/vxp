#
# Created on Tue Aug 08 2023
# The MIT License (MIT)
# Copyright (c) 2023 Yun-Jin Li (Jim), Technical University of Munich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
import torch
import torch.nn as nn
from typing import List

class V2PProjection(object):
    """Voxel to Pixel Projection module
    """

    def __init__(self,
                 fx: float = 964.828979 / 4,
                 fy: float = 964.828979 / 4,
                 cx: float = 643.788025 / 4,
                 cy: float = 484.407990 / 4,
                 h: int = 960 // 4,
                 w: int = 1280 // 4,
                 R: List[List] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
                 voxel_size: List[int] = [0.75, 0.75, 0.75],
                 pcd_range: List[int] = [0, -25, -25, 50, 25, 25],
                 device: str = 'cpu'
                 ):
        """Constructor

        Args:
            fx (float, optional): X focal length of the camera. Defaults to 964.828979/4.
            fy (float, optional): Y focal length of the camera. Defaults to 964.828979/4.
            cx (float, optional): X center of the camera. Defaults to 643.788025/4.
            cy (float, optional): Y center of the camera. Defaults to 484.407990/4.
            h (int, optional): Height of the image. Defaults to 960//4.
            w (int, optional): Width of the image. Defaults to 1280//4.
            R (List[List], optional): Axes alignment before projection using pinhole camera model
            voxel_size (List[int], optional): Size of the voxel grid. Defaults to [0.75, 0.75, 0.75].
            pcd_range (List[int], optional): Poincloud ranges [xmin, ymin, zmin, xmax, ymax, zmax]. Defaults to [0, -25, -25, 50, 25, 25].
            device (str, optional): cpu or cuda. Defaults to 'cpu'.
        """
        self.K = torch.tensor(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], requires_grad=False, dtype=torch.float32
        ).to(device)
        self.h = h
        self.w = w
        vx = voxel_size[0]
        vy = voxel_size[1]
        vz = voxel_size[2]
        x_offset = vx / 2 + pcd_range[0]
        y_offset = vy / 2 + pcd_range[1]
        z_offset = vz / 2 + pcd_range[2]

        self.R = torch.tensor(
            R, requires_grad=False, dtype=torch.float32
        ).to(device)

        self.V2P = torch.tensor(
            [
                [vx, 0, 0, x_offset],
                [0, vy, 0, y_offset],
                [0, 0, vz, z_offset]
            ], requires_grad=False, dtype=torch.float32
        ).to(device)

        self.KRV2P = self.K @ self.R @ self.V2P
        self.device = device

    def __call__(self, coors: torch.tensor) -> torch.tensor:
        """Forward pass of the projection module

        Args:
            coors (torch.tensor): Coordinate in {V} (voxel grid coordinate frame)

        Returns:
            torch.tensor: Pixel coordinate with (u, v, inv_depth)
        """
        # coors is with shape [N, 4], each row represents [batch_id, z, y, x]
        coors_cam = coors.clone().to(self.device)
        coors = torch.stack([coors[:, 0], coors[:, 3], coors[:, 2],
                             coors[:, 1], torch.ones(coors.shape[0]).to(self.device)], dim=-1)
        coors_cam[:, 1:] = torch.transpose(torch.matmul(
            self.KRV2P, torch.transpose(coors[:, 1:].type(torch.float32), 0, 1)), 0, 1)

        coors_cam = torch.cat((coors_cam, torch.arange(
            coors_cam.shape[0]).view(-1, 1).to(self.device)), dim=-1)

        coors_cam[:, 1:3] /= coors_cam[:, 3][:, None]
        mask_1 = (coors_cam[:, 1] > 0) & (coors_cam[:, 1] < self.w)
        mask_2 = (coors_cam[:, 2] > 0) & (coors_cam[:, 2] < self.h)
        final_mask = mask_1 & mask_2
        coors_cam = coors_cam[final_mask]
        coors_cam[:, 3] = 1 / coors_cam[:, 3]

        return coors_cam
    def __repr__(self):
        return self.__class__.__name__ + '(KRV2P={0}, device={1})'.format(self.KRV2P, self.device)