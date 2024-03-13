#
# Created on Tue Jul 04 2023
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
from typing import Union
import numpy as np


class OnlineLidarRangeImage(object):
    """Project LiDAR point cloud to the image by 'Pinhole' camera model
    """

    def __init__(self, fx: float = 964.828979 / 4,
                 fy: float = 964.828979 / 4,
                 cx: float = 643.788025 / 4,
                 cy: float = 484.407990 / 4,
                 h: int = 960 // 4,
                 w: int = 1280 // 4,
                 mode: str = 'image',
                 remove_far: bool = False
                 ):
        """Construtor

        Args:
            fx (float, optional): focal length x. Defaults to 964.828979/4.
            fy (float, optional): focal length y. Defaults to 964.828979/4.
            cx (float, optional): camera center x. Defaults to 643.788025/4.
            cy (float, optional): camera center y. Defaults to 484.407990/4.
            h (int, optional): image height. Defaults to 960//4.
            w (int, optional): image width. Defaults to 1280//4.
            mode (str, optional): Output form of the callable. Defaults to 'image'. Available mode [image, points]
            remove_far (bool, optional): Whether to only keep the projected point within a pixel 
                                        that has the closest distance
                                        to the camera center. Defaults to False.
        """
        self.K = torch.tensor(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], requires_grad=False, dtype=torch.float32
        )
        self.h = h
        self.w = w
        self.mode = mode
        self.remove_far = remove_far
        assert self.mode in ['image', 'points']

    def __call__(self, x: torch.tensor) -> torch.tensor:
        """Callable

        Args:
            x (torch.tensor): input submap with shape [N, 3] 

        Returns:
            torch.tensor: Output depens on mode:
                'image': Return sparse inverse depth map. Shape [H, W]
                'points': Return a list of projected points with inverse depth. Shape [N, 4]. 
                          Note that the last dimenstion is the indices corresponding to the original point sets
        """
        I = torch.zeros(size=(self.h, self.w))
        x = torch.transpose(torch.matmul(
            self.K, torch.transpose(x.type(torch.float32), 0, 1)), 0, 1)
        x = torch.cat((x, torch.arange(x.shape[0]).view(-1, 1)), dim=-1)

        # print(x.shape)
        # print(x)
        x[:, :2] /= x[:, 2][:, None]
        x = x[x[:, 0] < self.w, :]
        x = x[x[:, 1] < self.h, :]
        x = x[x[:, 0] > 0, :]
        x = x[x[:, 1] > 0, :]
        x[:, 2] = 1 / x[:, 2]
        if self.remove_far:
            unique_uv, indices = x[:, :2].type(
                torch.int).unique(dim=0, return_inverse=True)
            inv_depth = x[:, 2]
            x_new = torch.zeros(size=(unique_uv.shape[0], 4))
            for i, uv in enumerate(unique_uv):
                v, u = uv
                I[u.numpy(), v.numpy()], idx = inv_depth[(indices == i)].max(dim=0)
                x_new[i, 0] = x[(indices == i), 0][idx]
                x_new[i, 1] = x[(indices == i), 1][idx]
                x_new[i, 2] = I[u.numpy(), v.numpy()]
                x_new[i, 3] = x[(indices == i), 3][idx]
                # print(x_new[i, :])
            x = torch.clone(x_new)
        else:
            I[x[:, 1].type(torch.int), x[:, 0].type(torch.int)] = x[:, 2]

        if self.mode == 'image':
            return I / I.max()
        elif self.mode == 'points':
            return x


class OnlineLidarRangeDenseMap(object):
    """Generate dense depth map
    """

    def __init__(self, grid_size: int = 6, h: int = 960 // 4, w: int = 1280 // 4):
        """Constructor

        Args:
            grid_size (int, optional): Grid size. Defaults to 6.
            h (int, optional): Image height. Defaults to 960//4.
            w (int, optional): Image width. Defaults to 1280//4.
        """
        self.grid_size = grid_size
        self.h = h
        self.w = w

    def __dense_map(self, Pts: Union[torch.tensor, np.ndarray], n: int, m: int, grid: int) -> np.ndarray:
        """Generate dense map

        Args:
            Pts (Union[torch.tensor, np.ndarray]): List of points [x, y, inv_d]^T with shape [3, N]
            n (int): Image width
            m (int): Image height
            grid (int): Grid size

        Returns:
            np.ndarray: Output dense depth map
        """
        ng = 2 * grid + 1

        mX = np.zeros((m, n)) + float("inf")
        mY = np.zeros((m, n)) + float("inf")
        mD = np.zeros((m, n))
        mX[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
        mY[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
        mD[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[2]

        KmX = np.zeros((ng, ng, m - ng, n - ng))
        KmY = np.zeros((ng, ng, m - ng, n - ng))
        KmD = np.zeros((ng, ng, m - ng, n - ng))

        for i in range(ng):
            for j in range(ng):
                KmX[i, j] = mX[i: (m - ng + i),
                               j: (n - ng + j)] - grid - 1 + i
                KmY[i, j] = mY[i: (m - ng + i),
                               j: (n - ng + j)] - grid - 1 + i
                KmD[i, j] = mD[i: (m - ng + i), j: (n - ng + j)]
        S = np.zeros_like(KmD[0, 0])
        Y = np.zeros_like(KmD[0, 0])

        for i in range(ng):
            for j in range(ng):
                s = 1/np.sqrt(KmX[i, j] * KmX[i, j] +
                              KmY[i, j] * KmY[i, j])
                Y = Y + s * KmD[i, j]
                S = S + s

        S[S == 0] = 1
        out = np.zeros((m, n))
        out[grid + 1: -grid, grid + 1: -grid] = Y/S
        return out

    def __call__(self, x: Union[torch.tensor, np.ndarray]) -> np.ndarray:
        """Callable

        Args:
            x (Union[torch.tensor, np.ndarray]): List of points [x, y, inv_d] with shape [N, 3]

        Returns:
            np.ndarray: Output dense depth map
        """
        return self.__dense_map(x.T, self.w, self.h, self.grid_size)
