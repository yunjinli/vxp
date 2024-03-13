#
# Created on Jun 17 2023
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
import numpy as np
import torch


class OnlineRotX(object):
    """Online rotation along x axis for the pointcloud
    """

    def __init__(self, angle: float, use_degree=False):
        """Constructor 

        Args:
            angle (float): Rotation along x-axis
            use_degree (bool, optional): Flag for the angle unit. Defaults to False
        """
        if use_degree:
            angle = angle / 180 * np.pi

        self.rx = torch.tensor([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ], requires_grad=False)

    def __call__(self, x: torch.tensor) -> torch.tensor:
        """Apply transformation of the seleted point cloud

        Args:
            x (torch.tensor): Input point cloud with shape [N, 3]

        Returns:
            torch.tensor: Output point cloud with shape [N, 3]
        """
        x = torch.transpose(torch.matmul(
            self.rx, torch.transpose(x.type(torch.float64), 0, 1)), 0, 1)
        return x


class OnlineRotY(object):
    """Online rotation along y axis for the pointcloud
    """

    def __init__(self, angle: float, use_degree=False):
        """Constructor 

        Args:
            angle (float): Rotation along y-axis
            use_degree (bool, optional): Flag for the angle unit. Defaults to False
        """
        if use_degree:
            angle = angle / 180 * np.pi

        self.ry = torch.tensor([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], requires_grad=False)

    def __call__(self, x: torch.tensor) -> torch.tensor:
        """Apply transformation of the seleted point cloud

        Args:
            x (torch.tensor): Input point cloud with shape [N, 3]

        Returns:
            torch.tensor: Output point cloud with shape [N, 3]
        """
        x = torch.transpose(torch.matmul(
            self.ry, torch.transpose(x, 0, 1)), 0, 1)
        return x


class OnlineRotZ(object):
    """Online rotation along z axis for the pointcloud
    """

    def __init__(self, angle: float, use_degree=False):
        """Constructor 

        Args:
            angle (float): Rotation along z-axis
            use_degree (bool, optional): Flag for the angle unit. Defaults to False
        """
        if use_degree:
            angle = angle / 180 * np.pi

        self.rz = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], requires_grad=False)

    def __call__(self, x: torch.tensor) -> torch.tensor:
        """Apply transformation of the seleted point cloud

        Args:
            x (torch.tensor): Input point cloud with shape [N, 3]

        Returns:
            torch.tensor: Output point cloud with shape [N, 3]
        """
        x = torch.transpose(torch.matmul(
            self.rz, torch.transpose(x, 0, 1)), 0, 1)
        return x
