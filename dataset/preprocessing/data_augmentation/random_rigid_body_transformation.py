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
import torch
import random
import math

class RandomRigidBodyTransformation(object):
    """Random rigid body transformation for point cloud data
    """
    def __init__(self, x_limit=1.5, y_limit=1.5, z_limit=1.5, pitch_limit=2, yaw_limit=10, roll_limit=2):
        """Construtor

        Args:
            x_limit (float, optional): The maximum translation on x-axis in meter. Defaults to 1.5.
            y_limit (float, optional): The maximum translation on y-axis in meter. Defaults to 1.5.
            z_limit (float, optional): The maximum translation on z-axis in meter. Defaults to 1.5.
            pitch_limit (int, optional): The maximum rotation angle along y-axis in degree. Defaults to 2.
            yaw_limit (int, optional): The maximum rotation angle along z-axis in degree. Defaults to 10.
            roll_limit (int, optional): The maximum rotation angle along x-axis in degree. Defaults to 2.
        """
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.z_limit = z_limit
        self.pitch_limit = pitch_limit / 180 * math.pi
        self.yaw_limit = yaw_limit / 180 * math.pi
        self.roll_limit = roll_limit / 180 * math.pi

    def __call__(self, x: torch.tensor, verbose=False)-> torch.tensor:
        """Apply the random rigid body transformation of the given point cloud data

        Args:
            x (torch.tensor): Input point cloud with shape [N, 3]

        Returns:
            torch.tensor: Output point cloud with shape [N, 3]
        """
        roll = random.uniform(-self.roll_limit, self.roll_limit)
        pitch = random.uniform(-self.pitch_limit, self.pitch_limit)
        yaw = random.uniform(-self.yaw_limit, self.yaw_limit)
        
        trans_x = random.uniform(-self.x_limit, self.x_limit)
        trans_y = random.uniform(-self.y_limit, self.y_limit)
        trans_z = random.uniform(-self.z_limit, self.z_limit)

        rx = torch.tensor([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ], requires_grad=False, dtype=torch.double)

        ry = torch.tensor([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ], requires_grad=False, dtype=torch.double)

        rz = torch.tensor([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ], requires_grad=False, dtype=torch.double)

        rot = torch.matmul(torch.matmul(rx, ry), rz)
        # print(rot.dtype)
        # print(x.dtype)
        output = torch.transpose(torch.matmul(rot, torch.transpose(x, 0, 1)), 0, 1)
        output[:, 0] += trans_x
        output[:, 1] += trans_y
        output[:, 2] += trans_z
        
        if verbose:
            print(f"Rotation matrix: {rot}")
            print(f"Translation (x, y, z): ({trans_x}, {trans_y}, {trans_z})")
            
        return output
