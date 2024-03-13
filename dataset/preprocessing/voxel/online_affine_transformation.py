#
# Created on Wed Jun 28 2023
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


class OnlineAffineTransformation(object):
    """Online affine transformation (SE3) for the voxel coodainte
    """

    def __init__(self, x_axis: list, y_axis: list, z_axis: list, trans: list):
        """Constructor

        Args:
            x_axis (list): The new basis x w.r.t current frame
            y_axis (list): The new basis y w.r.t current frame
            z_axis (list): The new basis z w.r.t current frame
            trans (list): The origin translation w.r.t current origin
        """

        self.rot = torch.tensor([
            [x_axis[0], y_axis[0], z_axis[0]],
            [x_axis[1], y_axis[1], z_axis[1]],
            [x_axis[2], y_axis[2], z_axis[2]]
        ], requires_grad=False, dtype=torch.float32)
        self.trans = torch.tensor(
            [[trans[0], trans[1], trans[2]]], requires_grad=False)

    def __call__(self, x: tuple[torch.tensor, torch.tensor, torch.tensor]) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Apply transformation of the seleted voxel coordinates

        Args:
            x (tuple[torch.tensor, torch.tensor, torch.tensor]): Voxel feature tuple (voxel_features, voxel_coors, voxel_num_of_points)
                                                                Only voxel_coors would be processed.
                                                                voxel_coors (z, y, x) with shape [N, 3].
                                                                Note that the voxel coordinates are in the order of zyx, which means
                                                                that we have to flip it before applying the transformation and then flip it back
                                                                at the end.

        Returns:
            tuple[torch.tensor, torch.tensor, torch.tensor]: Output Voxel feature tuple (voxel_features, voxel_coors, voxel_num_of_points)
        """
        features, coors, num_points = x
        coors = torch.transpose(torch.matmul(
            self.rot, torch.transpose(torch.flip(coors.type(torch.float32), dims=[1]), 0, 1)), 0, 1)
        coors += self.trans
        coors = torch.flip(coors, dims=[1]).type(torch.int32)

        return features, coors, num_points
