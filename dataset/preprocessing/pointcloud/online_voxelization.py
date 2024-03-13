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
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape 
import torch

class OnlineVoxelization(object):
    """Online voxelization for point cloud data
    """
    def __init__(self, point_cloud_range=[0, -25, -25, 50, 25, 25],
                    max_num_points=35,
                    voxel_size=[0.75, 0.75, 0.75],
                    max_voxels=10000):
        """Constructor

        Args:
            point_cloud_range (list, optional): Range in point cloud to be voxelized [xmin, ymin, zmin, xmax, ymax, zmax]. Defaults to [0, -25, -25, 50, 25, 25].
            max_num_points (int, optional): Maximum number of points in a single voxel. Defaults to 35.
            voxel_size (list, optional): Size of the voxel [x, y, z]. Defaults to [0.75, 0.75, 0.75].
            max_voxels (int, optional): Maxumum number of voxels to be generated. Defaults to 10000.
        """
        self.vox = VoxelizationByGridShape(point_cloud_range=point_cloud_range,
                                max_num_points=max_num_points,
                                voxel_size=voxel_size,
                                max_voxels=max_voxels)
    def __call__(self, pcd: torch.tensor)-> tuple:
        """Perform voxelization on a single pointcloud data

        Args:
            pcd (torch.tensor): Input point cloud with shape [N, 3]

        Returns:
            tuple: Output voxelization tuple
                voxels_out: with shape [M, 35, 3] 
                coors_out: with shape [M, 3], note that the coordinate is in order [z, y, x]
                num_points_per_voxel_out: with shape [M, ]
        """
        voxels_out, coors_out, num_points_per_voxel_out = self.vox(pcd)
        voxels_out = voxels_out.to(torch.float32)
        return voxels_out, coors_out, num_points_per_voxel_out

