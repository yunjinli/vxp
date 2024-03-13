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
from typing import Tuple
from mmdet3d.models.voxel_encoders.voxel_encoder import HardVFE
from torch import nn
import torch
from .second.middle import SparseLocalDescriptor

TwoTupleIntType = Tuple[Tuple[int]]


class VoxelLocalFeatureExtractor(nn.Module):
    """ VoxelNet + Sparse convolution
    """

    def __init__(self,
                 vfe_feat_channels=[32, 128],
                 voxel_size=(0.2, 0.2, 1),
                 pcd_range=(0, -25, -25, 50, 25, 25),
                 grid_zyx=[50, 250, 250],
                 middle_spconv_in_channels=128,
                 middle_spconv_out_channels=64,
                 middle_base_channels=16,
                 middle_encoder_channels=(
                     (16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
                 middle_encoder_paddings=(
                     (1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
                 middle_encoder_downsampling_other_dim=[
                     False, False, False, False]
                 ):
        """Construtor

        Args:
            vfe_feat_channels (list, optional): VFE channels. Defaults to [32, 128].
            voxel_size (tuple, optional): Voxel size. Defaults to (0.2, 0.2, 1).
            pcd_range (tuple, optional): Point cloud range [xmin, ymin, zmin, xmax, ymax, zmax]. Defaults to (0, -25, -25, 50, 25, 25).
            grid_zyx (list, optional): Grid dimension (z, y, x). Defaults to [50, 250, 250].
            middle_spconv_in_channels (int, optional): Input channel dimenstion to the middle sparse feature extractor. Defaults to 128.
            middle_spconv_out_channels (int, optional): Ouput channel dimenstion to the middle sparse feature extractor. Defaults to 64.
            middle_base_channels (int, optional): Out channels for conv_input layer. Defaults to 16.
            middle_encoder_channels (TwoTupleIntType, optional): Convolutional channels of each encode block. 
                Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
            middle_encoder_paddings (TwoTupleIntType, optional): Paddings of each encode block. 
                Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        """
        super(VoxelLocalFeatureExtractor, self).__init__()
        self.vfe = HardVFE(
            in_channels=3,
            feat_channels=vfe_feat_channels,
            with_voxel_center=True,
            voxel_size=voxel_size,
            point_cloud_range=pcd_range
        )
        self.middle = SparseLocalDescriptor(
            in_channels=middle_spconv_in_channels,
            sparse_shape=grid_zyx,
            base_channels=middle_base_channels,
            output_channels=middle_spconv_out_channels,
            encoder_channels=middle_encoder_channels,
            encoder_paddings=middle_encoder_paddings,
            encoder_downsampling_other_dim=middle_encoder_downsampling_other_dim
        )
        self.return_vfe_output = False

    def forward(self, features: torch.tensor, num_points: torch.tensor, coors: torch.tensor, batch_size: torch.tensor) -> torch.tensor:
        """Forward pass method

        Args:
            features (torch.tensor): Voxel feature with shape (N, M, 3)
            num_points (torch.tensor): Number of points per voxel with shape (N, )
            coors (torch.tensor): Voxel coordinates with shape (N, 4), [batch_idx, z, y, x]
            batch_size (torch.tensor): Batch size of the processed mini-batch
            Note that:
                N is the total number of voxel in this mini-batch.
                M is the maximum number of points per voxel in the voxelization configuration
                batch_idx is used as an id to seperate different samples within a mini-batch

        Returns:
            torch.tensor: Output tensor
        """
        vfe_output = self.vfe(
            features=features, num_points=num_points, coors=coors)
        middle_output = self.middle(vfe_output, coors, int(batch_size))

        if self.return_vfe_output:
            return vfe_output, middle_output
        else:
            return middle_output
