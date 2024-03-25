#
# Created on May 23 2023
# The MIT License (MIT)
# Copyright (c) 2023 Yun-Jin Li (Jim), Technical University of Munich (TUM)
#
# Implementation based on the paper to the best of our understanding:
#           Global visual localization in LiDAR-maps through shared 2D-3D embedding space from Cattaneo et. al.
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
from .pooling.netvlad import NetVLAD
from .pooling.GeM import GeM
from detectron2.layers.aspp import ASPP
from mmdet3d.models.backbones.second import SECOND
from mmdet3d.models.middle_encoders.sparse_encoder import SparseEncoder
from .second.middle import SparseMiddleExtractor

TwoTupleIntType = Tuple[Tuple[int]]


class SecondNetvlad(nn.Module):
    def __init__(self,
                 vfe_feat_channels=[32, 128],
                 voxel_size=(0.2, 0.2, 1),
                 pcd_range=(0, -25, -25, 50, 25, 25),
                 grid_zyx=[50, 250, 250],
                 middle_spconv_in_channels=128,
                 middle_spconv_out_channels=64,
                 second_in_channels=128,
                 second_out_channels=[128, 256, 512],
                 second_layer_nums=[3, 5, 5],
                 second_layer_strides=[2, 2, 2],
                 num_clusters=64,
                 encoder_dim=512,
                 vladv2=False, output_dim=None, gating=False):
        super(SecondNetvlad, self).__init__()
        self.vfe = HardVFE(
            in_channels=3,
            feat_channels=vfe_feat_channels,
            with_voxel_center=True,
            voxel_size=voxel_size,
            point_cloud_range=pcd_range
        )
        self.middle = SparseEncoder(
            in_channels=middle_spconv_in_channels,
            sparse_shape=grid_zyx,
            output_channels=middle_spconv_out_channels
        )
        self.second = SECOND(
            in_channels=second_in_channels,
            out_channels=second_out_channels,
            layer_nums=second_layer_nums,
            layer_strides=second_layer_strides,
        )
        # NetVLAD
        self.netvlad = NetVLAD(
            num_clusters=num_clusters, dim=encoder_dim, output_dim=output_dim, gating=gating)

    def forward(self, features, num_points, coors, batch_size):
        vfe_output = self.vfe(
            features=features, num_points=num_points, coors=coors)
        middle_output = self.middle(vfe_output, coors, int(batch_size))
        # print(middle_output.shape)
        second_output = self.second(middle_output)
        # print(len(second_output))
        # for so in second_output:
        #     print(so.shape)
        # print(second_output[-1].shape)
        output = self.netvlad(second_output[-1])
        return output


class SecondAsppNetvlad(nn.Module):
    """SECOND + ASPP + NetVLAD
    """

    def __init__(self,
                 vfe_feat_channels=[32, 128],
                 voxel_size=(0.2, 0.2, 1),
                 pcd_range=(0, -25, -25, 50, 25, 25),
                 grid_zyx=[50, 250, 250],
                 middle_spconv_in_channels=128,
                 middle_spconv_out_channels=64,
                 # second_in_channels=128,
                 # second_out_channels=[128, 256, 512],
                 # second_layer_nums=[3, 5, 5],
                 # second_layer_strides=[2, 2, 2],
                 aspp_in_channels=192,
                 aspp_out_channels=512,
                 aspp_rate=[6, 12, 18],
                 aspp_norm='BN',
                 aspp_dropout=0.1,
                 use_depthwise_separable_conv=False,
                 num_clusters=64,
                 encoder_dim=512,
                 vladv2=False, output_dim=None, gating=False):
        """Construtor

        Args:
            vfe_feat_channels (list, optional): VFE channels. Defaults to [32, 128].
            voxel_size (tuple, optional): Voxel size. Defaults to (0.2, 0.2, 1).
            pcd_range (tuple, optional): Point cloud range [xmin, ymin, zmin, xmax, ymax, zmax]. Defaults to (0, -25, -25, 50, 25, 25).
            grid_zyx (list, optional): Grid dimension (z, y, x). Defaults to [50, 250, 250].
            middle_spconv_in_channels (int, optional): Input channel dimenstion to the middle sparse feature extractor. Defaults to 128.
            middle_spconv_out_channels (int, optional): Ouput channel dimenstion to the middle sparse feature extractor. Defaults to 64.
            aspp_in_channels (int, optional): ASPP input channel size. Defaults to 192.
            aspp_out_channels (int, optional): ASPP output channel size. Defaults to 512.
            aspp_rate (list, optional): ASPP dilation rates. Defaults to [6, 12, 18].
            aspp_norm (str, optional): ASPP normalization function. Defaults to 'BN'.
            aspp_dropout (float, optional): ASPP dropout probability. Defaults to 0.1.
            use_depthwise_separable_conv (bool, optional): ASPP enable depthwise convolution. Defaults to False.
            num_clusters (int, optional): NetVLAD number of clusters. Defaults to 64.
            encoder_dim (int, optional): NetVLAD local feature extractor dimenstion. Defaults to 512.
            vladv2 (bool, optional): Set NetVLAD version. Defaults to False.
            output_dim (_type_, optional): NetVLAD output dimenstion. Defaults to None.
            gating (bool, optional): NetVLAD set to enable context gating. Defaults to False.
        """
        super(SecondAsppNetvlad, self).__init__()
        self.vfe = HardVFE(
            in_channels=3,
            feat_channels=vfe_feat_channels,
            with_voxel_center=True,
            voxel_size=voxel_size,
            point_cloud_range=pcd_range
        )
        self.middle = SparseEncoder(
            in_channels=middle_spconv_in_channels,
            sparse_shape=grid_zyx,
            output_channels=middle_spconv_out_channels
        )
        # ASPP
        self.aspp = ASPP(aspp_in_channels,
                         aspp_out_channels,
                         aspp_rate,
                         norm=aspp_norm,
                         activation=nn.ReLU(),
                         dropout=aspp_dropout,
                         use_depthwise_separable_conv=use_depthwise_separable_conv)
        # NetVLAD
        self.netvlad = NetVLAD(
            num_clusters=num_clusters, dim=encoder_dim, output_dim=output_dim, gating=gating)

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
        aspp_output = self.aspp(middle_output)
        output = self.netvlad(aspp_output)
        return output


class SecondAsppNetvladV2(nn.Module):
    """ SECOND + ASPP + NetVLAD (v2)
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
                 middle_downsample_dimension='z',
                 middle_encoder_downsampling_other_dim=[
                     False, False, False, False],
                 aspp_in_channels=192,
                 aspp_out_channels=512,
                 aspp_rate=[6, 12, 18],
                 aspp_norm='BN',
                 aspp_dropout=0.1,
                 use_depthwise_separable_conv=False,
                 num_clusters=64,
                 encoder_dim=512,
                 vladv2=False, output_dim=None, gating=False):
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
            middle_downsample_dimension (str, optional): Downsampling dimension.
                z: Bird's eys view (From the top)
                x: Front view
                y: Side view
                Defaults to 'z'.
            aspp_in_channels (int, optional): ASPP input channel size. Defaults to 192.
            aspp_out_channels (int, optional): ASPP output channel size. Defaults to 512.
            aspp_rate (list, optional): ASPP dilation rates. Defaults to [6, 12, 18].
            aspp_norm (str, optional): ASPP normalization function. Defaults to 'BN'.
            aspp_dropout (float, optional): ASPP dropout probability. Defaults to 0.1.
            use_depthwise_separable_conv (bool, optional): ASPP enable depthwise convolution. Defaults to False.
            num_clusters (int, optional): NetVLAD number of clusters. Defaults to 64.
            encoder_dim (int, optional): NetVLAD local feature extractor dimenstion. Defaults to 512.
            vladv2 (bool, optional): Set NetVLAD version. Defaults to False.
            output_dim (_type_, optional): NetVLAD output dimenstion. Defaults to None.
            gating (bool, optional): NetVLAD set to enable context gating. Defaults to False.
        """
        super(SecondAsppNetvladV2, self).__init__()
        self.vfe = HardVFE(
            in_channels=3,
            feat_channels=vfe_feat_channels,
            with_voxel_center=True,
            voxel_size=voxel_size,
            point_cloud_range=pcd_range
        )
        self.middle = SparseMiddleExtractor(
            in_channels=middle_spconv_in_channels,
            sparse_shape=grid_zyx,
            base_channels=middle_base_channels,
            output_channels=middle_spconv_out_channels,
            encoder_channels=middle_encoder_channels,
            encoder_paddings=middle_encoder_paddings,
            downsample_dimension=middle_downsample_dimension,
            encoder_downsampling_other_dim=middle_encoder_downsampling_other_dim
        )
        # ASPP
        self.aspp = ASPP(aspp_in_channels,
                         aspp_out_channels,
                         aspp_rate,
                         norm=aspp_norm,
                         activation=nn.ReLU(),
                         dropout=aspp_dropout,
                         use_depthwise_separable_conv=use_depthwise_separable_conv)
        # NetVLAD
        self.netvlad = NetVLAD(
            num_clusters=num_clusters, dim=encoder_dim, output_dim=output_dim, gating=gating)

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
        aspp_output = self.aspp(middle_output)
        output = self.netvlad(aspp_output)
        return output

class SecondAsppGeM(nn.Module):
    """ SECOND + ASPP + GeM
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
                 middle_downsample_dimension='z',
                 middle_encoder_downsampling_other_dim=[
                     False, False, False, False],
                 aspp_in_channels=192,
                 aspp_out_channels=512,
                 aspp_rate=[6, 12, 18],
                 aspp_norm='BN',
                 aspp_dropout=0.1,
                 use_depthwise_separable_conv=False,
                 p: float = 3, eps: float = 1e-6, normalize: bool = True, dense_output_dim: Tuple[int] = None,):
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
            middle_downsample_dimension (str, optional): Downsampling dimension.
                z: Bird's eys view (From the top)
                x: Front view
                y: Side view
                Defaults to 'z'.
            aspp_in_channels (int, optional): ASPP input channel size. Defaults to 192.
            aspp_out_channels (int, optional): ASPP output channel size. Defaults to 512.
            aspp_rate (list, optional): ASPP dilation rates. Defaults to [6, 12, 18].
            aspp_norm (str, optional): ASPP normalization function. Defaults to 'BN'.
            aspp_dropout (float, optional): ASPP dropout probability. Defaults to 0.1.
            use_depthwise_separable_conv (bool, optional): ASPP enable depthwise convolution. Defaults to False.
            p (float, optional): GeM parameter. Defaults to 3.
            eps (float, optional): GeM parameter. Defaults to 1e-6.
            normalize (bool, optional): GeM parameter. Defaults to True.
            dense_output_dim (Tuple[int], optional): GeM parameter. Defaults to None.
        """
        super(SecondAsppGeM, self).__init__()
        self.vfe = HardVFE(
            in_channels=3,
            feat_channels=vfe_feat_channels,
            with_voxel_center=True,
            voxel_size=voxel_size,
            point_cloud_range=pcd_range
        )
        self.middle = SparseMiddleExtractor(
            in_channels=middle_spconv_in_channels,
            sparse_shape=grid_zyx,
            base_channels=middle_base_channels,
            output_channels=middle_spconv_out_channels,
            encoder_channels=middle_encoder_channels,
            encoder_paddings=middle_encoder_paddings,
            downsample_dimension=middle_downsample_dimension,
            encoder_downsampling_other_dim=middle_encoder_downsampling_other_dim
        )
        # ASPP
        self.aspp = ASPP(aspp_in_channels,
                         aspp_out_channels,
                         aspp_rate,
                         norm=aspp_norm,
                         activation=nn.ReLU(),
                         dropout=aspp_dropout,
                         use_depthwise_separable_conv=use_depthwise_separable_conv)
        # NetVLAD
        self.gem = GeM(p=p, eps=eps, normalize=normalize,
                            dense_output_dim=dense_output_dim)

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
        aspp_output = self.aspp(middle_output)
        output = self.gem(aspp_output)
        return output