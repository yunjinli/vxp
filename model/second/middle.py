#
# Created on May 23 2023
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

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmengine.runner import amp

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential

TwoTupleIntType = Tuple[Tuple[int]]


class SparseLocalDescriptor(nn.Module):
    """Compute sparse local descriptor 
    """

    def __init__(
            self,
            in_channels: int,
            sparse_shape: List[int],
            order: Optional[Tuple[str]] = ('conv', 'norm', 'act'),
            norm_cfg: Optional[dict] = dict(
                type='BN1d', eps=1e-3, momentum=0.01),
            base_channels: Optional[int] = 16,
            output_channels: Optional[int] = 128,
            encoder_channels: Optional[TwoTupleIntType] = ((16, ), (32, 32,
                                                                    32),
                                                           (64, 64,
                                                            64), (64, 64, 64)),
            encoder_paddings: Optional[TwoTupleIntType] = ((1, ), (1, 1, 1),
                                                           (1, 1, 1),
                                                           ((0, 1, 1), 1, 1)),
            block_type: Optional[str] = 'conv_module',
            encoder_downsampling_other_dim: Optional[List[bool]] = [
                False, False, False, False]
    ):
        """Constructor

        Args:
            in_channels (int): The number of input channels.
            sparse_shape (List[int]):  The sparse shape of input tensor.
            order (Optional[Tuple[str]], optional): Order of conv module.. Defaults to ('conv', 'norm', 'act').
            norm_cfg (Optional[dict], optional): Config of normalization layer. Defaults to dict( type='BN1d', eps=1e-3, momentum=0.01).
            base_channels (Optional[int], optional): Out channels for conv_input layer. Defaults to 16.
            output_channels (Optional[int], optional): Out channels for conv_out layer. Defaults to 128.
            encoder_channels (Optional[TwoTupleIntType], optional): Convolutional channels of each encode block. 
                Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
            encoder_paddings (Optional[TwoTupleIntType], optional): Paddings of each encode block. 
                Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
            block_type (Optional[str], optional): Type of the block to use. Defaults to 'conv_module'.
            encoder_downsampling_other_dim (Optional[List[bool]], optional): Whether to perform downsampling in the given layers. Defaults to [False, False, False, False]
        """
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.return_feature_maps = False
        self.encoder_downsampling_other_dim = encoder_downsampling_other_dim
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=3,
            stride=2,
            norm_cfg=norm_cfg,
            padding=1,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')

    @amp.autocast(enabled=False)
    def forward(self, voxel_features: Tensor, coors: Tensor,
                batch_size: int) -> Union[Tensor, Tuple[Tensor, list]]:
        """Forward of SparseMiddleExtractor.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list]: Return spatial features 
            include:
            - spatial_features (torch.Tensor): Spatial features are out from
                the last layer.
            - encode_features (List[SparseConvTensor], optional): Middle layer
                output features. When self.return_feature_maps is True, the
                module returns middle features.

        """
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        if self.return_feature_maps:
            encode_features = []
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)
                encode_features.append(x)
        else:
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)

        out = self.conv_out(x)

        if self.return_feature_maps:
            return out, encode_features
        else:
            return out

    def make_encoder_layers(
        self,
        make_block: nn.Module,
        norm_cfg: Dict,
        in_channels: int,
        block_type: Optional[str] = 'conv_module',
        conv_cfg: Optional[dict] = dict(type='SubMConv3d')
    ) -> int:
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    # If downsampling flag is set, downsampling is applied to all dimensions
                    if not self.encoder_downsampling_other_dim[i]:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                norm_cfg=norm_cfg,
                                stride=1,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels

    def return_all_feature_maps(self):
        """Set to return all intermidiate feature maps
        """
        self.return_feature_maps = True


class SparseMiddleExtractor(nn.Module):
    """Sparse encoder for SECOND.
    """

    def __init__(
            self,
            in_channels: int,
            sparse_shape: List[int],
            order: Optional[Tuple[str]] = ('conv', 'norm', 'act'),
            norm_cfg: Optional[dict] = dict(
                type='BN1d', eps=1e-3, momentum=0.01),
            base_channels: Optional[int] = 16,
            output_channels: Optional[int] = 128,
            encoder_channels: Optional[TwoTupleIntType] = ((16, ), (32, 32,
                                                                    32),
                                                           (64, 64,
                                                            64), (64, 64, 64)),
            encoder_paddings: Optional[TwoTupleIntType] = ((1, ), (1, 1, 1),
                                                           (1, 1, 1),
                                                           ((0, 1, 1), 1, 1)),
            block_type: Optional[str] = 'conv_module',
            downsample_dimension: Optional[str] = 'z',
            encoder_downsampling_other_dim: Optional[List[bool]] = [
                False, False, False, False]
    ):
        """Constructor

        Args:
            in_channels (int): The number of input channels.
            sparse_shape (List[int]):  The sparse shape of input tensor.
            order (Optional[Tuple[str]], optional): Order of conv module.. Defaults to ('conv', 'norm', 'act').
            norm_cfg (Optional[dict], optional): Config of normalization layer. Defaults to dict( type='BN1d', eps=1e-3, momentum=0.01).
            base_channels (Optional[int], optional): Out channels for conv_input layer. Defaults to 16.
            output_channels (Optional[int], optional): Out channels for conv_out layer. Defaults to 128.
            encoder_channels (Optional[TwoTupleIntType], optional): Convolutional channels of each encode block. 
                Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
            encoder_paddings (Optional[TwoTupleIntType], optional): Paddings of each encode block. 
                Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
            block_type (Optional[str], optional): Type of the block to use. Defaults to 'conv_module'.
            downsample_dimension (Optional[str], optional): Downsampling dimension.
                z: Bird's eyes view (BEV) (From the top)
                x: Front view (FV)
                y: Side view (SV)
                Defaults to 'z'.
            encoder_downsampling_other_dim (Optional[List[bool]], optional): If downsampling (2x) is performed in the encoder block.
                Defaults to [False, False, False, False]   
        """
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.downsample_dimension = downsample_dimension
        self.encoder_downsampling_other_dim = encoder_downsampling_other_dim
        self.return_feature_maps = False
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}
        assert downsample_dimension in ['x', 'y', 'z']

        if downsample_dimension == 'z':
            self.downsample_kernel_size = (3, 1, 1)
            self.downsample_stride = (2, 1, 1)
        elif downsample_dimension == 'x':
            self.downsample_kernel_size = (1, 1, 3)
            self.downsample_stride = (1, 1, 2)
        elif downsample_dimension == 'y':
            self.downsample_kernel_size = (1, 3, 1)
            self.downsample_stride = (1, 2, 1)

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=self.downsample_kernel_size,
            stride=self.downsample_stride,
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')

    @amp.autocast(enabled=False)
    def forward(self, voxel_features: Tensor, coors: Tensor,
                batch_size: int) -> Union[Tensor, Tuple[Tensor, list]]:
        """Forward of SparseMiddleExtractor.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list]: Return spatial features 
            include:
            - spatial_features (torch.Tensor): Spatial features are out from
                the last layer.
            - encode_features (List[SparseConvTensor], optional): Middle layer
                output features. When self.return_feature_maps is True, the
                module returns middle features.

        """
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        if self.return_feature_maps:
            encode_features = []
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)
                encode_features.append(x)
        else:
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        # out = self.conv_out(encode_features[-1])
        out = self.conv_out(x)
        spatial_features = out.dense()

        if self.downsample_dimension == 'z':
            N, C, D, H, W = spatial_features.shape
            spatial_features = spatial_features.view(N, C * D, H, W)
        elif self.downsample_dimension == 'x':
            # print(f"downampling axis: {self.downsample_dimension}. Original shape: {spatial_features.shape}")
            spatial_features = spatial_features.swapaxes(2, 4)
            N, C, D, H, W = spatial_features.shape
            spatial_features = spatial_features.reshape((N, C * D, H, W))
        elif self.downsample_dimension == 'y':
            spatial_features = spatial_features.swapaxes(2, 3)
            N, C, D, H, W = spatial_features.shape
            spatial_features = spatial_features.reshape((N, C * D, H, W))

        if self.return_feature_maps:
            return spatial_features, encode_features
        else:
            return spatial_features

    def make_encoder_layers(
        self,
        make_block: nn.Module,
        norm_cfg: Dict,
        in_channels: int,
        block_type: Optional[str] = 'conv_module',
        conv_cfg: Optional[dict] = dict(type='SubMConv3d')
    ) -> int:
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    # If downsampling flag is set, downsampling is applied to all dimensions
                    if not self.encoder_downsampling_other_dim[i]:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                norm_cfg=norm_cfg,
                                stride=self.downsample_stride,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=1,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels

    def return_all_feature_maps(self):
        """Set to return all intermidiate feature maps
        """
        self.return_feature_maps = True
