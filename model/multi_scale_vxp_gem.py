#
# Created on Sun Oct 22 2023
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
from torch import nn
import torch
from typing import Tuple, List
from .layers.projection import V2PProjection
from .pooling.GeM import GeMVoxel
from .voxel_local_feature_extractor import VoxelLocalFeatureExtractor


class MultiScaleVXP(nn.Module):
    """MultiScaleVXP module
    """

    def __init__(self,
                 widths: List[int] = [28],
                 heights: List[int] = [28],
                 multi_voxel_sizes: List[List] = [[0.2, 0.2, 0.2]],
                 v2i_pcd_range: List = [0, -25, -25, 50, 25, 25],
                 device: str = 'cuda',
                 **kwargs):
        """Constrcutor

        Args:
            widths (List[int], optional): Width of the image. Defaults to [28].
            heights (List[int], optional): Height of the image. Defaults to [28].
            multi_voxel_sizes (List[List], optional): Different upsampling voxel size for different scales. Defaults to [[0.2, 0.2, 0.2]].
            v2i_pcd_range (List, optional): Pointcloud range. Defaults to [0, -25, -25, 50, 25, 25].
            device (str, optional): cpu or cuda. Defaults to 'cuda'.
        """
        super(MultiScaleVXP, self).__init__()
        assert len(multi_voxel_sizes) == 3
        assert len(widths) == 3
        assert len(heights) == 3
        self.voxel_feat_ext = VoxelLocalFeatureExtractor(**kwargs)
        self.proj_s1 = V2PProjection(
            fx=964.828979 / 1280 * widths[0],
            fy=964.828979 / 960 * heights[0],
            cx=643.788025 / 1280 * widths[0],
            cy=484.407990 / 960 * heights[0],
            h=heights[0],
            w=widths[0],
            voxel_size=multi_voxel_sizes[0],
            pcd_range=v2i_pcd_range,
            device=device
        )
        self.proj_s2 = V2PProjection(
            fx=964.828979 / 1280 * widths[1],
            fy=964.828979 / 960 * heights[1],
            cx=643.788025 / 1280 * widths[1],
            cy=484.407990 / 960 * heights[1],
            h=heights[1],
            w=widths[1],
            voxel_size=multi_voxel_sizes[1],
            pcd_range=v2i_pcd_range,
            device=device
        )
        self.proj_s3 = V2PProjection(
            fx=964.828979 / 1280 * widths[2],
            fy=964.828979 / 960 * heights[2],
            cx=643.788025 / 1280 * widths[2],
            cy=484.407990 / 960 * heights[2],
            h=heights[2],
            w=widths[2],
            voxel_size=multi_voxel_sizes[2],
            pcd_range=v2i_pcd_range,
            device=device
        )
        self.device = device

    def init(self):
        """Initialization of the module
        """
        self.voxel_feat_ext.return_vfe_output = True
        self.voxel_feat_ext.middle.return_all_feature_maps()

    def forward(self, features: torch.tensor, num_points: torch.tensor, coors: torch.tensor, batch_size: torch.tensor):
        vfe_embs, (final_sparse_voxel_embs, encoder_features) = self.voxel_feat_ext(
            features, num_points, coors, batch_size)
        proj_uv_s1 = self.proj_s1(coors.detach().double())
        proj_uv_s2 = self.proj_s2(
            encoder_features[-1].indices.detach().double())
        proj_uv_s3 = self.proj_s3(
            final_sparse_voxel_embs.indices.detach().double())

        proj_desc_s1 = vfe_embs[proj_uv_s1[:, 4].int()]
        proj_desc_s2 = encoder_features[-1].features[proj_uv_s2[:, 4].int()]
        proj_desc_s3 = final_sparse_voxel_embs.features[proj_uv_s3[:, 4].int()]

        return proj_desc_s1, proj_desc_s2, proj_desc_s3, proj_uv_s1, proj_uv_s2, proj_uv_s3


# class MultiScaleVXPGeMV2(nn.Module):
#     def __init__(self,
#                  **kwargs):
#         super(MultiScaleVXPGeMV2, self).__init__()
#         self.voxel_feat_ext = MultiScaleVXP(**kwargs)
#         self.device = device

#     def init(self):
#         self.voxel_feat_ext.voxel_feat_ext.return_vfe_output = True
#         self.voxel_feat_ext.voxel_feat_ext.middle.return_all_feature_maps()

#     def forward(self, features: torch.tensor, num_points: torch.tensor, coors: torch.tensor, batch_size: torch.tensor):
#         vfe_embs, (final_sparse_voxel_embs, encoder_features) = self.voxel_feat_ext(
#             features, num_points, coors, batch_size)
#         proj_uv_s1 = self.proj_s1(coors.detach().double())
#         proj_uv_s2 = self.proj_s2(
#             encoder_features[-1].indices.detach().double())
#         proj_uv_s3 = self.proj_s3(
#             final_sparse_voxel_embs.indices.detach().double())

#         proj_desc_s1 = vfe_embs[proj_uv_s1[:, 4].int()]
#         proj_desc_s2 = encoder_features[-1].features[proj_uv_s2[:, 4].int()]
#         proj_desc_s3 = final_sparse_voxel_embs.features[proj_uv_s3[:, 4].int()]

#         return proj_desc_s1, proj_desc_s2, proj_desc_s3, proj_uv_s1, proj_uv_s2, proj_uv_s3


class MultiScaleVXPGeM(nn.Module):
    """MultiScaleVXPGeM module
    """

    def __init__(self,
                 widths: List[int] = [28],
                 heights: List[int] = [28],
                 multi_voxel_sizes: List[List] = [[0.2, 0.2, 0.2]],
                 v2i_pcd_range: List = [0, -25, -25, 50, 25, 25],
                 device: str = 'cuda',
                 p: float = 3,
                 eps: float = 1e-6,
                 normalize: bool = True,
                 dense_output_dims: List[Tuple[int]] = None,
                 **kwargs):
        """Constructor

        Args:
            widths (List[int], optional): Widths of the image at different scales. Defaults to [28].
            heights (List[int], optional): Heights of the image at different scales. Defaults to [28].
            multi_voxel_sizes (List[List], optional): Upsampled voxel size at different scales. Defaults to [[0.2, 0.2, 0.2]].
            v2i_pcd_range (List, optional): Pointcloud range. Defaults to [0, -25, -25, 50, 25, 25].
            device (str, optional): cpu or cuda. Defaults to 'cuda'.
            p (float, optional): GeM parameter. Defaults to 3.
            eps (float, optional): GeM parameter. Defaults to 1e-6.
            normalize (bool, optional): GeM parameter. Defaults to True.
            dense_output_dims (List[Tuple[int]], optional): GeM parameter. Defaults to None.
        """
        super(MultiScaleVXPGeM, self).__init__()
        assert len(multi_voxel_sizes) == 3
        assert len(dense_output_dims) == 3
        assert len(widths) == 3
        assert len(heights) == 3
        self.voxel_feat_ext = VoxelLocalFeatureExtractor(**kwargs)

        self.gem_s1 = GeMVoxel(p=p, eps=eps, normalize=normalize,
                               dense_output_dim=dense_output_dims[0])
        self.gem_s2 = GeMVoxel(p=p, eps=eps, normalize=normalize,
                               dense_output_dim=dense_output_dims[1])
        self.gem_s3 = GeMVoxel(p=p, eps=eps, normalize=normalize,
                               dense_output_dim=dense_output_dims[2])

        self.proj_s1 = V2PProjection(
            fx=964.828979 / 1280 * widths[0],
            fy=964.828979 / 960 * heights[0],
            cx=643.788025 / 1280 * widths[0],
            cy=484.407990 / 960 * heights[0],
            h=heights[0],
            w=widths[0],
            voxel_size=multi_voxel_sizes[0],
            pcd_range=v2i_pcd_range,
            device=device
        )
        self.proj_s2 = V2PProjection(
            fx=964.828979 / 1280 * widths[1],
            fy=964.828979 / 960 * heights[1],
            cx=643.788025 / 1280 * widths[1],
            cy=484.407990 / 960 * heights[1],
            h=heights[1],
            w=widths[1],
            voxel_size=multi_voxel_sizes[1],
            pcd_range=v2i_pcd_range,
            device=device
        )
        self.proj_s3 = V2PProjection(
            fx=964.828979 / 1280 * widths[2],
            fy=964.828979 / 960 * heights[2],
            cx=643.788025 / 1280 * widths[2],
            cy=484.407990 / 960 * heights[2],
            h=heights[2],
            w=widths[2],
            voxel_size=multi_voxel_sizes[2],
            pcd_range=v2i_pcd_range,
            device=device
        )
        self.device = device

    def init(self):
        """Initialization of the module
        """
        self.voxel_feat_ext.return_vfe_output = True
        self.voxel_feat_ext.middle.return_all_feature_maps()

    def forward(self, features: torch.tensor, num_points: torch.tensor, coors: torch.tensor, batch_size: torch.tensor) -> torch.tensor:
        """Forward pass

        Args:
            features (torch.tensor): Voxel local features
            num_points (torch.tensor): Number of points per voxel
            coors (torch.tensor): Voxel grid coordinates
            batch_size (torch.tensor): Batch size

        Returns:
            torch.tensor: Output global descriptor
        """
        vfe_embs, (final_sparse_voxel_embs, encoder_features) = self.voxel_feat_ext(
            features, num_points, coors, batch_size)
        proj_uv_s1 = self.proj_s1(coors.detach().double())
        proj_uv_s2 = self.proj_s2(
            encoder_features[-1].indices.detach().double())
        proj_uv_s3 = self.proj_s3(
            final_sparse_voxel_embs.indices.detach().double())

        proj_desc_s1 = vfe_embs[proj_uv_s1[:, 4].int()]
        proj_desc_s2 = encoder_features[-1].features[proj_uv_s2[:, 4].int()]
        proj_desc_s3 = final_sparse_voxel_embs.features[proj_uv_s3[:, 4].int()]

        return self.gem_s1(proj_desc_s1, proj_uv_s1[:, 0].int()) + self.gem_s2(proj_desc_s2, proj_uv_s2[:, 0].int()) + self.gem_s3(proj_desc_s3, proj_uv_s3[:, 0].int())
