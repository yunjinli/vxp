#
# Created on Wed Aug 23 2023
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


class VXIGeM(nn.Module):
    """VXI-GeM module
    """

    def __init__(self,
                 fx: float = 964.828979 / 1280, 
                 fy: float = 964.828979 / 960, 
                 cx: float = 643.788025 / 1280, 
                 cy: float = 484.407990 / 960, 
                 width: int = 28, 
                 height: int = 28, 
                 up_voxel_size: List = [0.2, 0.2, 0.2], 
                 v2i_pcd_range: List = [0, -25, -25, 50, 25, 25], 
                 device: str = 'cuda',
                 R: List[List] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
                 p: float = 3, eps: float = 1e-6, normalize: bool = True, dense_output_dim: Tuple[int] = None,
                 **kwargs):
        """Constructor

        Args:
            width (int, optional): Width of the image. Defaults to 28.
            height (int, optional): Height of the image. Defaults to 28.
            up_voxel_size (List, optional): upsampling voxel size. Defaults to [0.2, 0.2, 0.2].
            v2i_pcd_range (List, optional): pointcloud range. Defaults to [0, -25, -25, 50, 25, 25].
            device (str, optional): cuda or cpu. Defaults to 'cuda'.
            p (float, optional): GeM parameter. Defaults to 3.
            eps (float, optional): GeM parameter. Defaults to 1e-6.
            normalize (bool, optional): GeM parameter. Defaults to True.
            dense_output_dim (Tuple[int], optional): GeM parameter. Defaults to None.
        """
        super(VXIGeM, self).__init__()
        self.voxel_feat_ext = VoxelLocalFeatureExtractor(**kwargs)
        self.gem = GeMVoxel(p=p, eps=eps, normalize=normalize,
                            dense_output_dim=dense_output_dim)
        self.proj = V2PProjection(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            h=height,
            w=width,
            voxel_size=up_voxel_size,
            pcd_range=v2i_pcd_range,
            R=R,
            device=device
        )
        self.device = device

    def forward(self, features: torch.tensor, num_points: torch.tensor, coors: torch.tensor, batch_size: torch.tensor) -> torch.tensor:
        """Forward pass of the VXI-GeM module

        Args:
            features (torch.tensor): Voxel local features
            num_points (torch.tensor): Number of points
            coors (torch.tensor): Voxel grid coordinate
            batch_size (torch.tensor): Batch size 

        Returns:
            torch.tensor: Output global descriptor
        """
        submap_embs = self.voxel_feat_ext(
            features, num_points, coors, batch_size)
        proj_uv = self.proj(submap_embs.indices.detach().double())
        ## include inverse depth
        proj_desc = submap_embs.features[proj_uv[:, 4].int()] * proj_uv[:, 3].reshape(proj_uv.shape[0], 1).float()
        ## Ignore inverse depth
        # proj_desc = submap_embs.features[proj_uv[:, 4].int()]
        
        return self.gem(proj_desc, proj_uv[:, 0].int())
