#
# Created on Sat Jul 08 2023
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
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class GeM(nn.Module):
    """Generalized Mean Pooling refer to https://amaarora.github.io/posts/2020-08-30-gempool.html
    """

    def __init__(self, p: float = 3, eps: float = 1e-6, normalize: bool = True, dense_output_dim: Tuple[int] = None, context_gating: int = None):
        """Constructor

        Args:
            p (float, optional): Initial norm. Defaults to 3.
            eps (float, optional): Minumum threshold. Defaults to 1e-6.
            normalize (bool, optional): Normalize the output. Defaults to True.
            dense_output_dim (tuple(int, int), optional): Dense layer to reduce the embedding dimension. Defaults to None.
            context_gating (int, optional): If to use context gating layer at the end. Defaults to None.
        """
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.normalize = normalize

        if dense_output_dim is not None:
            self.dense_head = nn.Linear(
                in_features=dense_output_dim[0], out_features=dense_output_dim[1])
        else:
            self.dense_head = nn.Sequential()

        if context_gating is not None:
            self.context_gating_layer = GatingContext(
                dim=context_gating, add_batch_norm=True)
        else:
            self.context_gating_layer = nn.Sequential()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass for GeM

        Args:
            x (torch.tensor): Input feature maps

        Returns:
            torch.tensor: Output tensor
        """
        if self.normalize:
            return F.normalize(
                self.context_gating_layer(self.dense_head(self.gem(x, p=self.p, eps=self.eps).squeeze(dim=(2, 3)))), p=2, dim=1)
        else:
            return self.context_gating_layer(self.dense_head(self.gem(x, p=self.p, eps=self.eps).squeeze(dim=(2, 3))))

    def gem(self, x: torch.tensor, p: float = 3, eps: float = 1e-6) -> torch.tensor:
        """GeM pass (Directly from the formula in the paper)

        Args:
            x (torch.tensor): Input feature maps
            p (float, optional): Norm. Defaults to 3.
            eps (float, optional): Min. threshold. Defaults to 1e-6.

        Returns:
            torch.tensor: Output tensor
        """
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class GeMVoxel(nn.Module):
    """Generalized Mean Pooling for voxel projections refer to https://amaarora.github.io/posts/2020-08-30-gempool.html
    """

    def __init__(self, p: float = 3, eps: float = 1e-6, normalize: bool = True, dense_output_dim: Tuple[int] = None):
        """Constructor

        Args:
            p (float, optional): Initial norm. Defaults to 3.
            eps (float, optional): Minumum threshold. Defaults to 1e-6.
            normalize (bool, optional): Normalize the output. Defaults to True.
            dense_output_dim (tuple(int, int), optional): Dense layer to reduce the embedding dimension. Defaults to None.
        """
        super(GeMVoxel, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.normalize = normalize

        if dense_output_dim is not None:
            self.dense_head = nn.Linear(
                in_features=dense_output_dim[0], out_features=dense_output_dim[1])
        else:
            self.dense_head = nn.Sequential()

    def forward(self, x: torch.tensor, batch_indices: torch.tensor) -> torch.tensor:
        """Forward pass for GeM

        Args:
            x (torch.tensor): Input feature maps with shape [N', C]
            batch_indices (torch.tensor): Batch indices for the mini-batch with shape [N,]

        Returns:
            torch.tensor: Output tensor
        """
        batch_size = batch_indices.max() + 1
        global_descs = []
        for i in range(batch_size):
            single_mask = (batch_indices == i)
            global_descs.append(
                self.gem(x[single_mask, :], p=self.p, eps=self.eps))
        global_descs = torch.cat(global_descs, dim=0)

        if self.normalize:
            return F.normalize(self.dense_head(global_descs), p=2, dim=1)
        else:
            return self.dense_head(global_descs)

    def gem(self, x: torch.tensor, p: float = 3, eps: float = 1e-6) -> torch.tensor:
        """GeM pass (Directly from the formula in the paper)

        Args:
            x (torch.tensor): Input descriptor for single data with shape [N, C]
            p (float, optional): Norm. Defaults to 3.
            eps (float, optional): Min. threshold. Defaults to 1e-6.

        Returns:
            torch.tensor: Output tensor
        """
        x = x.clamp(min=eps).pow(p)
        x = x.mean(dim=0, keepdim=True).pow(1./p)
        return x


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation
