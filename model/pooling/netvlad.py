import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
import math

# Refer to https://github.com/Nanne/pytorch-NetVlad/blob/master/netvlad.py for the pytorch version
# Refer to https://github.com/cattaneod/PointNetVlad-Pytorch/blob/master/models/PointNetVlad.py for the Gating Context
# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py for the matlab version


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False, output_dim=None, gating=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 1.0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(
            dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()
        self.output_dim = output_dim
        self.gating = gating
        if self.output_dim is not None:
            self.hidden1_weights = nn.Parameter(
                torch.randn(self.num_clusters * self.dim, self.output_dim) * 1 / math.sqrt(self.dim))
            self.bn2 = nn.BatchNorm1d(self.output_dim)

            if self.gating:
                self.context_gating = GatingContext(
                    self.output_dim)

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def init_params(self, clsts, traindescs):
        # TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) /
                          np.mean(dots[0, :] - dots[1, :])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(
                self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1)  # TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[0])
            del knn
            self.alpha = (-np.log(0.01) /
                          np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )
        # self.init = True

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C],
                           dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                self.centroids[C:C+1, :].expand(
                    x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C+1, :].unsqueeze(2)
            vlad[:, C:C+1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        # if self.init:
        #     print("Netvlad init")
        if self.output_dim is not None:
            vlad = torch.matmul(vlad, self.hidden1_weights)
            vlad = self.bn2(vlad)
            if self.gating:
                vlad = self.context_gating(vlad)
        return vlad


class NetVLADVoxel(nn.Module):
    """NetVLADVoxel layer implementation"""

    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False, output_dim=None):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLADVoxel, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 1.0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        # self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.fc = nn.Linear(dim, num_clusters)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        # self._init_params()
        self.output_dim = output_dim
        # self.gating = gating
        if self.output_dim is not None:
            self.hidden1_weights = nn.Parameter(
                torch.randn(self.num_clusters * self.dim, self.output_dim) * 1 / math.sqrt(self.dim))
            self.bn2 = nn.BatchNorm1d(self.output_dim)

    def forward(self, x):
        # voxel feature x has shape [N, C] and batch_indices [N, 1] (batch_id, )
        batch_size = batch_indices.max() + 1

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        for i in range(batch_size):
            single_mask = (batch_indices == i)
            batch_x = x[single_mask, :]
            N, C = batch_x.shape

        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C],
                           dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                self.centroids[C:C+1, :].expand(
                    x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C+1, :].unsqueeze(2)
            vlad[:, C:C+1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        # if self.init:
        #     print("Netvlad init")
        if self.output_dim is not None:
            vlad = torch.matmul(vlad, self.hidden1_weights)
            vlad = self.bn2(vlad)
            if self.gating:
                vlad = self.context_gating(vlad)
        return vlad


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
