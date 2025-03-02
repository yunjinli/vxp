## Code from https://github.com/alexjunholee/LC2_crossmatching/blob/main/lclc/models.py

import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_data):
        return F.normalize(input_data, p=2, dim=self.dim)


class DualEncoder(nn.Module):
    def __init__(self, arch, enc_dim):
        super(DualEncoder, self).__init__()
        backbone_arch = getattr(
            models, arch)
        encoder_d = backbone_arch(
            weights=None)
        layers_d = list(encoder_d.features.children())[:-2]

        self.encoder_d = nn.Sequential(*layers_d)

        encoder_r = backbone_arch(
            weights=None)
        layers_r = list(encoder_r.features.children())[:-2]

        self.encoder_r = nn.Sequential(*layers_r)

        self.enc_dim = enc_dim

    def forward(self, disp, rimg):
        emb_d = self.encoder_d(disp)
        emb_r = self.encoder_r(rimg)

        return emb_d, emb_r

class dual_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_d = models.vgg16(weights=None)
        encoder_r = models.vgg16(weights=None)
        layers_d = list(encoder_d.features.children())[:-2]
        layers_r = list(encoder_r.features.children())[:-2]
        # only train conv5_1, conv5_2, and conv5_3 (leave rest same as Imagenet trained weights)
        for layer in layers_d[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        for layer in layers_r[:-5]:
            for p in layer.parameters():
                p.requires_grad = False

        self.encoder_d = nn.Sequential(*layers_d)
        self.encoder_r = nn.Sequential(*layers_r)
        self.enc_dim = 512

    def forward(self, x, x_isrange):
        idx_disp = ~x_isrange
        out_disp = self.encoder_d(x)
        out_disp = idx_disp[:, None, None, None] * out_disp

        idx_range = x_isrange
        out_range = self.encoder_r(x)
        out_range = idx_range[:, None, None, None] * out_range

        out = out_disp + out_range
        return out
    
class lclcNetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False, use_faiss=True):
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
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        # noinspection PyArgumentList
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.use_faiss = use_faiss

    def init_params(self, clsts, traindescs):
        if not self.vladv2:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
            # noinspection PyArgumentList
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            # noinspection PyArgumentList
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            if not self.use_faiss:
                knn = NearestNeighbors(n_jobs=-1)
                knn.fit(traindescs)
                del traindescs
                ds_sq = np.square(knn.kneighbors(clsts, 2)[1])
                del knn
            else:
                index = faiss.IndexFlatL2(traindescs.shape[1])
                # noinspection PyArgumentList
                index.add(traindescs)
                del traindescs
                # noinspection PyArgumentList
                ds_sq = np.square(index.search(clsts, 2)[1])
                del index

            self.alpha = (-np.log(0.01) / np.mean(ds_sq[:, 1] - ds_sq[:, 0])).item()
            # noinspection PyArgumentList
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, ds_sq

            # noinspection PyArgumentList
            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            # noinspection PyArgumentList
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                self.centroids[C:C + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C + 1, :].unsqueeze(2)
            vlad[:, C:C + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad