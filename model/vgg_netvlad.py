#
# Created on Mon May 16 2023
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
# The implementation is mostly inspired from https://github.com/Nanne/pytorch-NetVlad

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pooling.netvlad import NetVLAD


class VGGNetVLAD(nn.Module):
    """VGG16 + NetVLAD 
    """

    def __init__(self, num_clusters=64, encoder_dim=512, vladv2=False, fine_tuning=True, use_pretrained_weight=True, pretrained_weight=None, output_dim=None, gating=False):
        """Constructor for the VGGNetVLAD model

        Args:
            num_clusters (int, optional): Number of clusters of the NetVLAD layer. Defaults to 64.
            encoder_dim (int, optional): Dimenstion of the local descriptor from VGG16 . Defaults to 512.
            vladv2 (bool, optional): Flag to set the NetVLAD version . Defaults to False.
            fine_tuning (bool, optional): Flag for Fined-tune the network, 
                                        if set to true, only part of the VGG would be trained. Defaults to True.
            use_pretrained_weight (bool, optional): If to use the pretrained weight. Defaults to True.
            pretrained_weight (string, optional): File path to the pretrained weight file. Defaults to None.
            output_dim (int, optional): shrinked output dimenstion from the NetVLAD layer. Defaults to None.
            gating (bool, optional): Flag for enabling Context Gating layer in the NetVLAD. Defaults to False.
        """
        super().__init__()
        if use_pretrained_weight:
            encoder = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            layers = list(encoder.features.children())[:-2]
            if fine_tuning:
                for l in layers[:-5]:
                    # for l in layers:
                    for p in l.parameters():
                        p.requires_grad = False
            encoder = nn.Sequential(*layers)
            if pretrained_weight is not None:
                encoder = torch.load(pretrained_weight)
                if fine_tuning:
                    for layer in encoder[:-5]:
                        # for layer in encoder:
                        for p in layer.parameters():
                            p.requires_grad = False
            encoder = nn.Sequential(*layers)
        else:
            encoder = models.vgg16()
            layers = list(encoder.features.children())[:-2]
            encoder = nn.Sequential(*layers)

        self.model = nn.Sequential(
            encoder,
            NetVLAD(num_clusters=num_clusters, dim=encoder_dim,
                    vladv2=vladv2, output_dim=output_dim, gating=gating)
        )

    def forward(self, x):
        """Foward pass of the VGGNetVLAD

        Args:
            x (torch.tensor): Input tensor with shape B x C x H x W

        Returns:
            torch.tensor: Output from the model
        """
        output = self.model(x)
        return output
