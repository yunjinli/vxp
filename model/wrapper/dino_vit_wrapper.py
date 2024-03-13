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


class DinoViTWrapper(nn.Module):
    def __init__(self, dino: str = 'dino_vits8'):
        """Constructor for the DinoViTWrapper model

        Args:
            dino (str, optional): dino type
        """
        super().__init__()
        assert dino in ['dino_vits8', 'dino_vits16']
        self.encoder = torch.hub.load('facebookresearch/dino:main', dino)

    def forward(self, x):
        """Foward pass of the DinoViTWrapper

        Args:
            x (torch.tensor): Input tensor with shape B x C x H x W

        Returns:
            torch.tensor: Output from the model
        """
        x = self.encoder.prepare_tokens(x)
        for blk in self.encoder.blocks:
            x = blk(x)
        _, N, D = x.shape
        x = x[:, 1:, :].swapaxes(
            1, 2).view(-1, D, int(N ** 0.5), int(N ** 0.5))
        return x
