#
# Created on Nov 7 2023
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
from torchvision import transforms


class Copy(object):
    """Copy one-channel data to multiple channels
    """

    def __init__(self, channels: int):
        """Constructor

        Args:
            channels (int): Number of desired channels number
        """
        self.channels = channels
        self.transform = transforms.Compose([
            transforms.Lambda(
                lambda x: x.repeat(self.channels, 1, 1)),
        ])

    def __call__(self, x: torch.tensor) -> torch.tensor:
        """Pass

        Args:
            x (torch.tensor): input tensor

        Returns:
            torch.tensor: output tensor
        """
        return self.transform(x)

    def __repr__(self):
        return self.__class__.__name__ + '(channels={0})'.format(self.channels)
