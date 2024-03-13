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
from typing import Union
import numpy as np


class Crop2FoV(object):
    """Crop the range image to camera's FoV
    """

    def __init__(self, hfov: float, vfov: float):
        """Constructor

        Args:
            hfov (float): Horizonatal field of view of the camera in degree
            vfov (float): Vetical field of view of the camera in degree
        """
        self.hfov = hfov
        self.vfov = vfov

    def __call__(self, x: Union[torch.tensor, np.array]) -> Union[torch.tensor, np.array]:
        """Pass

        Args:
            x (Union[torch.tensor, np.array]): Input range image

        Returns:
            Union[torch.tensor, np.array]: Output cropped range image
        """
        assert len(x.shape) == 2, "The size of the input must be [h, w]"

        if self.vfov is None and self.hfov is not None:
            x = x[:, x.shape[1] // 2 - self.hfov //
                  2: x.shape[1] // 2 + self.hfov // 2]
        elif self.hfov is None and self.vfov is not None:
            x = x[max(0, x.shape[0] - self.vfov):, :]
        else:
            x = x[max(0, x.shape[0] - self.vfov):, x.shape[1] // 2 - self.hfov //
                  2: x.shape[1] // 2 + self.hfov // 2]
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(hfov={0}, vfov={1})'.format(self.hfov, self.vfov)
