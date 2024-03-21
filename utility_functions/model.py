#
# Created on June 17 2023
# The MIT License (MIT)
# Copyright (c) 2023 Yun-Jin Li (Jim), Technical University of Munich (TUM)
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

import logging
import torch
from torchinfo import summary
import torch.nn as nn
import torchvision.models
from collections import OrderedDict

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter(
    '[%(levelname)s] [%(name)s] [%(process)d] %(asctime)s: %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


def model_factory(model_collection, model_setup: dict) -> torch.nn.Module:
    """Return the given model from the model collections

    Args:
        model_collection: Collection of the available models
        model_setup (dict): Setup dict specifying parameters of the selected model

    Raises:
        NotImplementedError: The selected model is not implemented

    Returns:
        torch.nn.Module: The selected torch model
    """
    try:
        model_arch = getattr(model_collection, model_setup['arch'])
    except:
        raise NotImplementedError
    model = model_arch(**model_setup['parameters'])
    try:
        logger.info(
            summary(model, input_size=model_setup['input_size'], depth=5))
    except:
        logger.warning("Fail to create torch summary with input size")
        logger.info(summary(model, depth=5))
    return model


def model_factory_v2(model_collection, model_setup: dict) -> torch.nn.Module:
    """Return the given model from the model collections

    Args:
        model_collection: Collection of the available models
        model_setup (dict): Setup dict specifying parameters of the selected model

    Raises:
        NotImplementedError: The selected model is not implemented

    Returns:
        torch.nn.Module: The selected torch model
    """
    model_dict = OrderedDict()

    if model_setup['backbone']['arch'] == 'vgg16':
        backbone_arch = getattr(
            torchvision.models, model_setup['backbone']['arch'])
        backbone = backbone_arch(
            weights=torchvision.models.VGG16_Weights.DEFAULT)
        layers = list(backbone.features.children())[:-2]
        backbone = nn.Sequential(*layers)
        model_dict['backbone'] = backbone
    elif model_setup['backbone']['arch'] == 'resnet18':
        backbone_arch = getattr(
            torchvision.models, model_setup['backbone']['arch'])
        backbone = backbone_arch(
            weights=torchvision.models.ResNet18_Weights.DEFAULT)
        layers = list(backbone.children())[:-2]
        backbone = nn.Sequential(*layers)
        model_dict['backbone'] = backbone
    elif model_setup['backbone']['arch'] == 'resnet34':
        backbone_arch = getattr(
            torchvision.models, model_setup['backbone']['arch'])
        backbone = backbone_arch(
            weights=torchvision.models.ResNet34_Weights.DEFAULT)
        layers = list(backbone.children())[:-2]
        backbone = nn.Sequential(*layers)
        model_dict['backbone'] = backbone
    else:
        # Custom backbone
        backbone_arch = getattr(
            model_collection, model_setup['backbone']['arch'])
        backbone = backbone_arch(**model_setup['backbone']['parameters'])
        ## For one freeze DINO experiment
        # for param in backbone.parameters():
        #     param.requires_grad = False
        model_dict['backbone'] = backbone
    
    pooling_arch = getattr(model_collection, model_setup['pooling']['arch'])
    pooling = pooling_arch(**model_setup['pooling']['parameters'])
    model_dict['pooling'] = pooling

    model = nn.Sequential(model_dict)
    try:
        stat = summary(model, input_size=model_setup['input_size'], depth=5)
        logger.info(stat)
        output_dim = stat.summary_list[-1].output_size[1]
        model_setup.update({"output_dim": output_dim})
        # model_setup['output_dim'] = output_dim
    except:
        logger.warning("Fail to create torch summary with input size")
        logger.info(summary(model, depth=5))

    return model


def load_pretrained_weight(model: torch.nn.Module, model_path: str, device='cpu') -> torch.nn.Module:
    """Load the pretrained weight from the given path

    Args:
        model (torch.nn.Module): The initial torch model
        model_path (str): The .pth weight file path
        device (str, optional): The device being used ('cpu', 'cuda'). Defaults to 'cpu'.

    Returns:
        torch.nn.Module: Updated torch model
    """
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device(device)))

    return model
