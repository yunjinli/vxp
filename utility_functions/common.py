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
import yaml
from yaml.loader import FullLoader
import logging
import torchvision

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


def load_setup_file(path: str) -> dict:
    """Return setup file dict from the given yml file path

    Args:
        path (str): yml file path

    Returns:
        dict: setup dict
    """
    with open(path) as f:
        data = yaml.load(f, Loader=FullLoader)
    logger.info(f"Successfully loaded setup file: {path}")
    logger.info(data)
    return data


def save_setup_file(setup: dict, path: str):
    """Save setup file dict to the given yml file path

    Args:
        setup (dict): selected setup file to be saved
        path (str): saved yml file path
    """
    with open(path, 'w') as yaml_file:
        yaml.dump(setup, yaml_file, default_flow_style=False, sort_keys=False)
    logger.info(f"Successfully saved setup file: {path}")


def load_data_augmentation(data_augmentations={}, custom_data_augmentation_modules=None) -> list:
    """Helper function for parsing the data augmentation modules from a dictionary

    Args:
        data_augmentations (dict, optional): Input data augmentation dictionary. Defaults to {}.
        custom_data_augmentation_modules (modules, optional): Custom data augmentation or preprocessing operations. Defaults to None.
    Raises:
        NotImplementedError: The requested data augmentation modules is neither suppoerted in torchvision nor in our custom modules

    Returns:
        list: A list of parsed data augmentation. 
    """
    transforms = []

    for k, v in data_augmentations.items():
        try:
            transform_class = getattr(torchvision.transforms, k)
        except:
            try:
                transform_class = getattr(custom_data_augmentation_modules, k)
            except:
                raise NotImplementedError
        transform = transform_class(**v['parameters'])
        transforms.append(transform)
    logger.info(f"Successfully load data transformations {transforms}")
    return transforms
