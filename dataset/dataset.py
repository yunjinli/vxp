#
# Created on May 14 2023
# The MIT License (MIT)
# Copyright (c) 2023 Yun-Jin Li (Jim), Technical University of Munich (TUM)
# Part of the code refer to https://github.com/jac99/MinkLocMultimodal/blob/master/datasets/oxford.py
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
from torch.utils.data import Dataset, default_collate
from bitarray import bitarray
import pandas as pd
import pickle
from PIL import Image
import numpy as np
import random
import torch
import psutil
import logging
from tqdm import tqdm
import os
# from preprocessing.pointcloud import OnlineRotX
from .preprocessing.data_augmentation.horizontal_mirroring import HorizontalMirroring
# from preprocessing import DataAugmentationHorizontalMirroring
# from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms.functional import hflip
import math
from typing import Union, List
from shapely.geometry import Point, Polygon

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


def make_collate_fn(dataset: Dataset):
    """Construct collate_fn for different dataset type

    Args:
        dataset (Dataset): Used dataset

    Returns:
        collate_fn: collate_fn for ```dataset```
    """
    def collate_fn(data_list) -> tuple:
        """Return data in a mini-batch with different formats

        Args:
            data_list (list): Data of the mini-batch

        Returns:
            tuple: 
                OxfordImageDataset:
                    tuple(imgs, positives_mask, negatives_mask)
                OxfordImageVoxelDataset:
                    tuple(imgs, voxels_out_list, coors_out_list, num_points_per_voxel_out_list)
                InferenceImageVoxelDataset:
                    tuple(imgs, voxels_out_list, coors_out_list, num_points_per_voxel_out_list, indices)
                InferenceImagePointcloudDataset
                    tuple(imgs, pcds, indices)
        """
        if dataset.get_dataset_type() == "OxfordImageDataset":
            # Constructs a batch object
            imgs = [e[0] for e in data_list]
            labels = [e[1] for e in data_list]
            imgs = default_collate(imgs)

            # Compute positives and negatives mask
            # dataset.queries[label]['positives'] is bitarray
            positives_mask = [[dataset.queries[label]['positives'][e]
                               for e in labels] for label in labels]
            negatives_mask = [[dataset.queries[label]['negatives'][e]
                               for e in labels] for label in labels]

            positives_mask = torch.tensor(positives_mask, dtype=torch.bool)
            negatives_mask = torch.tensor(negatives_mask, dtype=torch.bool)

            return imgs, positives_mask, negatives_mask
        elif dataset.get_dataset_type() == "LC2OxfordDatasetPhase2":
            # Constructs a batch object
            disps = [e[0] for e in data_list]
            rimgs = [e[1] for e in data_list]
            labels = [e[2] for e in data_list]
            disps = default_collate(disps)
            rimgs = default_collate(rimgs)

            # Compute positives and negatives mask
            # dataset.queries[label]['positives'] is bitarray
            positives_mask = [[dataset.queries[label]['positives'][e]
                               for e in labels] for label in labels]
            negatives_mask = [[dataset.queries[label]['negatives'][e]
                               for e in labels] for label in labels]

            positives_mask = torch.tensor(positives_mask, dtype=torch.bool)
            negatives_mask = torch.tensor(negatives_mask, dtype=torch.bool)

            return disps, rimgs, positives_mask, negatives_mask
        elif dataset.get_dataset_type() == "LC2OxfordDatasetPhase1":
            # disp_a, disp_p, label1, rimg_a, rimg_p, label2, label3, label4
            disp_as = [e[0] for e in data_list]
            disp_ps = [e[1] for e in data_list]
            rimg_as = [e[3] for e in data_list]
            rimg_ps = [e[4] for e in data_list]

            label1s = [e[2] for e in data_list]
            label2s = [e[5] for e in data_list]
            label3s = [e[6] for e in data_list]
            label4s = [e[7] for e in data_list]

            disp_as = default_collate(disp_as)
            disp_ps = default_collate(disp_ps)
            rimg_as = default_collate(rimg_as)
            rimg_ps = default_collate(rimg_ps)
            label1s = default_collate(label1s)
            label2s = default_collate(label2s)
            label3s = default_collate(label3s)
            label4s = default_collate(label4s)

            return disp_as, disp_ps, rimg_as, rimg_ps, label1s, label2s, label3s, label4s
        elif dataset.get_dataset_type() == "OxfordImageVoxelTripletDataset":
            imgs = [e[0] for e in data_list]
            voxels = [e[1] for e in data_list]
            labels = [e[2] for e in data_list]

            imgs = default_collate(imgs)
            voxels_out_list = []
            num_points_per_voxel_out_list = []
            coors_out_list = []
            for i, voxel in enumerate(voxels):
                voxels_out, coors_out, num_points_per_voxel_out = voxel
                batch_indices = torch.ones((coors_out.shape[0], 1)) * i
                coors_out = torch.cat((batch_indices, coors_out), dim=1)
                voxels_out_list.append(voxels_out)
                num_points_per_voxel_out_list.append(num_points_per_voxel_out)
                coors_out_list.append(coors_out)
            voxels_out_list = torch.cat(voxels_out_list, dim=0)
            num_points_per_voxel_out_list = torch.cat(
                num_points_per_voxel_out_list, dim=0)
            coors_out_list = torch.cat(coors_out_list, dim=0)

            positives_mask = [[dataset.queries[label]['positives'][e]
                               for e in labels] for label in labels]
            negatives_mask = [[dataset.queries[label]['negatives'][e]
                               for e in labels] for label in labels]

            positives_mask = torch.tensor(positives_mask, dtype=torch.bool)
            negatives_mask = torch.tensor(negatives_mask, dtype=torch.bool)

            return imgs, voxels_out_list, coors_out_list, num_points_per_voxel_out_list, positives_mask, negatives_mask

        elif dataset.get_dataset_type() == "OxfordImageVoxelDataset":
            imgs = [e[0] for e in data_list]
            voxels = [e[1] for e in data_list]
            imgs = default_collate(imgs)
            voxels_out_list = []
            num_points_per_voxel_out_list = []
            coors_out_list = []
            for i, voxel in enumerate(voxels):
                voxels_out, coors_out, num_points_per_voxel_out = voxel
                batch_indices = torch.ones((coors_out.shape[0], 1)) * i
                coors_out = torch.cat((batch_indices, coors_out), dim=1)
                voxels_out_list.append(voxels_out)
                num_points_per_voxel_out_list.append(num_points_per_voxel_out)
                coors_out_list.append(coors_out)
            voxels_out_list = torch.cat(voxels_out_list, dim=0)
            num_points_per_voxel_out_list = torch.cat(
                num_points_per_voxel_out_list, dim=0)
            coors_out_list = torch.cat(coors_out_list, dim=0)
            return imgs, voxels_out_list, coors_out_list, num_points_per_voxel_out_list
        elif dataset.get_dataset_type() == "InferenceImageVoxelDataset":
            imgs = [e[0] for e in data_list]
            voxels = [e[1] for e in data_list]
            indices = [e[2] for e in data_list]
            imgs = default_collate(imgs)
            indices = default_collate(indices)

            voxels_out_list = []
            num_points_per_voxel_out_list = []
            coors_out_list = []
            for i, voxel in enumerate(voxels):
                voxels_out, coors_out, num_points_per_voxel_out = voxel
                batch_indices = torch.ones((coors_out.shape[0], 1)) * i
                coors_out = torch.cat((batch_indices, coors_out), dim=1)
                voxels_out_list.append(voxels_out)
                num_points_per_voxel_out_list.append(num_points_per_voxel_out)
                coors_out_list.append(coors_out)
            voxels_out_list = torch.cat(voxels_out_list, dim=0)
            num_points_per_voxel_out_list = torch.cat(
                num_points_per_voxel_out_list, dim=0)
            coors_out_list = torch.cat(coors_out_list, dim=0)

            return imgs, voxels_out_list, coors_out_list, num_points_per_voxel_out_list, indices
        elif dataset.get_dataset_type() == "InferenceImagePointcloudDataset":
            imgs = [e[0] for e in data_list]
            pcds = [e[1] for e in data_list]
            indices = [e[2] for e in data_list]

            imgs = default_collate(imgs)
            indices = default_collate(indices)

            submaps = []

            return imgs, pcds, indices
        elif dataset.get_dataset_type() == "InferenceDispRimgDataset":
            disps = [e[0] for e in data_list]
            rimgs = [e[1] for e in data_list]
            indices = [e[2] for e in data_list]

            disps = default_collate(disps)
            rimgs = default_collate(rimgs)
            indices = default_collate(indices)

            return disps, rimgs, indices

        elif dataset.get_dataset_type() == "LC2OxfordDatasetPhase2Random":
            disp_as = [e[0] for e in data_list]
            disp_ps = [e[1] for e in data_list]
            disp_ns = [e[2] for e in data_list]
            rimg_as = [e[3] for e in data_list]
            rimg_ps = [e[4] for e in data_list]
            rimg_ns = [e[5] for e in data_list]

            disp_as = default_collate(disp_as)
            disp_ps = default_collate(disp_ps)
            disp_ns = default_collate(disp_ns)
            rimg_as = default_collate(rimg_as)
            rimg_ps = default_collate(rimg_ps)
            rimg_ns = default_collate(rimg_ns)

            return disp_as, rimg_ps, rimg_ns, rimg_as, disp_ps, disp_ns

    return collate_fn


class OxfordImageDataset(Dataset):
    """Dataset wrapper for Oxford dataset from PointNetVLAD project.
    (The implementation is inspired by https://github.com/KamilZywanowski/MinkLoc3D-SI/blob/master/datasets/oxford.py)
    """

    def __init__(self, query_filepath: str, transform=None, max_elems=None, use_undistorted=True, data_augmentation=None):
        """Constructor for the OxfordImageDataset

        Args:
            query_filepath (str): Training / Val pickle file path
            transform (torchvision.transforms.Compose, optional): The default transformation for the image data. Defaults to None.
            max_elems (int, optional): Maximum number of data to be processed, for debugging usage. Defaults to None.
            use_undistorted (bool, optional): Flag determining whether to use the undistorted image. Defaults to True.
            data_augmentation (torchvision.transforms.Compose, optional): Data augmentation for images. Defaults to None.
        """
        # transform: transform applied to each element
        self.use_undistorted = use_undistorted
        self.query_filepath = query_filepath
        assert os.path.exists(
            self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.max_elems = max_elems

        cached_query_filepath = self.query_filepath.replace(
            '.pickle', '_cached.pickle')
        # cached_query_filepath = os.path.splitext(self.query_filepath)[0] + '_cached.pickle'
        if not os.path.exists(cached_query_filepath):
            # Pre-process query file
            self.queries = self.preprocess_queries(
                self.query_filepath, cached_query_filepath)
        else:
            logger.info('Loading preprocessed query file: {}...'.format(
                cached_query_filepath))
            with open(cached_query_filepath, 'rb') as handle:
                # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
                self.queries = pickle.load(handle)

        if max_elems is not None:
            filtered_queries = {}
            for ndx in self.queries:
                if ndx >= self.max_elems:
                    break
                filtered_queries[ndx] = {'query_img': self.queries[ndx]['query_img'],
                                         'positives': self.queries[ndx]['positives'][0:max_elems],
                                         'negatives': self.queries[ndx]['negatives'][0:max_elems]}
            self.queries = filtered_queries

        logger.info('{} queries in the dataset'.format(len(self)))

    def preprocess_queries(self, query_filepath: str, cached_query_filepath: str) -> dict:
        """Preprocess the queries pickle file to make it more memory-efficient

        Args:
            query_filepath (str): File path of the original query
            cached_query_filepath (str): Output path of the preprocessed query

        Returns:
            dict: Output preprocessed query dictionary
        """
        logger.info('Loading query file: {}...'.format(query_filepath))
        with open(query_filepath, 'rb') as handle:
            # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
            queries = pickle.load(handle)

        # Convert to bitarray
        for ndx in tqdm(queries):
            queries[ndx]['positives'] = set(queries[ndx]['positives'])
            queries[ndx]['negatives'] = set(queries[ndx]['negatives'])
            pos_mask = [e_ndx in queries[ndx]['positives']
                        for e_ndx in range(len(queries))]
            neg_mask = [e_ndx in queries[ndx]['negatives']
                        for e_ndx in range(len(queries))]
            queries[ndx]['positives'] = bitarray(pos_mask)
            queries[ndx]['negatives'] = bitarray(neg_mask)

        with open(cached_query_filepath, 'wb') as handle:
            pickle.dump(queries, handle)

        return queries

    def __len__(self):
        """Return the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.queries)

    def __getitem__(self, ndx: int):
        """Output the image with corresponding image

        Args:
            ndx (int): Index

        Returns:
            Tuple: Image and index
        """
        # Load img and apply transform
        if self.use_undistorted:
            filename = self.queries[ndx]['query_img']
        else:
            filename = self.queries[ndx]['query_img'].replace(
                '/undistorted', '')
        img = Image.open(filename)
        if self.data_augmentation is not None:
            img = self.data_augmentation(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, ndx

    def get_positives_ndx(self, ndx):
        """Get all the positive indices for a given query

        Args:
            ndx (int): query index

        Returns:
            list: Output list of positive sample index
        """
        # Get list of indexes of similar images
        return self.queries[ndx]['positives'].search(bitarray([True]))

    def get_negatives_ndx(self, ndx):
        """Get all the negative indices for a given query

        Args:
            ndx (int): query index

        Returns:
            list: Output list of negative sample index
        """
        # Get list of indexes of dissimilar images
        return self.queries[ndx]['negatives'].search(bitarray([True]))

    def set_dataset_name(self, name: str):
        """Set the dataset name

        Args:
            name (str): dataset name to be set
        """
        self.name = name

    def get_dataset_type(self) -> str:
        """Return the dataset type

        Returns:
            str: Return "OxfordImageDataset"
        """
        return "OxfordImageDataset"


# class LC2OxfordDatasetPhase1(Dataset):
#     """Dataset wrapper for LC2 Oxford dataset for phase 1 training.
#     """

#     def __init__(self, query_filepath: str, transform_disp=None, transform_rimg=None, data_augmentation_disp=None, data_augmentation_rimg=None, cam_fov=66, cam_range=50, lidar_fov=165, lidar_range=50, sector_step=3, crop_num=3, rebase_dir=None, mode='train'):
#         # transform: transform applied to each element
#         self.query_filepath = query_filepath
#         assert mode in ['train', 'val']
#         assert os.path.exists(
#             self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
#         with open(self.query_filepath, 'rb') as handle:
#             self.queries = pickle.load(handle)
#         self.transform_disp = transform_disp
#         self.data_augmentation_disp = data_augmentation_disp
#         self.transform_rimg = transform_rimg
#         self.data_augmentation_rimg = data_augmentation_rimg
#         self.cam_fov = cam_fov
#         self.cam_range = cam_range
#         self.lidar_fov = lidar_fov
#         self.lidar_range = lidar_range
#         self.sector_step = sector_step
#         self.crop_num = crop_num
#         self.crop_degree = self.lidar_fov / (self.crop_num + 1)
#         self.mode = mode
#         self.rebase_dir = rebase_dir

#         logger.info('{} queries in the dataset'.format(len(self)))

#     def __len__(self):
#         """Return the length of the dataset

#         Returns:
#             int: Length of the dataset
#         """
#         return len(self.queries)

#     def __getitem__(self, idx: int):
#         if self.queries[idx]['positives'].shape[0] == 0:
#             logger.warning(
#                 f"There is no positive samples for data {idx}, load itself instead...")
#             pidx = idx
#         else:
#             pidx = self.__get_positive_idx(self.queries[idx]['positives'])

#         if self.rebase_dir is not None:
#             disp_a_path = self.queries[idx]['query_disp'].replace(
#                 '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)
#             disp_p_path = self.queries[pidx]['query_disp'].replace(
#                 '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)
#             rimg_a_path = self.queries[idx]['query_rimg'].replace(
#                 '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)
#             rimg_p_path = self.queries[pidx]['query_rimg'].replace(
#                 '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)
#         else:
#             disp_a_path = self.queries[idx]['query_disp']
#             disp_p_path = self.queries[pidx]['query_disp']
#             rimg_a_path = self.queries[idx]['query_rimg']
#             rimg_p_path = self.queries[pidx]['query_rimg']

#         disp_a = np.load(disp_a_path)
#         disp_a = disp_a.squeeze()
#         rimg_a = np.load(rimg_a_path).astype(np.float32)
#         # print(f"Anchor range image path: {self.queries[idx]['query_rimg']}")

#         disp_p = np.load(disp_p_path)
#         disp_p = disp_p.squeeze()
#         rimg_p = np.load(rimg_p_path).astype(np.float32)
#         # print(f"Positive range image path: {self.queries[pidx]['query_rimg']}")

#         if self.transform_disp is not None:
#             disp_a = self.transform_disp(disp_a)
#             disp_p = self.transform_disp(disp_p)

#         if self.data_augmentation_disp is not None:
#             disp_a = self.data_augmentation_disp(disp_a)
#             disp_p = self.data_augmentation_disp(disp_p)

#         if self.data_augmentation_rimg is not None:
#             rimg_a = self.data_augmentation_rimg(rimg_a)
#             rimg_p = self.data_augmentation_rimg(rimg_p)

#         xa = self.queries[idx]['x']
#         ya = self.queries[idx]['y']
#         yawa = self.queries[idx]['yaw']

#         # print(f"Anchor Info (x, y, yaw) = ({xa}, {ya}, {yawa})")

#         xp = self.queries[pidx]['x']
#         yp = self.queries[pidx]['y']
#         yawp = self.queries[pidx]['yaw']

#         # print(f"Positive Info (x, y, yaw) = ({xp}, {yp}, {yawp})")

#         # when training, we have 50% chance to crop the range image into sub-range image
#         if self.mode == 'train' and random.random() > 2:
#             rimg_a, yawa_new = self.__crop_rimg(rimg=rimg_a, yaw=yawa)
#             rimg_p, yawp_new = self.__crop_rimg(rimg=rimg_p, yaw=yawp)
#             fov_new = 2 * self.crop_degree
#         else:
#             yawa_new = yawa
#             yawp_new = yawp
#             fov_new = self.lidar_fov

#         if self.transform_rimg is not None:
#             rimg_a = self.transform_rimg(rimg_a)
#             rimg_p = self.transform_rimg(rimg_p)

#         # print(f"Prev anchor yaw before crop: {yawa}")
#         # print(f"Prev positive yaw before crop: {yawp}")

#         # print(f"New anchor yaw after crop: {yawa_new}")
#         # print(f"New positive yaw after crop: {yawp_new}")
#         label1 = self.__compute_iou(t1=self.__sector(x=ya, y=xa, yaw=math.degrees(yawa), fov=self.cam_fov, radius=self.cam_range, steps=self.sector_step),
#                                     t2=self.__sector(x=yp, y=xp, yaw=math.degrees(yawp), fov=self.cam_fov, radius=self.cam_range, steps=self.sector_step))
#         label2 = self.__compute_iou(t1=self.__sector(x=ya, y=xa, yaw=math.degrees(yawa_new), fov=fov_new, radius=self.lidar_range, steps=self.sector_step),
#                                     t2=self.__sector(x=yp, y=xp, yaw=math.degrees(yawp_new), fov=fov_new, radius=self.lidar_range, steps=self.sector_step))
#         label3 = self.__compute_iou(t1=self.__sector(x=ya, y=xa, yaw=math.degrees(yawa), fov=self.cam_fov, radius=self.cam_range, steps=self.sector_step),
#                                     t2=self.__sector(x=yp, y=xp, yaw=math.degrees(yawp_new), fov=fov_new, radius=self.lidar_range, steps=self.sector_step))
#         label4 = self.__compute_iou(t1=self.__sector(x=ya, y=xa, yaw=math.degrees(yawa_new), fov=fov_new, radius=self.lidar_range, steps=self.sector_step),
#                                     t2=self.__sector(x=yp, y=xp, yaw=math.degrees(yawp), fov=self.cam_fov, radius=self.cam_range, steps=self.sector_step))
#         return disp_a, disp_p, label1, rimg_a, rimg_p, label2, label3, label4

#     def __get_positive_idx(self, positives: Union[List, np.array]):
#         return np.random.choice(positives)

#     def __compute_iou(self, t1: Polygon, t2: Polygon) -> float:
#         return t1.intersection(t2).area / (t1.area + t2.area - t1.intersection(t2).area)

#     def __sector(self, x, y, yaw, fov, radius, steps=200):
#         def polar_point(origin_point, angle,  distance):
#             return [origin_point.x + math.sin(math.radians(angle)) * distance, origin_point.y + math.cos(math.radians(angle)) * distance]

#         center = Point(x, y)
#         start_angle = yaw - fov / 2
#         end_angle = yaw + fov / 2

#         if start_angle > end_angle:
#             start_angle = start_angle - 360
#         else:
#             pass
#         step_angle_width = (end_angle-start_angle) / steps
#         sector_width = (end_angle-start_angle)
#         segment_vertices = []

#         segment_vertices.append(polar_point(center, 0, 0))
#         segment_vertices.append(polar_point(center, start_angle, radius))

#         for z in range(1, steps):
#             segment_vertices.append(
#                 (polar_point(center, start_angle + z * step_angle_width, radius)))
#         segment_vertices.append(polar_point(
#             center, start_angle+sector_width, radius))
#         segment_vertices.append(polar_point(center, 0, 0))
#         return Polygon(segment_vertices)

#     def __crop_rimg(self, rimg, yaw):
#         h, w = rimg.shape
#         # crop_degree = self.lidar_fov / (self.crop_num + 1)
#         crop_indices = list(range(self.crop_num))
#         id = random.choice(crop_indices)
#         start_degree = self.crop_degree * id
#         end_degree = start_degree + 2 * self.crop_degree

#         new_yaw = yaw + math.radians(self.lidar_fov / 2) - \
#             math.radians(start_degree + self.crop_degree)
#         new_rimg = rimg[:, w - int(end_degree / self.lidar_fov * w)
#                                    : w - int(start_degree / self.lidar_fov * w)]
#         return new_rimg, new_yaw

#     def get_dataset_type(self) -> str:
#         """Return the dataset type

#         Returns:
#             str: Return "LC2OxfordDatasetPhase1"
#         """
#         return "LC2OxfordDatasetPhase1"
class LC2OxfordDatasetPhase1(Dataset):
    """Dataset wrapper for LC2 Oxford dataset for phase 1 training.
    """

    def __init__(self, query_filepath: str, transform_disp=None, transform_rimg=None, data_augmentation_disp=None, data_augmentation_rimg=None, cam_fov=66, cam_range=50, lidar_fov=165, lidar_range=50, sector_step=3, crop_num=3, rebase_dir=None, mode='train'):
        # transform: transform applied to each element
        self.query_filepath = query_filepath
        assert mode in ['train', 'val']
        assert os.path.exists(
            self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        with open(self.query_filepath, 'rb') as handle:
            self.queries = pickle.load(handle)
        self.transform_disp = transform_disp
        self.data_augmentation_disp = data_augmentation_disp
        self.transform_rimg = transform_rimg
        self.data_augmentation_rimg = data_augmentation_rimg
        self.cam_fov = cam_fov
        self.cam_range = cam_range
        self.lidar_fov = lidar_fov
        self.lidar_range = lidar_range
        self.sector_step = sector_step
        self.crop_num = crop_num
        self.crop_degree = self.lidar_fov / (self.crop_num + 1)
        self.mode = mode
        self.rebase_dir = rebase_dir
        self.max_pos_dist = 10
        # self.max_yaw_ang_rad = 0.5 + math.radians(self.lidar_fov / 2) - \
        #     math.radians(self.crop_degree)
        self.max_yaw_ang_rad = 0.5 + math.radians(self.lidar_fov / 2)
        # self.max_yaw_ang_rad = 0.5
        logger.info('{} queries in the dataset'.format(len(self)))

    def __len__(self):
        """Return the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.queries)

    def __getitem__(self, idx: int):
        if self.queries[idx]['positives'].shape[0] == 0:
            logger.warning(
                f"There is no positive samples for data {idx}, load itself instead...")
            pidx = idx
        else:
            pidx = self.__get_positive_idx(self.queries[idx]['positives'])

        if self.rebase_dir is not None:
            disp_a_path = self.queries[idx]['query_disp'].replace(
                '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)
            disp_p_path = self.queries[pidx]['query_disp'].replace(
                '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)
            rimg_a_path = self.queries[idx]['query_rimg'].replace(
                '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)
            rimg_p_path = self.queries[pidx]['query_rimg'].replace(
                '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)
        else:
            disp_a_path = self.queries[idx]['query_disp']
            disp_p_path = self.queries[pidx]['query_disp']
            rimg_a_path = self.queries[idx]['query_rimg']
            rimg_p_path = self.queries[pidx]['query_rimg']

        disp_a = np.load(disp_a_path)
        disp_a = disp_a.squeeze()
        rimg_a = np.load(rimg_a_path).astype(np.float32)
        # print(f"Anchor range image path: {self.queries[idx]['query_rimg']}")

        disp_p = np.load(disp_p_path)
        disp_p = disp_p.squeeze()
        rimg_p = np.load(rimg_p_path).astype(np.float32)
        # print(f"Positive range image path: {self.queries[pidx]['query_rimg']}")

        if self.transform_disp is not None:
            disp_a = self.transform_disp(disp_a)
            disp_p = self.transform_disp(disp_p)

        if self.data_augmentation_disp is not None:
            disp_a = self.data_augmentation_disp(disp_a)
            disp_p = self.data_augmentation_disp(disp_p)

        if self.data_augmentation_rimg is not None:
            rimg_a = self.data_augmentation_rimg(rimg_a)
            rimg_p = self.data_augmentation_rimg(rimg_p)

        xa = self.queries[idx]['x']
        ya = self.queries[idx]['y']
        yawa = self.queries[idx]['yaw']

        # print(f"Anchor Info (x, y, yaw) = ({xa}, {ya}, {yawa})")

        xp = self.queries[pidx]['x']
        yp = self.queries[pidx]['y']
        yawp = self.queries[pidx]['yaw']

        # print(f"Positive Info (x, y, yaw) = ({xp}, {yp}, {yawp})")


        if self.mode == 'train' and random.random() > 0.5: ## when training, we have 50% chance to crop the range image into sub-range image
            rimg_a, yawa_new = self.__crop_rimg(rimg=rimg_a, yaw=yawa)
            rimg_p, yawp_new = self.__crop_rimg(rimg=rimg_p, yaw=yawp)
            fov_new = 2 * self.crop_degree
        else:
            yawa_new = yawa
            yawp_new = yawp
            fov_new = self.lidar_fov

        if self.transform_rimg is not None:
            rimg_a = self.transform_rimg(rimg_a)
            rimg_p = self.transform_rimg(rimg_p)

        label1 = self.__get_label(xa=xa, ya=ya, yawa=yawa, xp=xp, yp=yp, yawp=yawp)
        label2 = self.__get_label(xa=xa, ya=ya, yawa=yawa_new, xp=xp, yp=yp, yawp=yawp_new)
        label3 = self.__get_label(xa=xa, ya=ya, yawa=yawa, xp=xp, yp=yp, yawp=yawp_new)
        label4 = self.__get_label(xa=xa, ya=ya, yawa=yawa_new, xp=xp, yp=yp, yawp=yawp)

        return disp_a, disp_p, label1, rimg_a, rimg_p, label2, label3, label4
        
    def __get_label(self, xa, ya, yawa, xp, yp, yawp):
        diff = math.fabs(yawa - yawp)
        while diff >= 2 * math.pi:
            diff -= 2 * math.pi
        if diff >= math.pi:
            diff -= 2 * math.pi
        label = (self.max_pos_dist - math.sqrt((xa - xp)**2 + (ya - yp)**2)) / self.max_pos_dist * (self.max_yaw_ang_rad - math.fabs(diff)) / self.max_yaw_ang_rad
        if label < 0 or label > 1:
            logger.error(f"Distance: {math.sqrt((xa - xp)**2 + (ya - yp)**2)}, max dist. is {self.max_pos_dist}")
            logger.error(f"Yaw diff: { math.fabs(yawa - yawp)}, max yaw diff is: {self.max_yaw_ang_rad}")
            raise ValueError
        return label

    def __get_positive_idx(self, positives: Union[List, np.array]):
        return np.random.choice(positives)

    def __crop_rimg(self, rimg, yaw):
        h, w = rimg.shape
        # crop_degree = self.lidar_fov / (self.crop_num + 1)
        crop_indices = list(range(self.crop_num))
        id = random.choice(crop_indices)
        start_degree = self.crop_degree * id
        end_degree = start_degree + 2 * self.crop_degree

        new_yaw = yaw + math.radians(self.lidar_fov / 2) - \
            math.radians(start_degree + self.crop_degree)
        new_rimg = rimg[:, w - int(end_degree / self.lidar_fov * w)                        : w - int(start_degree / self.lidar_fov * w)]
        return new_rimg, new_yaw

    def get_dataset_type(self) -> str:
        """Return the dataset type

        Returns:
            str: Return "LC2OxfordDatasetPhase1"
        """
        return "LC2OxfordDatasetPhase1"


class LC2OxfordDatasetPhase2Random(Dataset):
    """Dataset wrapper for LC2 Oxford dataset for phase 1 training (Random positive and negative).
    """

    def __init__(self, query_filepath: str, transform_disp=None, transform_rimg=None, data_augmentation_disp=None, data_augmentation_rimg=None):
        # transform: transform applied to each element
        self.query_filepath = query_filepath
        assert os.path.exists(
            self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        with open(self.query_filepath, 'rb') as handle:
            self.queries = pickle.load(handle)
        self.transform_disp = transform_disp
        self.data_augmentation_disp = data_augmentation_disp
        self.transform_rimg = transform_rimg
        self.data_augmentation_rimg = data_augmentation_rimg
        logger.info('{} queries in the dataset'.format(len(self)))

    def __len__(self):
        """Return the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.queries)

    def __getitem__(self, idx: int):
        disp_a = np.load(self.queries[idx]['query_img'])
        disp_a = disp_a.squeeze()
        rimg_a = np.load(self.queries[idx]['query_submap']).astype(np.float32)
        # print(f"Anchor range image path: {self.queries[idx]['query_rimg']}")
        pidx = self.__get_random_idx(self.queries[idx]['positives'])
        nidx = self.__get_random_idx(self.queries[idx]['negatives'])

        disp_p = np.load(self.queries[pidx]['query_img'])
        disp_p = disp_p.squeeze()
        rimg_p = np.load(self.queries[pidx]['query_submap']).astype(np.float32)

        disp_n = np.load(self.queries[nidx]['query_img'])
        disp_n = disp_p.squeeze()
        rimg_n = np.load(self.queries[nidx]['query_submap']).astype(np.float32)

        if self.transform_disp is not None:
            disp_a = self.transform_disp(disp_a)
            disp_p = self.transform_disp(disp_p)
            disp_n = self.transform_disp(disp_n)

        if self.data_augmentation_disp is not None:
            disp_a = self.data_augmentation_disp(disp_a)
            disp_p = self.data_augmentation_disp(disp_p)
            disp_n = self.data_augmentation_disp(disp_n)

        if self.transform_rimg is not None:
            rimg_a = self.transform_rimg(rimg_a)
            rimg_p = self.transform_rimg(rimg_p)
            rimg_n = self.transform_rimg(rimg_n)

        if self.data_augmentation_rimg is not None:
            rimg_a = self.data_augmentation_rimg(rimg_a)
            rimg_p = self.data_augmentation_rimg(rimg_p)
            rimg_n = self.data_augmentation_rimg(rimg_n)

        return disp_a, disp_p, disp_n, rimg_a, rimg_p, rimg_n

    def __get_random_idx(self, pools: Union[List, np.array]):
        return np.random.choice(pools)

    def get_dataset_type(self) -> str:
        """Return the dataset type

        Returns:
            str: Return "LC2OxfordDatasetPhase2Random"
        """
        return "LC2OxfordDatasetPhase2Random"


# class OxfordImageRandomTripletDataset(Dataset):
#     """Dataset wrapper for Oxford dataset for generating random triplets.
#     """

#     def __init__(self, query_filepath: str, transform=None, max_elems=None, use_undistorted=True, data_augmentation=None):
#         """Constructor for the OxfordImageDataset

#         Args:
#             query_filepath (str): Training / Val pickle file path
#             transform (torchvision.transforms.Compose, optional): The default transformation for the image data. Defaults to None.
#             max_elems (int, optional): Maximum number of data to be processed, for debugging usage. Defaults to None.
#             use_undistorted (bool, optional): Flag determining whether to use the undistorted image. Defaults to True.
#             data_augmentation (torchvision.transforms.Compose, optional): Data augmentation for images. Defaults to None.
#         """
#         # transform: transform applied to each element
#         self.use_undistorted = use_undistorted
#         self.query_filepath = query_filepath
#         assert os.path.exists(
#             self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
#         self.transform = transform
#         self.data_augmentation = data_augmentation
#         self.max_elems = max_elems

#         cached_query_filepath = self.query_filepath.replace(
#             '.pickle', '_cached.pickle')
#         # cached_query_filepath = os.path.splitext(self.query_filepath)[0] + '_cached.pickle'
#         if not os.path.exists(cached_query_filepath):
#             # Pre-process query file
#             self.queries = self.preprocess_queries(
#                 self.query_filepath, cached_query_filepath)
#         else:
#             logger.info('Loading preprocessed query file: {}...'.format(
#                 cached_query_filepath))
#             with open(cached_query_filepath, 'rb') as handle:
#                 # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
#                 self.queries = pickle.load(handle)

#         if max_elems is not None:
#             filtered_queries = {}
#             for ndx in self.queries:
#                 if ndx >= self.max_elems:
#                     break
#                 filtered_queries[ndx] = {'query_img': self.queries[ndx]['query_img'],
#                                          'positives': self.queries[ndx]['positives'][0:max_elems],
#                                          'negatives': self.queries[ndx]['negatives'][0:max_elems]}
#             self.queries = filtered_queries

#         logger.info('{} queries in the dataset'.format(len(self)))

#     def preprocess_queries(self, query_filepath: str, cached_query_filepath: str) -> dict:
#         """Preprocess the queries pickle file to make it more memory-efficient

#         Args:
#             query_filepath (str): File path of the original query
#             cached_query_filepath (str): Output path of the preprocessed query

#         Returns:
#             dict: Output preprocessed query dictionary
#         """
#         logger.info('Loading query file: {}...'.format(query_filepath))
#         with open(query_filepath, 'rb') as handle:
#             # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
#             queries = pickle.load(handle)

#         # Convert to bitarray
#         for ndx in tqdm(queries):
#             queries[ndx]['positives'] = set(queries[ndx]['positives'])
#             queries[ndx]['negatives'] = set(queries[ndx]['negatives'])
#             pos_mask = [e_ndx in queries[ndx]['positives']
#                         for e_ndx in range(len(queries))]
#             neg_mask = [e_ndx in queries[ndx]['negatives']
#                         for e_ndx in range(len(queries))]
#             queries[ndx]['positives'] = bitarray(pos_mask)
#             queries[ndx]['negatives'] = bitarray(neg_mask)

#         with open(cached_query_filepath, 'wb') as handle:
#             pickle.dump(queries, handle)

#         return queries

#     def __len__(self):
#         """Return the length of the dataset

#         Returns:
#             int: Length of the dataset
#         """
#         return len(self.queries)

#     def __getitem__(self, ndx: int):
#         """Output the image with corresponding image

#         Args:
#             ndx (int): Index

#         Returns:
#             Tuple: Image and index
#         """
#         # Load img and apply transform
#         if self.use_undistorted:
#             filename = self.queries[ndx]['query_img']
#         else:
#             filename = self.queries[ndx]['query_img'].replace(
#                 '/undistorted', '')
#         img = Image.open(filename)

#         pos_filename = queries[ndx]['positives']

#         pos_img = Image.open()
#         neg_img = Image.open()
#         if self.data_augmentation is not None:
#             img = self.data_augmentation(img)
#         if self.transform is not None:
#             img = self.transform(img)

#         return img, ndx

#     def get_positives_ndx(self, ndx):
#         """Get all the positive indices for a given query

#         Args:
#             ndx (int): query index

#         Returns:
#             list: Output list of positive sample index
#         """
#         # Get list of indexes of similar images
#         return self.queries[ndx]['positives'].search(bitarray([True]))

#     def get_negatives_ndx(self, ndx):
#         """Get all the negative indices for a given query

#         Args:
#             ndx (int): query index

#         Returns:
#             list: Output list of negative sample index
#         """
#         # Get list of indexes of dissimilar images
#         return self.queries[ndx]['negatives'].search(bitarray([True]))

#     def set_dataset_name(self, name: str):
#         """Set the dataset name

#         Args:
#             name (str): dataset name to be set
#         """
#         self.name = name

#     def get_dataset_type(self) -> str:
#         """Return the dataset type

#         Returns:
#             str: Return "OxfordImageDataset"
#         """
#         return "OxfordImageDataset"


class OxfordImageVoxelTripletDataset(Dataset):
    """Dataset wrapper for generating image and voxel triplet dataset.
    (The implementation is inspired by https://github.com/KamilZywanowski/MinkLoc3D-SI/blob/master/datasets/oxford.py)
    """

    def __init__(self, query_filepath: str,
                 transform_img=None,
                 transform_pcd=None,
                 img_data_augmentation=None,
                 pcd_data_augmentation=None,
                 default_coordinate_frame_transformation=None,
                 rebase_dir=None):
        """Constructor for the OxfordImageVoxelTripletDataset

        Args:
            query_filepath (str): Training / Val pickle file path
            transform (torchvision.transforms.Compose, optional): The default transformation for the image data. Defaults to None.
            max_elems (int, optional): Maximum number of data to be processed, for debugging usage. Defaults to None.
            use_undistorted (bool, optional): Flag determining whether to use the undistorted image. Defaults to True.
            data_augmentation (torchvision.transforms.Compose, optional): Data augmentation for images. Defaults to None.
        """
        # transform: transform applied to each element
        self.query_filepath = query_filepath
        assert os.path.exists(
            self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform_img = transform_img
        self.transform_pcd = transform_pcd
        self.img_data_augmentation = img_data_augmentation
        self.pcd_data_augmentation = pcd_data_augmentation
        self.default_coordinate_frame_transformation = default_coordinate_frame_transformation
        self.rebase_dir = rebase_dir

        cached_query_filepath = self.query_filepath.replace(
            '.pickle', '_cached.pickle')
        # cached_query_filepath = os.path.splitext(self.query_filepath)[0] + '_cached.pickle'
        if not os.path.exists(cached_query_filepath):
            # Pre-process query file
            self.queries = self.preprocess_queries(
                self.query_filepath, cached_query_filepath)
        else:
            logger.info('Loading preprocessed query file: {}...'.format(
                cached_query_filepath))
            with open(cached_query_filepath, 'rb') as handle:
                # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
                self.queries = pickle.load(handle)

        logger.info('{} queries in the dataset'.format(len(self)))

    def preprocess_queries(self, query_filepath: str, cached_query_filepath: str) -> dict:
        """Preprocess the queries pickle file to make it more memory-efficient

        Args:
            query_filepath (str): File path of the original query
            cached_query_filepath (str): Output path of the preprocessed query

        Returns:
            dict: Output preprocessed query dictionary
        """
        logger.info('Loading query file: {}...'.format(query_filepath))
        with open(query_filepath, 'rb') as handle:
            # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
            queries = pickle.load(handle)

        # Convert to bitarray
        for ndx in tqdm(queries):
            queries[ndx]['positives'] = set(queries[ndx]['positives'])
            queries[ndx]['negatives'] = set(queries[ndx]['negatives'])
            pos_mask = [e_ndx in queries[ndx]['positives']
                        for e_ndx in range(len(queries))]
            neg_mask = [e_ndx in queries[ndx]['negatives']
                        for e_ndx in range(len(queries))]
            queries[ndx]['positives'] = bitarray(pos_mask)
            queries[ndx]['negatives'] = bitarray(neg_mask)

        with open(cached_query_filepath, 'wb') as handle:
            pickle.dump(queries, handle)

        return queries

    def __len__(self):
        """Return the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.queries)

    def __getitem__(self, ndx: int):
        """Output the image with corresponding image

        Args:
            ndx (int): Index

        Returns:
            Tuple: Image and index
        """
        # Load img and apply transform
        if self.rebase_dir is None:
            img = Image.open(
                self.queries[ndx]['query_img'])
            submap = torch.from_numpy(
                np.load(self.queries[ndx]['query_submap']))
        else:
            img = Image.open(self.queries[ndx]['query_img'].replace(
                '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir))
            submap = torch.from_numpy(
                np.load(self.queries[ndx]['query_submap'].replace('/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)))

        if self.default_coordinate_frame_transformation is not None:
            submap = self.default_coordinate_frame_transformation(submap)

        if self.img_data_augmentation is not None:
            img = self.img_data_augmentation(img)

        if self.transform_img is not None:
            img = self.transform_img(img)

        if self.pcd_data_augmentation is not None:
            submap = self.pcd_data_augmentation(submap)

        if self.transform_pcd is not None:
            submap = self.transform_pcd(submap)

        return img, submap, ndx

    def get_positives_ndx(self, ndx):
        """Get all the positive indices for a given query

        Args:
            ndx (int): query index

        Returns:
            list: Output list of positive sample index
        """
        # Get list of indexes of similar images
        return self.queries[ndx]['positives'].search(bitarray([True]))

    def get_negatives_ndx(self, ndx):
        """Get all the negative indices for a given query

        Args:
            ndx (int): query index

        Returns:
            list: Output list of negative sample index
        """
        # Get list of indexes of dissimilar images
        return self.queries[ndx]['negatives'].search(bitarray([True]))

    def set_dataset_name(self, name: str):
        """Set the dataset name

        Args:
            name (str): dataset name to be set
        """
        self.name = name

    def get_dataset_type(self) -> str:
        """Return the dataset type

        Returns:
            str: Return "OxfordImageVoxelTripletDataset"
        """
        return "OxfordImageVoxelTripletDataset"


class LC2OxfordDatasetPhase2(Dataset):
    """Dataset wrapper for LC2 phase 2 training.
    """

    def __init__(self, query_filepath: str,
                 transform_img=None,
                 transform_pcd=None,
                 img_data_augmentation=None,
                 pcd_data_augmentation=None
                 ):
        # transform: transform applied to each element
        self.query_filepath = query_filepath
        assert os.path.exists(
            self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform_img = transform_img
        self.transform_pcd = transform_pcd
        self.img_data_augmentation = img_data_augmentation
        self.pcd_data_augmentation = pcd_data_augmentation

        cached_query_filepath = self.query_filepath.replace(
            '.pickle', '_cached.pickle')
        # cached_query_filepath = os.path.splitext(self.query_filepath)[0] + '_cached.pickle'
        if not os.path.exists(cached_query_filepath):
            # Pre-process query file
            self.queries = self.preprocess_queries(
                self.query_filepath, cached_query_filepath)
        else:
            logger.info('Loading preprocessed query file: {}...'.format(
                cached_query_filepath))
            with open(cached_query_filepath, 'rb') as handle:
                # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
                self.queries = pickle.load(handle)

        logger.info('{} queries in the dataset'.format(len(self)))

    def preprocess_queries(self, query_filepath: str, cached_query_filepath: str) -> dict:
        """Preprocess the queries pickle file to make it more memory-efficient

        Args:
            query_filepath (str): File path of the original query
            cached_query_filepath (str): Output path of the preprocessed query

        Returns:
            dict: Output preprocessed query dictionary
        """
        logger.info('Loading query file: {}...'.format(query_filepath))
        with open(query_filepath, 'rb') as handle:
            # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
            queries = pickle.load(handle)

        # Convert to bitarray
        for ndx in tqdm(queries):
            queries[ndx]['positives'] = set(queries[ndx]['positives'])
            queries[ndx]['negatives'] = set(queries[ndx]['negatives'])
            pos_mask = [e_ndx in queries[ndx]['positives']
                        for e_ndx in range(len(queries))]
            neg_mask = [e_ndx in queries[ndx]['negatives']
                        for e_ndx in range(len(queries))]
            queries[ndx]['positives'] = bitarray(pos_mask)
            queries[ndx]['negatives'] = bitarray(neg_mask)

        with open(cached_query_filepath, 'wb') as handle:
            pickle.dump(queries, handle)

        return queries

    def __len__(self):
        """Return the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.queries)

    def __getitem__(self, ndx: int):
        """Output the image with corresponding image

        Args:
            ndx (int): Index

        Returns:
            Tuple: Image and index
        """
        # Load img and apply transform
        disp = np.load(self.queries[ndx]['query_img'])
        disp = disp.squeeze()
        rimg = np.load(self.queries[ndx]['query_submap']).astype(np.float32)

        if self.transform_img is not None:
            disp = self.transform_img(disp)

        if self.img_data_augmentation is not None:
            disp = self.img_data_augmentation(disp)

        if self.transform_pcd is not None:
            rimg = self.transform_pcd(rimg)

        if self.pcd_data_augmentation is not None:
            rimg = self.pcd_data_augmentation(rimg)

        return disp, rimg, ndx

    def get_positives_ndx(self, ndx):
        """Get all the positive indices for a given query

        Args:
            ndx (int): query index

        Returns:
            list: Output list of positive sample index
        """
        # Get list of indexes of similar images
        return self.queries[ndx]['positives'].search(bitarray([True]))

    def get_negatives_ndx(self, ndx):
        """Get all the negative indices for a given query

        Args:
            ndx (int): query index

        Returns:
            list: Output list of negative sample index
        """
        # Get list of indexes of dissimilar images
        return self.queries[ndx]['negatives'].search(bitarray([True]))

    def set_dataset_name(self, name: str):
        """Set the dataset name

        Args:
            name (str): dataset name to be set
        """
        self.name = name

    def get_dataset_type(self) -> str:
        """Return the dataset type

        Returns:
            str: Return "LC2OxfordDatasetPhase2"
        """
        return "LC2OxfordDatasetPhase2"


class OxfordImagePointcloudDataset(Dataset):
    """Dataset generate pair of image and pointcloud
    """

    def __init__(self, annotation_path: str,
                 sample_index_list_path: str,
                 transform_img=None,
                 transform_pcd=None,
                 img_data_augmentation=None,
                 pcd_data_augmentation=None,
                 random_horizontal_mirroring_p=None,
                 default_coordinate_frame_transformation=None,
                 voxelization=False,
                 verbose=False,
                 rebase_dir=None):
        """Constructor

        Args:
            annotation_path (str): Annotation file for the parsed image and pointcloud dataset
            sample_index_list_path (str): Row indices in the annotation file that would be used in this dataset
            transform_img (torchvision.transforms.Compose, optional): Image preprocessing transformation. Defaults to None.
            transform_pcd (torchvision.transforms.Compose, optional): Pointcloud preprocessing transformation. Defaults to None.
            img_data_augmentation (torchvision.transforms.Compose, optional): Image data augmentation. Defaults to None.
            pcd_data_augmentation (torchvision.transforms.Compose, optional): Pointcloud data augmentation. Defaults to None.
            random_horizontal_mirroring_p (float or None, optional): Part of the data augmentation but apply to both image and pointcloud data.
                                                                    When set to 0.5, it means that 50% mirroring for both images and pointcloud.
                                                                    Default to None.
            default_coordinate_frame_transformation (torchvision.transforms.Compose, optional): The coordinate transformation of the point cloud data to the desired configuration. Defaults to None.
            voxelization (bool, optional): Please set this flag if "transform_pcd" contains Voxelization. Defaults to False.
            verbose (bool, optional): Set the flag to print all debug information. Defaults to False.
            rebase_dir (str, optional): Redirect the base directory if specify. Defaults to None
        """
        self.annotation = pd.read_csv(annotation_path)
        with open(sample_index_list_path, 'rb') as file:
            self.indices = pickle.load(file)

        self.transform_img = transform_img
        self.transform_pcd = transform_pcd
        self.img_data_augmentation = img_data_augmentation
        self.pcd_data_augmentation = pcd_data_augmentation
        self.random_horizontal_mirroring_p = random_horizontal_mirroring_p
        self.default_coordinate_frame_transformation = default_coordinate_frame_transformation
        self.voxelization = voxelization
        self.verbose = verbose
        self.rebase_dir = rebase_dir

    def __len__(self):
        '''
        Description:
            Return the number of pairs
        '''
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Return corresponding image and submap data

        Args:
            idx (int): Index

        Returns:
            tuple: 
                img (torch.tensor): The requested image
                submap (torch.tensor or tuple(torch.tensor)): The requested submap
        """
        if self.rebase_dir is None:
            img = Image.open(
                self.annotation.iloc[self.indices[idx]]['img_path'])
            submap_path = self.annotation.iloc[self.indices[idx]]['submap_path']
            if '.bin' not in submap_path:
                submap = torch.from_numpy(
                np.load(submap_path))
            else:
                submap = np.fromfile(submap_path, dtype=np.float32).reshape((-1,4))
                submap = torch.from_numpy(submap[:, 0:3]) # lidar xyz (front, left, up)
        else:
            img = Image.open(self.annotation.iloc[self.indices[idx]]['img_path'].replace(
                '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir))
            submap_path = self.annotation.iloc[self.indices[idx]]['submap_path'].replace('/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)
            if '.bin' not in submap_path:
                submap = torch.from_numpy(
                np.load(submap_path))
            else:
                submap = np.fromfile(submap_path, dtype=np.float32).reshape((-1,4))
                submap = torch.from_numpy(submap[:, 0:3]) # lidar xyz (front, left, up)
        if self.verbose:
            print(
                f"Raw image path {self.annotation.iloc[self.indices[idx]]['img_path']}")
            print(
                f"Raw submap path {self.annotation.iloc[self.indices[idx]]['submap_path']}")

        if self.default_coordinate_frame_transformation is not None:
            submap = self.default_coordinate_frame_transformation(submap)

        if self.random_horizontal_mirroring_p is not None:
            p = random.random()
            if p < self.random_horizontal_mirroring_p:
                img = hflip(img)
                submap = HorizontalMirroring()(submap)
                if self.verbose:
                    print("image and pointcloud mirroring applied")

        if self.img_data_augmentation is not None:
            img = self.img_data_augmentation(img)

        if self.transform_img is not None:
            img = self.transform_img(img)

        if self.pcd_data_augmentation is not None:
            submap = self.pcd_data_augmentation(submap)

        if self.transform_pcd is not None:
            submap = self.transform_pcd(submap)

        return img, submap

    def get_dataset_type(self) -> str:
        """Return the dataset type

        Returns:
            str: Return dataset type
        """
        if self.voxelization:
            return "OxfordImageVoxelDataset"
        else:
            return "OxfordImagePointcloudDataset"


class PlaceRecognitionInferenceQuery(Dataset):
    """The dataset for the inference query
    """

    def __init__(self, pickle_path: str, transform_img=None, transform_submap=None, default_coordinate_frame_transformation=None, verbose=False, voxelization=False, rebase_dir=None, model='vxp'):
        """Constructor

        Args:
            pickle_path (str): File path to the query pickle
            transform_img (torchvision.transforms.Compose, optional): Image transformation (e.g. normalization...). Defaults to None.
            transform_submap (torchvision.transforms.Compose, optional): Pointcloud transformation (e.g. voxelization...). Defaults to None.
            default_coordinate_frame_transformation (_type_, optional): Pointcloud coordinate frame transformation. Defaults to None.
            verbose (bool, optional): Set to print debug message. Defaults to False.
            voxelization (bool, optional): Set to indicate voxel mode. Defaults to False.
        """
        assert model in ['vxp', 'lc2', 'cattaneo']

        self.transform_img = transform_img
        self.transform_submap = transform_submap
        with open(pickle_path, 'rb') as f:
            self.q = pickle.load(f)
        self.test_index = 0
        self.db_index = 0
        self.verbose = verbose
        self.default_coordinate_frame_transformation = default_coordinate_frame_transformation
        self.voxelization = voxelization
        self.rebase_dir = rebase_dir
        self.model = model

    def set_test_index(self, index: int):
        """Set the current query index

        Args:
            index (int): current query index
        """
        self.test_index = index

    def set_db_index(self, index: int):
        """Set the current database index

        Args:
            index (int): current database index
        """
        self.db_index = index

    def __len__(self) -> int:
        """Return the number of current queries 

        Returns:
            int: Return the number of queries
        """
        return len(self.q[self.test_index])

    def __getitem__(self, idx: int) -> tuple:
        """Get item from dataset with index

        Args:
            idx (int): Index for the dataset

        Returns:
            tuple: Retrn tuple(image, submap, index)
        """
        if self.rebase_dir is not None:
            img_path = self.q[self.test_index][idx]['img_path'].replace(
                '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)
            submap_path = self.q[self.test_index][idx]['submap_path'].replace(
                '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)
        else:
            img_path = self.q[self.test_index][idx]['img_path']
            submap_path = self.q[self.test_index][idx]['submap_path']

        if self.model == 'lc2':
            disp_name = os.path.basename(img_path)
            img_path = img_path.replace(
                disp_name, os.path.join("disp_npy", disp_name)).replace('.png', '.npy')
            submap_path = submap_path.replace(
                'dataset_every_5m_45runs', 'dataset_every_5m_45runs_w_ground').replace('submap', 'rimg')
            img = np.load(img_path)
            img = img.squeeze()
            submap = np.load(submap_path).astype(np.float32)
        else:
            img = Image.open(img_path)
            if '.bin' not in submap_path:
                submap = torch.from_numpy(
                np.load(submap_path))
            else:
                submap = np.fromfile(submap_path, dtype=np.float32).reshape((-1,4))
                submap = torch.from_numpy(submap[:, 0:3]) # lidar xyz (front, left, up)

        # if self.verbose:
        #     print(f"Raw image path {self.q[self.test_index][idx]['img_path']}")
        #     print(
        #         f"Raw submap path {self.q[self.test_index][idx]['submap_path']}")

        if self.default_coordinate_frame_transformation is not None:
            submap = self.default_coordinate_frame_transformation(submap)

        if self.transform_img is not None:
            img = self.transform_img(img)

        if self.transform_submap is not None:
            submap = self.transform_submap(submap)

        return img, submap, torch.tensor(idx, dtype=torch.int64)

    def get_gt(self) -> list:
        """Return ground-truth of the current query

        Returns:
            list: Ground-truth of the current query
        """
        gt = []

        for i in range(len(self.q[self.test_index].keys())):
            gt.append(np.array(self.q[self.test_index][i][self.db_index]))

        return gt

    def getUTM(self, idx: int) -> tuple:
        """Return the UTM coordinate for a sample

        Args:
            idx (int): sample index

        Returns:
            tuple: tuple(x, y) 
        """
        x = self.q[self.test_index][idx]['northing']
        y = self.q[self.test_index][idx]['easting']

        return x, y

    def get_dataset_type(self) -> str:
        """Return dataset type

        Returns:
            str: type
        """
        if self.voxelization:
            return "InferenceImageVoxelDataset"
        else:
            if self.model == 'lc2':
                return "InferenceDispRimgDataset"
            else:
                return "InferenceImagePointcloudDataset"


class PlaceRecognitionInferenceDb(Dataset):
    """The dataset for inference database
    """

    def __init__(self, pickle_path: str, transform_img=None, transform_submap=None, default_coordinate_frame_transformation=None, verbose=False, voxelization=False, rebase_dir=None, model='vxp'):
        """Constructor

        Args:
            pickle_path (str): File path to the database pickle file
            transform_img (torchvision.transforms.Compose, optional): Image transformation (e.g. normalization...). Defaults to None.
            transform_submap (torchvision.transforms.Compose, optional): Pointcloud transformation (e.g. voxelization...). Defaults to None.
            default_coordinate_frame_transformation (_type_, optional): Pointcloud coordinate frame transformation. Defaults to None.
            verbose (bool, optional): Set to print debug message. Defaults to False.
            voxelization (bool, optional): Set to indicate voxel mode. Defaults to False.
        """
        assert model in ['vxp', 'lc2', 'cattaneo']
        self.transform_img = transform_img
        self.transform_submap = transform_submap
        with open(pickle_path, 'rb') as f:
            self.db = pickle.load(f)
        self.db_index = 0
        self.verbose = verbose
        self.default_coordinate_frame_transformation = default_coordinate_frame_transformation
        self.voxelization = voxelization
        self.rebase_dir = rebase_dir
        self.model = model

    def set_db_index(self, index: int):
        """Set the current database id

        Args:
            index (int): Current database id
        """
        self.db_index = index

    def __len__(self) -> int:
        """Return the length of the current database

        Returns:
            int: Length of the current database
        """
        return len(self.db[self.db_index])

    def __getitem__(self, idx: int) -> tuple:
        """Get sample from the dataset with index

        Args:
            idx (int): sample index

        Returns:
            tuple: tuple(image, submap, index)
        """

        if self.rebase_dir is not None:
            img_path = self.db[self.db_index][idx]['img_path'].replace(
                '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)
            submap_path = self.db[self.db_index][idx]['submap_path'].replace(
                '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)
        else:
            img_path = self.db[self.db_index][idx]['img_path']
            submap_path = self.db[self.db_index][idx]['submap_path']

        if self.model == 'lc2':
            disp_name = os.path.basename(img_path)
            img_path = img_path.replace(
                disp_name, os.path.join("disp_npy", disp_name)).replace('.png', '.npy')
            submap_path = submap_path.replace(
                'dataset_every_5m_45runs', 'dataset_every_5m_45runs_w_ground').replace('submap', 'rimg')
            img = np.load(img_path)
            img = img.squeeze()
            submap = np.load(submap_path).astype(np.float32)
        else:
            img = Image.open(img_path)
            if '.bin' not in submap_path:
                submap = torch.from_numpy(
                np.load(submap_path))
            else:
                submap = np.fromfile(submap_path, dtype=np.float32).reshape((-1,4))
                submap = torch.from_numpy(submap[:, 0:3]) # lidar xyz (front, left, up)
            

        # if self.rebase_dir is not None:
        #     img = Image.open(img_path.replace(
        #         '/storage/user/lyun/Oxford_Robocar/', self.rebase_dir))
        #     submap = torch.from_numpy(
        #         np.load(submap_path.replace('/storage/user/lyun/Oxford_Robocar/', self.rebase_dir)))
        # else:
        #     img = Image.open(img_path)
        #     submap = torch.from_numpy(
        #         np.load(submap_path))

        if self.default_coordinate_frame_transformation is not None:
            submap = self.default_coordinate_frame_transformation(submap)

        if self.transform_img is not None:
            img = self.transform_img(img)
        if self.transform_submap is not None:
            submap = self.transform_submap(submap)

        return img, submap, torch.tensor(idx, dtype=torch.int64)

    def getUTM(self, idx: int) -> tuple:
        """Return UTM coordinate for the sample

        Args:
            idx (int): sample index

        Returns:
            tuple: tuple(x, y)
        """
        x = self.db[self.db_index][idx]['northing']
        y = self.db[self.db_index][idx]['easting']

        return x, y

    def get_dataset_type(self) -> str:
        """Return dataset type

        Returns:
            str: type
        """
        if self.voxelization:
            return "InferenceImageVoxelDataset"
        else:
            if self.model == 'lc2':
                return "InferenceDispRimgDataset"
            else:
                return "InferenceImagePointcloudDataset"
