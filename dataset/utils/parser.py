#
# Created on Mon May 1 2023
# The MIT License (MIT)
# Copyright (c) 2023 Yun-Jin Li (Jim)
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
import os
from PIL import Image
from tqdm import tqdm
import math
import open3d as o3d
import numpy as np
import pandas as pd
import yaml

# From the Oxford RobotCar SDK
from camera_model import CameraModel
from transform import build_se3_transform
from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from build_pointcloud import build_pointcloud
from transform import so3_to_euler, se3_to_components
from typing import List

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


class Parser:
    def __init__(self, 
                 base_dir = "/storage/group/dataset_mirrors/01_incoming/Oxford_RobotCar/",
                 ins_dir = None,
                 save_dir = "/storage/user/lyun/Oxford_Robocar_test/",
                 lidar_dir = None,
                 cam_central = False,
                 max_depth = 50.0, 
                 max_lateral = 25.0,
                 max_vertical = 25.0,
                 depth_at_least = 30.0,
                 every_x_meter = 5.0,
                 plane_error_threshold = 0.1,
                 ransac_threshold = 0.2,
                 ransac_iteration = 1000,
                 save_submap_with_ground = False,
                 write_to_annotation = True,
                 ):
        self.base_dir = base_dir
        self.save_dir = save_dir
        self.max_depth = max_depth
        self.max_lateral = max_lateral
        self.max_vertical = max_vertical
        self.depth_at_least = depth_at_least
        self.every_x_meter = every_x_meter
        self.plane_error_threshold = plane_error_threshold
        self.ransac_threshold = ransac_threshold
        self.ransac_iteration = ransac_iteration
        # self.remove_ground = setup['parameter']['remove_ground']
        self.ins_dir = ins_dir
        self.cam_central = cam_central
        self.lidar_base_dir = lidar_dir
        self.save_submap_with_ground = save_submap_with_ground
        self.write_to_annotation = write_to_annotation

        logger.info("Submap span from " +
                    "both x-direction of the camera" if self.cam_central else "positive x-direction of the camera")
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # with open(os.path.join(self.save_dir, 'setup.yml'), 'w') as outfile:
        #     yaml.dump(setup, outfile, default_flow_style=False)

        logger.info(f"Base directory: {os.path.abspath(self.base_dir)}")
        self.extrinsics_dir = 'extrinsics'
        gps = 'gps'
        img = 'image'
        lidar = 'lms_front'
        self.img_save_dir = os.path.join(self.save_dir, img)
        if not os.path.exists(self.img_save_dir):
            os.mkdir(self.img_save_dir)

        self.submap_save_dir = os.path.join(self.save_dir, 'submap')
        self.raw_submap_save_dir = os.path.join(self.save_dir, 'raw_submap')
        if not os.path.exists(self.submap_save_dir):
            os.mkdir(self.submap_save_dir)
        if not os.path.exists(self.raw_submap_save_dir):
            os.mkdir(self.raw_submap_save_dir)
        if self.ins_dir is not None:
            self.gps_base_dir = self.ins_dir
        else:
            self.gps_base_dir = os.path.join(self.base_dir, gps)

        if not os.path.exists(self.gps_base_dir):
            logger.error(
                F"Cannot find ground truth pose base with directory: {os.path.abspath(self.gps_base_dir)}")
            raise RuntimeError()

        self.img_base_dir = os.path.join(self.base_dir, img)
        if not os.path.exists(self.img_base_dir):
            logger.error(
                F"Cannot find image base with directory: {os.path.abspath(self.img_base_dir)}")
            raise RuntimeError()

        if self.lidar_base_dir is None:
            self.lidar_base_dir = os.path.join(self.base_dir, lidar)
        if not os.path.exists(self.lidar_base_dir):
            logger.error(
                F"Cannot find LiDAR base with directory: {os.path.abspath(self.lidar_base_dir)}")
            raise RuntimeError()

        logger.info(f"Base directory exists")

        self.model = CameraModel('models', self.img_base_dir)
        logger.info(self.model.focal_length)
        logger.info(self.model.principal_point)
        logger.info(self.model.G_camera_image)
        logger.info(f"Create camera model")

        extrinsics_path = os.path.join(
            self.extrinsics_dir, self.model.camera + '.txt')
        with open(extrinsics_path) as extrinsics_file:
            extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

        self.G_camera_vehicle = build_se3_transform(extrinsics)
        self.G_camera_posesource = None

        with open(os.path.join(self.extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            logger.debug(f'INS extrinsics: {extrinsics}')
            self.G_camera_posesource = self.G_camera_vehicle * \
                build_se3_transform([float(x) for x in extrinsics.split(' ')])

    def run(self, date='2014-11-11-11-06-25', force_write=False, only_save: List = None):
        logger.info(f"Parsing data: {date}")
        cam_timestamps = []
        img_dir = os.path.join(self.img_base_dir, date)
        if not os.path.exists(img_dir):
            logger.error(f"{date} sequence not exist")
            return
        for root, dirs, files in os.walk(img_dir):
            for dirname in sorted(files):
                try:
                    cam_timestamps.append((int)(dirname.split('.')[0]))
                except:
                    logger.warn(f"Cannot parser {dirname}")

        lidar_timestamps = []
        lidar_dir = os.path.join(self.lidar_base_dir, date)

        if not os.path.exists(lidar_dir):
            logger.error(f"{date} sequence not exist")
            return
        for root, dirs, files in os.walk(lidar_dir):
            for dirname in sorted(files):
                try:
                    lidar_timestamps.append((int)(dirname.split('.')[0]))
                except:
                    logger.warn(f"Cannot parser {dirname}")

        # if self.ins_dir is not None:
        poses_file = os.path.join(self.gps_base_dir, date, 'ins.csv')
        # else:
        #     poses_file = os.path.join(self.rtk_base_dir, date, 'rtk.csv')

        if not os.path.exists(poses_file):
            logger.error(
                f"No available gps file found in directory: {os.path.abspath(poses_file)}")
            raise RuntimeError()

        try:
            # print((self.ins_dir is None))
            start_pose_abs, poses = interpolate_ins_poses(
                poses_file, cam_timestamps, cam_timestamps[0], use_rtk=False)
        except:
            logger.error(f"Error when parsing the sequence {date}")
            return
        delta_sum = 0.0
        counter = 0

        date_submap_save_dir = os.path.join(self.submap_save_dir, date)
        date_raw_submap_save_dir = os.path.join(self.raw_submap_save_dir, date)

        date_image_save_dir = os.path.join(self.img_save_dir, date)
        date_image_undistorted_save_dir = os.path.join(
            self.img_save_dir, 'undistorted', date)

        if not os.path.exists(date_submap_save_dir):
            os.mkdir(date_submap_save_dir)
        if not os.path.exists(date_raw_submap_save_dir):
            os.mkdir(date_raw_submap_save_dir)

        if not os.path.exists(date_image_save_dir):
            os.mkdir(date_image_save_dir)
        if not os.path.exists(os.path.join(self.img_save_dir, 'undistorted')):
            os.mkdir(os.path.join(self.img_save_dir, 'undistorted'))
        if not os.path.exists(date_image_undistorted_save_dir):
            os.mkdir(date_image_undistorted_save_dir)

        # annotation = pd.DataFrame(columns=['timestamp', 'date', 'northing', 'easting', 'yaw', 'submap_path', 'img_path'])
        annotation = pd.DataFrame({})
        counter = 0
        for i in tqdm(range(1, len(poses), 1)):
            if delta_sum < self.every_x_meter:
                delta_x = poses[i][0, 3] - poses[i - 1][0, 3]
                delta_x **= 2
                delta_y = poses[i][1, 3] - poses[i - 1][1, 3]
                delta_y **= 2
                try:
                    delta_sum += math.sqrt(delta_x + delta_y)
                except:
                    logger.warn(
                        f'Compute square root fail: {delta_x} + {delta_y} = {delta_x + delta_y}')
                continue
            delta_sum = 0.0

            timestamp = cam_timestamps[i]
            if only_save is not None:
                if str(timestamp) not in only_save:
                    logger.info(
                        f"You enable the only_save command. Only sample with timestamps {only_save} would be save. Skiping {timestamp}")
                    continue
            src_img = os.path.join(
                img_dir, str(timestamp) + '.png')

            dst_img = os.path.join(date_image_save_dir,
                                   str(timestamp) + '.png')
            dst_undistorted_img = os.path.join(
                date_image_undistorted_save_dir, str(timestamp) + '.png')
            dst_img_exist = False
            dst_undistorted_img_exist = False
            submap_exist = False
            if not force_write:
                if not os.path.exists(dst_img):
                    try:
                        img = Image.open(src_img)
                        img = img.resize((1280 // 4, 960 // 4), Image.BILINEAR)
                        img.save(dst_img)
                    except:
                        continue
                else:
                    dst_img_exist = True
                if not os.path.exists(dst_undistorted_img):
                    try:
                        img = Image.open(src_img)
                        img = img.resize((1280, 960), Image.BILINEAR)
                        img = self.model.undistort(np.asarray(img))
                        img = Image.fromarray(img)
                        img = img.resize((1280 // 4, 960 // 4), Image.BILINEAR)
                        img.save(dst_undistorted_img)
                    except:
                        logger.warning(
                            f"Fail to generate undistorted image at {timestamp}")
                        continue
                else:
                    dst_undistorted_img_exist = True
                if os.path.exists(os.path.join(date_submap_save_dir, str(timestamp) + '.npy')):
                    # continue
                    submap_exist = True
                    if submap_exist:
                        logger.info(
                            f"Sample at {timestamp} already exists, write to the annotation and skip")
                        xyzrpy = se3_to_components(start_pose_abs * poses[i])
                        new_row = {'timestamp': timestamp, 'date': date, 'northing': xyzrpy[0], 'easting': xyzrpy[1], 'yaw': xyzrpy[-1], 'submap_path': os.path.abspath(
                            os.path.join(date_submap_save_dir, str(timestamp) + '.npy')), 'img_path': os.path.abspath(dst_undistorted_img)}
                        logger.info(
                            f"Write the dataframe at {timestamp}: {new_row}")
                        annotation = pd.concat(
                            [annotation, pd.DataFrame([new_row])], ignore_index=True)
                        counter += 1
                        continue
                    else:
                        # Do nothing
                        pass
            else:
                try:
                    img = Image.open(src_img)
                    img = img.resize((1280 // 4, 960 // 4), Image.BILINEAR)
                    img.save(dst_img)
                except:
                    continue
                try:
                    img = Image.open(src_img)
                    img = img.resize((1280, 960), Image.BILINEAR)
                    img = self.model.undistort(np.asarray(img))
                    img = Image.fromarray(img)
                    img = img.resize((1280 // 4, 960 // 4), Image.BILINEAR)
                    img.save(dst_undistorted_img)
                except:
                    logger.warning(
                        f"Fail to generate undistorted image at {timestamp}")
                    continue

            if not self.cam_central:
                try:
                    pointcloud, _ = build_pointcloud(lidar_dir, poses_file, self.extrinsics_dir,
                                                     timestamp, timestamp + 2e7, timestamp, lidar_timestamps)
                except KeyboardInterrupt:
                    logger.exception("Keyboard interrupt")
                    break
                except:
                    logger.warn(
                        f"Generate submap at timestamp {timestamp} fails")
                    continue
            else:
                # try:
                pointcloud1, start_pose_abs1 = build_pointcloud(lidar_dir, poses_file, self.extrinsics_dir,
                                                                timestamp, timestamp + 2e7, timestamp, lidar_timestamps)
                pointcloud1_abs = np.dot(start_pose_abs1, pointcloud1)
                pointcloud2, start_pose_abs2 = build_pointcloud(lidar_dir, poses_file, self.extrinsics_dir,
                                                                timestamp - 2e7, timestamp, timestamp - 2e7, lidar_timestamps)
                pointcloud2_abs = np.dot(start_pose_abs2, pointcloud2)
                # print(pointcloud2_abs.shape)
                # print(pointcloud1_abs.shape)
                pointcloud_abs = np.concatenate(
                    (pointcloud2_abs, pointcloud1_abs), axis=1)
                pose_abs = start_pose_abs * poses[i]
                pointcloud = np.dot(np.linalg.inv(pose_abs), pointcloud_abs)
                # except KeyboardInterrupt:
                #     logger.exception("Keyboard interrupt")
                #     break
                # except:
                #     logger.warn(f"Generate submap at timestamp {timestamp} fails")
                #     continue

            pointcloud = np.dot(self.G_camera_posesource, pointcloud)
            filtered = np.zeros((pointcloud.shape[1], 3))
            filtered[:, 0] = np.squeeze(np.transpose(pointcloud)[:, 0])
            filtered[:, 1] = np.squeeze(np.transpose(pointcloud)[:, 1])
            filtered[:, 2] = np.squeeze(np.transpose(pointcloud)[:, 2])

            if not self.cam_central:
                filtered = filtered[filtered[:, 0] < self.max_depth]
                filtered = filtered[filtered[:, 0] > 0.0]
                filtered = filtered[filtered[:, 1] < self.max_lateral]
                filtered = filtered[filtered[:, 1] > -self.max_lateral]
                filtered = filtered[filtered[:, 2] < self.max_vertical]
                filtered = filtered[filtered[:, 2] > -self.max_vertical]
                if filtered.shape[0] == 0:
                    logger.warn(f"No points after filtering {timestamp}")
                    continue
                if np.max(filtered[:, 0]) < self.depth_at_least:
                    logger.warn(
                        f"The span pointcloud submap is only {np.max(filtered[:, 0])} which is smaller than the the desire configuration ({self.depth_at_least})")
                    continue
            else:
                filtered = filtered[np.abs(
                    filtered[:, 0]) < self.max_depth / 2]
                filtered = filtered[filtered[:, 1] < self.max_lateral]
                filtered = filtered[filtered[:, 1] > -self.max_lateral]
                filtered = filtered[filtered[:, 2] < self.max_vertical]
                filtered = filtered[filtered[:, 2] > -self.max_vertical]
                if filtered.shape[0] == 0:
                    logger.warn(f"No points after filtering {timestamp}")
                    continue

                filtered = filtered[np.abs(filtered[:, 0]) < np.max(
                    np.abs(filtered[:, 0]))]

                if np.max(np.abs(filtered[:, 0])) < self.depth_at_least / 2:
                    logger.warn(
                        f"The span pointcloud submap is only {np.max(np.abs(filtered[:, 0])) * 2} which is smaller than the the desire configuration ({self.depth_at_least})")
                    continue

            if self.save_submap_with_ground:
                raw_filtered = np.copy(filtered)
                np.random.shuffle(raw_filtered)
                save_path = os.path.join(
                    date_raw_submap_save_dir, str(timestamp) + '.npy')
                np.save(save_path, raw_filtered)

            # if self.remove_ground is not None:
            sub_filtered = np.copy(filtered)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sub_filtered)

            coeff, plane_indices = pcd.segment_plane(
                self.ransac_threshold, 3, self.ransac_iteration)

            if np.abs(np.sum(coeff[:3] * np.array([0, 0, 1]))) < (1 - self.plane_error_threshold):
                # Not fit the ground
                logger.warning(
                    "The fit gound is not correct, discard this sample")
                continue
            else:
                logger.debug(
                    f"Successfully remove {len(plane_indices)} point on the ground")

            sub_filtered = np.delete(sub_filtered, plane_indices, axis=0)

            np.random.shuffle(sub_filtered)

            # ========== Save the PCD ========== #
            save_path = os.path.join(
                date_submap_save_dir, str(timestamp) + '.npy')
            np.save(save_path, sub_filtered)
            if self.write_to_annotation:
                xyzrpy = se3_to_components(start_pose_abs * poses[i])
                # euler = so3_to_euler(poses[i][:3, :3])
                logger.debug(f"6d pose is: {xyzrpy}")
                logger.debug(f"{xyzrpy.shape}")
                # 'timestamp', 'date', 'northing', 'easting', 'yaw', 'submap_path', 'img_path'
                # annotation = annotation.append(pd.DataFrame({'timestamp': timestamp, 'date': date, 'northing': poses[i][0, 3], 'easting': poses[i][1, 3], 'yaw': euler[2], 'submap_path': os.path.abspath(save_path), 'img_path': os.path.abspath(dst_img)}))
                new_row = {'timestamp': timestamp, 'date': date, 'northing': xyzrpy[0], 'easting': xyzrpy[1], 'yaw': xyzrpy[-1], 'submap_path': os.path.abspath(
                    save_path), 'img_path': os.path.abspath(dst_undistorted_img)}
                annotation = pd.concat(
                    [annotation, pd.DataFrame([new_row])], ignore_index=True)
            counter += 1
        logger.info(
            f"Successfully generate {counter} submaps from {date} sequence")
        if self.write_to_annotation:
            annotation_filename = f'annotation_{date}.csv'
            annotation_path = os.path.join(
                date_submap_save_dir, annotation_filename)
            annotation.to_csv(annotation_path)
            logger.info(
                f"Annotation file for sequence {date} has been saved to {os.path.abspath(annotation_path)}")
