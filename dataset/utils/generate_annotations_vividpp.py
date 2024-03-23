#
# Created on Fri Mar 22 2024
# The MIT License (MIT)
# Copyright (c) 2024 Mariia Gladkova, Technical University of Munich
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

import argparse
import os
import csv
import numpy as np
import bisect
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] [%(name)s] %(asctime)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def interpolate_poses(gps_poses, gps_timestamps, requested_timestamps):
    upper_ids = np.array([bisect.bisect(gps_timestamps, pt)
                 for pt in requested_timestamps])
    lower_ids = np.array([u - 1 for u in upper_ids])

    fractions = (requested_timestamps - gps_timestamps[lower_ids]) / \
                (gps_timestamps[upper_ids] - gps_timestamps[lower_ids])
    fractions = np.expand_dims(fractions, axis=1)
    logger.info(fractions)
    
    pos_lower = gps_poses[lower_ids, :]
    pos_upper = gps_poses[upper_ids, :]
    pos_interp = np.multiply(np.tile(1 - fractions, (1, 3)), pos_lower) + \
        np.multiply(np.tile(fractions, (1, 3)), pos_upper)

    return pos_interp

def main(datapath, save_dir=None):
    seqname = os.path.basename((os.path.normpath(datapath)))
    logger.info(f"create sub-annotation for sequence {seqname}")
    imagepath = os.path.join(datapath, 'images')
    lidarpath = os.path.join(datapath, 'lidar')
    gpspath = os.path.join(datapath, 'gpslist_enu.txt')

    imagelist = sorted(os.listdir(imagepath))
    lidarlist =  sorted(os.listdir(lidarpath))

    logger.info('Image files: ', len(imagelist))
    logger.info('Lidar files: ', len(lidarlist))

    gps_poses = []
    gps_timestamps = []
    with open(gpspath, 'r') as f:
        for line in f:
            gps_timestamps.append(float(line.split(',')[-1]))
            gps_poses.append([float(x) for x in line.split(',')[:3]])
    gps_poses = np.array(gps_poses)
    gps_timestamps = np.array(gps_timestamps)
    logger.info('GPS poses: ', gps_poses.shape)

    # Interpolate GPS poses for each lidar frame
    lidar_timestamps = []
    image_timestamps = []
    for ltimestamp in tqdm(range(len(lidarlist))):
        abs_lidarstamps = [float(lidarlist[i][:-4]) + 1621837330.818392 for i in range(len(lidarlist))]
        closest_image_idx = 0
        closest_image_dist = abs(float(imagelist[0][:-4]) - abs_lidarstamps[ltimestamp])
        for i in range(1, len(imagelist)):
            diff = abs(float(imagelist[i][:-4]) - abs_lidarstamps[ltimestamp])
            if diff < closest_image_dist:
                closest_image_dist = diff
                closest_image_idx = i

        timestamp = closest_image_idx
        lidar_timestamps.append(float(abs_lidarstamps[ltimestamp]))
        image_timestamps.append(imagelist[timestamp][:-4])

    logger.info('Lidar timestamps: ', len(lidar_timestamps))
    logger.info('Image timestamps: ', len(image_timestamps))
    
    if seqname == 'city_day2': ## There are some problems in the raw gps reading
        lidar_poses = []
        removed_indices = []
        for index, t_lidar in enumerate(lidar_timestamps):
            upper_index = bisect.bisect(gps_timestamps, t_lidar)
            diff = t_lidar - gps_timestamps[upper_index - 1]
            # logger.info(diff)
            fraction = (t_lidar - gps_timestamps[upper_index - 1]) / \
                        (gps_timestamps[upper_index] - gps_timestamps[upper_index - 1])
                    
            pos_lower = gps_poses[upper_index - 1, :]
            pos_upper = gps_poses[upper_index, :]
            pos_interp = (1 - fraction) * pos_lower + fraction * pos_upper
            lidar_poses.append(pos_interp)
            if diff > 0.05:
                logger.info("GPS OFF...")
                # lidar_timestamps.pop(index)
                # image_timestamps.pop(index)
                # lidarlist.pop(index)
                removed_indices.append(index)
                
    else:
        lidar_poses = interpolate_poses(gps_poses, gps_timestamps, lidar_timestamps)
        
    # Save poses
    if save_dir is None:
        save_dir = datapath
    with open(os.path.join(save_dir, f'annotation_{seqname}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'date', 'northing', 'easting', 'yaw', 'submap_path', 'img_path'])
        for i in range(len(lidar_timestamps)):
            if seqname == 'city_day2':
                if i not in removed_indices:
                    data = [lidarlist[i][:-4], datapath.split('/')[-1], lidar_poses[i][0], lidar_poses[i][1], 0.0,
                            os.path.join(lidarpath, lidarlist[i]), os.path.join(imagepath, image_timestamps[i] + '.png')]
                    writer.writerow(data)
            else:
                data = [lidarlist[i][:-4], datapath.split('/')[-1], lidar_poses[i][0], lidar_poses[i][1], 0.0,
                        os.path.join(lidarpath, lidarlist[i]), os.path.join(imagepath, image_timestamps[i] + '.png')]
                writer.writerow(data)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate annotations for Vivid++ dataset")
    argparser.add_argument("--datapath", help="Path to data folder")
    opt = argparser.parse_args()
    main(opt.datapath)