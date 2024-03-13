#
# Created on Thu Nov 23 2023
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
from tqdm import tqdm
import pandas as pd
from transform import *
import pykitti
import os
# 00: 2011_10_03_drive_0027 000000 004540
# 01: 2011_10_03_drive_0042 000000 001100
# 02: 2011_10_03_drive_0034 000000 004660
# 03: 2011_09_26_drive_0067 000000 000800
# 04: 2011_09_30_drive_0016 000000 000270
# 05: 2011_09_30_drive_0018 000000 002760
# 06: 2011_09_30_drive_0020 000000 001100
# 07: 2011_09_30_drive_0027 000000 001100
# 08: 2011_09_30_drive_0028 001100 005170
# 09: 2011_09_30_drive_0033 000000 001590
# 10: 2011_09_30_drive_0034 000000 001200

# pose_base_dir = '/storage/group/dataset_mirrors/01_incoming/kitti_data'
# save_dir = '/storage/user/lyun/kitti'

dates = ['2011_10_03', '2011_10_03', '2011_10_03', '2011_09_26' ,'2011_09_30' ,'2011_09_30' ,'2011_09_30' ,'2011_09_30' ,'2011_09_30' ,'2011_09_30' ,'2011_09_30']
drives = ['0027', '0042', '0034', '0067', '0016', '0018', '0020', '0027', '0028', '0033', '0034']
assert len(dates) == len(drives)
seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
ranges = [
    (0, 4540),
    (0, 1100),
    (0, 4660),
    (0, 800),
    (0, 270),
    (0, 2760),
    (0, 1100),
    (0, 1100),
    (1100, 5170),
    (0, 1590),
    (0, 1200),
]
assert len(seqs) == len(ranges)

def parse_sequence(seq_id, pose_base_dir, date, drive, image_base_dir, lidar_base_dir, timestamp_dir, pose_range):
    df = pd.DataFrame({})
    
    img_names = []
    for _, _, files in os.walk(image_base_dir):
        for filename in sorted(files):
            img_names.append(filename)
        break
    print(f"There are {len(img_names)} images")
    lidar_names = []
    for _, _, files in os.walk(lidar_base_dir):
        for filename in sorted(files):
            lidar_names.append(filename)
        break
    print(f"There are {len(lidar_names)} LiDAR scans")
    
    data = pykitti.raw(pose_base_dir, date, drive)
    print(f"There are {len(data.oxts)} poses")
    
    # with open(pose_path, 'r') as file:
    timestamp_file = open(timestamp_dir, 'r')
    timestamps = timestamp_file.readlines()
    # m = file.readlines()
    # print(len(m))
    # for i, line in tqdm(enumerate(m)):
    counter = 0
    print(f"Parse poses from {pose_range[0]} to {pose_range[1]}")
    for i in tqdm(range(pose_range[0], pose_range[1] + 1)):
        # se3 = np.array(line.split(' '), dtype=np.float32).reshape(3, 4)
        # se3 = np.concatenate([se3, np.array([[0, 0, 0, 1]])], axis=0)
        xyzrpy = se3_to_components(se3=data.oxts[i].T_w_imu)
        # print(xyzrpy)
        t = (float)(timestamps[i-pose_range[0]].split('\n')[0])
        
        new_row = pd.DataFrame([{'timestamp': t, 
                                 'date': seq_id,
                                 'northing': xyzrpy[0], 
                                 'easting': xyzrpy[1], 
                                 'yaw': xyzrpy[5], 
                                 'submap_path': os.path.join(lidar_base_dir, lidar_names[i-pose_range[0]]), 
                                 'img_path': os.path.join(image_base_dir, img_names[i-pose_range[0]])}])
        df = pd.concat([df, new_row], ignore_index=True)
        counter += 1
            # print(se3.shape)
    assert counter == len(img_names)
    assert counter == len(lidar_names)
    return df
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate KITTI dataset')
    parser.add_argument('--kitti_odom_color', type=str, default="/storage/group/dataset_mirrors/01_incoming/kitti_odom_color")
    parser.add_argument('--SemanticKitti', type=str, default="/storage/group/dataset_mirrors/01_incoming/SemanticKitti")
    parser.add_argument('--pose_base_dir', type=str, default="/storage/group/dataset_mirrors/01_incoming/kitti_data")
    parser.add_argument('--save_dir', type=str, default="/storage/user/lyun/kitti")
    args = parser.parse_args()
    for seq, date, drive, r in zip(seqs, dates, drives, ranges):
        try:
            df = parse_sequence(seq_id=seq, 
                                pose_base_dir=args.pose_base_dir, 
                                date=date, 
                                drive=drive,
                                image_base_dir=os.path.join(args.kitti_odom_color, f'sequences/{seq}/image_2'), 
                                lidar_base_dir=os.path.join(args.SemanticKitti, f'sequences/{seq}/velodyne'),
                                timestamp_dir=os.path.join(args.kitti_odom_color, f'sequences/{seq}/times.txt'),
                                pose_range=r
                                )
            df.to_csv(os.path.join(args.save_dir, f'{seq}.csv'))
        except Exception as e:
            print(f"[ERROR] Parse {seq} failed, catch exception {e}")
            
    all_annotation = pd.DataFrame({})
    counter = 0
    # save_dir = '/storage/user/lyun/kitti'
    sub_annotations_path = []
    for _, _, files in os.walk(args.save_dir):
        for filename in sorted(files):
            sub_annotations_path.append(os.path.join(args.save_dir, filename))
        break
    for sub_annotation_path in sub_annotations_path:
        try:
            sub_annotation = pd.read_csv(sub_annotation_path)
            all_annotation = pd.concat([all_annotation, sub_annotation], ignore_index=True)
            counter += 1
        except:
            print(f"Cannot find sub annotation file from directory: {sub_annotation_path}")
    all_annotation.to_csv(os.path.join(args.save_dir, 'all_annotation.csv'))
        
        