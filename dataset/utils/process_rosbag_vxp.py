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

# Adopted from the authors of ViViD++ dataset https://visibilitydataset.github.io/4_download.html

import rosbag
import rospy
import ros_numpy
import sys
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
from PIL import Image
from pyproj import Proj, transform
import pymap3d as pm
import argparse
import logging
from tqdm import tqdm
## global variables
imgno = 0
lidarno = 0
imglist = []
lidarlist = []
cvbdg = CvBridge()
proj_UTMK = Proj(init='epsg:5178')
proj_WGS84 = Proj(init='epsg:4326')
initpt = np.array([988118.672713562, 1818806.4875518726]) ## origin of campus, just for convenience.

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] [%(name)s] %(asctime)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def write_gps(msg, f_gps):
    global proj_UTMK, proj_WGS84
    tnow = msg.header.stamp.to_sec()
    lat = msg.latitude
    lon = msg.longitude
    alt = msg.altitude
    long_0, lat_0 = transform(proj_UTMK,proj_WGS84,initpt[0],initpt[1])
    e, n, u = pm.geodetic2enu(lat, lon, alt, lat_0, long_0, 0)
    f_gps.write('%.6f, %.6f, %.6f, %.6f\n'%(e, n, u, tnow))

def write_images(msg, folder, f):
    global imgno, imglist, cvbdg
    imgno += 1
    tnow = msg.header.stamp.to_sec()
    cv_img = cvbdg.imgmsg_to_cv2(msg,desired_encoding="passthrough")

    DIM = (1280, 1024)
    K_1 = np.array([[702.6030497884977, 0., 644.9296487349911], [0., 703.4541726858521, 526.4572414665469], [0., 0., 1.]])
    D_1 = np.array([[0.5424763304847061, -1.7515099175022195, 6.050489512760127,-5.786170959900578]])
    map1_1, map2_1 = cv2.fisheye.initUndistortRectifyMap(K_1, D_1, np.eye(3), K_1, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(cv_img, map1_1, map2_1, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imwrite(os.path.join(folder,'%.6f.png'%(tnow)), undistorted_img)
    imglist.append([str(imgno) + "%06i.png"%(imgno)])
    f.write("%12.12f, %06i\n"%(tnow,imgno))

def write_ptcld(msg, folder, f):
    global lidarno, proj_UTMK, proj_WGS84
    lidarno += 1

    tnow = msg.header.stamp.to_sec()
    pts = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
    pts = pts.reshape(-1,3)
    pts = pts[~np.isnan(pts).any(axis=1)]
    pts = pts[~np.isinf(pts).any(axis=1)]

    np.save(os.path.join(folder,'%.6f.npy'%(tnow)), pts)
    lidarlist.append([str(lidarno) + "%06i.npy"%(lidarno)])
    f.write("%12.12f, %06i\n"%(tnow,lidarno))

def raw_to_kelvin(val):
    return ( 1428.0 / np.log( 408825.0 / ( val + 58.417 ) + 1.0 ) )

def main(input_dir, seqname, folder):
    bag = rosbag.Bag(os.path.join(input_dir, seqname + '.bag'))
    os.makedirs(folder, exist_ok=True)
    f_gps = open(os.path.join(folder,'gpslist_enu.txt'),'w')
    ## write images with timestamp assignment
    for _, msg, _ in tqdm(bag.read_messages(topics=['/gps'])):
        write_gps(msg, f_gps)
    f_gps.close()

    img_folder = os.path.join(folder, "images")
    os.makedirs(img_folder, exist_ok=True)
    f = open(os.path.join(folder,'imglist.txt'),'w')
    for _, msg, _ in tqdm(bag.read_messages(topics=['/camera/image_color'])):
        write_images(msg, img_folder, f)
    f.close()

    lidar_folder = os.path.join(folder, "lidar")
    os.makedirs(lidar_folder, exist_ok=True)
    f_lidar = open(os.path.join(folder,'lidarlist.txt'),'w')
    for _, msg, _ in tqdm(bag.read_messages(topics=['/os1_cloud_node/points'])):
        write_ptcld(msg, lidar_folder, f_lidar)
    f_lidar.close()

    logger.info('saved '+ str(imgno) + ' images timstamps to detections.')

    bag.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process ROS bag files')
    parser.add_argument('--data', required=True, help='Input directory containing ROS bag files')
    parser.add_argument('--seq', required=True, help='Sequence name')
    parser.add_argument('--output', required=True, help='Output directory')

    args = parser.parse_args()
    main(args.data, args.seq, args.output)

