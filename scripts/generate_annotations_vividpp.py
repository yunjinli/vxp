import argparse
import os
import csv
import numpy as np
import bisect

def interpolate_poses(gps_poses, gps_timestamps, requested_timestamps):
    upper_ids = np.array([bisect.bisect(gps_timestamps, pt)
                 for pt in requested_timestamps])
    lower_ids = np.array([u - 1 for u in upper_ids])

    fractions = (requested_timestamps - gps_timestamps[lower_ids]) / \
                (gps_timestamps[upper_ids] - gps_timestamps[lower_ids])
    fractions = np.expand_dims(fractions, axis=1)

    pos_lower = gps_poses[lower_ids, :]
    pos_upper = gps_poses[upper_ids, :]
    pos_interp = np.multiply(np.tile(1 - fractions, (1, 3)), pos_lower) + \
        np.multiply(np.tile(fractions, (1, 3)), pos_upper)

    return pos_interp

def main(args):
    datapath = args.datapath
    imagepath = os.path.join(datapath, 'images')
    lidarpath = os.path.join(datapath, 'lidar')
    gpspath = os.path.join(datapath, 'gpslist_enu.txt')

    imagelist = sorted(os.listdir(imagepath))
    lidarlist =  sorted(os.listdir(lidarpath))

    print('Image files: ', len(imagelist))
    print('Lidar files: ', len(lidarlist))

    gps_poses = []
    gps_timestamps = []
    with open(gpspath, 'r') as f:
        for line in f:
            gps_timestamps.append(float(line.split(',')[-1]))
            gps_poses.append([float(x) for x in line.split(',')[:3]])
    gps_poses = np.array(gps_poses)
    gps_timestamps = np.array(gps_timestamps)
    print('GPS poses: ', gps_poses.shape)

    # Interpolate GPS poses for each lidar frame
    lidar_timestamps = []
    image_timestamps = []
    for ltimestamp in range(len(lidarlist)):
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

    print('Lidar timestamps: ', len(lidar_timestamps))
    print('Image timestamps: ', len(image_timestamps))

    lidar_poses = interpolate_poses(gps_poses, gps_timestamps, lidar_timestamps)

    # Save poses
    with open(os.path.join(datapath, 'annotation.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'date', 'northing', 'easting', 'yaw', 'submap_path', 'img_path'])
        for i in range(len(lidar_timestamps)):
            data = [lidarlist[i][:-4], datapath.split('/')[-1], lidar_poses[i][0], lidar_poses[i][1], 0.0,
                    os.path.join(lidarpath, lidarlist[i]), os.path.join(imagepath, image_timestamps[i] + '.png')]
            writer.writerow(data)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate annotations for Vivid++ dataset")
    argparser.add_argument("--datapath", help="Path to data folder")
    opt = argparser.parse_args()
    main(opt)