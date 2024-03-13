#
# Created on Wed May 3 2023
# The MIT License (MIT)
# Copyright (c) 2023 Yun-Jin Li (Jim)
# Part of the code refers to https://github.com/mikacuy/pointnetvlad
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
from datetime import datetime
import pandas as pd
import os
from sklearn.neighbors import KDTree
import math
from tqdm import tqdm
import yaml
from yaml.loader import FullLoader
import pickle
import numpy as np
import random
import psutil
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter(
    '[%(levelname)s] [%(name)s] %(asctime)s: %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


def load_setup_file(path):
    with open(path) as f:
        data = yaml.load(f, Loader=FullLoader)
    return data


def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if (point[0] - x_width < northing < point[0] + x_width and
                point[1] - y_width < easting < point[1] + y_width):
            in_test_set = True
            break
    return in_test_set


def construct_query_dict_new(df_centroids, filename, positive_th):
    '''
    Since the old way constructing the training tuple is quite bruto-force,
    we are afraid that if might have bad impact on the training, therefore, we
    refer to the way how Cattaneo's way of doing this.
    '''
    # Generate tree that each neighbors are within 5 meter
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=5)
    assert len(df_centroids) == len(ind_nn)
    # queries_online = {}
    # the index is the row_id and the value is the label
    labels = np.zeros(len(ind_nn), dtype=np.int32)
    has_label = []  # Which row from the original annotation already has label
    label = 1
    label_centroid = []

    for i in tqdm(range(len(ind_nn))):
        query = df_centroids.iloc[i]["timestamp"]
        run = df_centroids.iloc[i]["date"]
        # Apart from only considering purely on distance,
        # here we also consider the yaw angle,
        # and only accept yaw difference liying in 0.5 rad / 28.66 degree
        positives = np.setdiff1d(ind_nn[i], [i]).tolist()
        positives_yaw = []
        for pos in positives:
            diff = math.fabs(
                df_centroids.iloc[i]['yaw'] - df_centroids.iloc[pos]['yaw'])
            while diff >= 2 * math.pi:
                diff -= 2*math.pi
            if diff >= math.pi:
                diff -= 2 * math.pi
            if math.fabs(diff) < 0.5:
                positives_yaw.append(pos)
        positives_yaw = np.array(positives_yaw)
        # queries_online[i] = {"query_img": df_centroids.iloc[i]['img_path'], "query_submap": df_centroids.iloc[i]['submap_path'], "date": run, "timestamp": query}

        # If row i doesn't have label yet, we make row i into a
        # label_centroid and give row i a label id, also give all
        # of its neighbors the same label id
        if i not in has_label:
            label_centroid.append([df_centroids.iloc[i]['northing'],
                                   df_centroids.iloc[i]['easting'],
                                   df_centroids.iloc[i]['yaw']])
            labels[i] = label
            has_label.append(i)
            for j in positives_yaw:
                if j not in has_label:
                    labels[j] = label
                    has_label.append(j)
            label += 1

    logger.info(f"Number of labels: {label-1}")
    # Some of the label id might only contains a few sample,
    # we have to delete them

    remove_label = []
    new_centroids = []
    # Iterate through label id
    for i in range(1, label):
        # if label id only has samples less than 5,
        # we put this id into remove buffer
        if (labels == i).sum() < 5:
            remove_label.append(i)
            logger.info(
                f"There are only {(labels == i).sum()} < 5 samples in label {i}, put into the remove label buffer")
        # Otherwise, we put into a new label centroid
        else:
            new_centroids.append(label_centroid[i])

    # new_labels = []
    # the index is the row_id and the value is the label
    new_labels = np.zeros(len(ind_nn), dtype=np.int32)
    queries_new = {}
    # queries_new_online = {}
    # count = 0
    df_new = pd.DataFrame({})
    for i in range(len(labels)):
        if labels[i] in remove_label:
            pass
            # _ = queries_online.pop(i)
        else:
            # df_new = df_new.append(df_centroids.iloc[i], ignore_index=True)
            new_row = pd.DataFrame([{'timestamp': df_centroids.iloc[i]['timestamp'], 'date': df_centroids.iloc[i]['date'], 'northing': df_centroids.iloc[i]['northing'],
                                   'easting': df_centroids.iloc[i]['easting'], 'yaw': df_centroids.iloc[i]['yaw'], 'submap_path': df_centroids.iloc[i]['submap_path'], 'img_path': df_centroids.iloc[i]['img_path']}])
            df_new = pd.concat([df_new, new_row], ignore_index=True)
            # new_labels.append(labels[i])
            new_labels[i] = labels[i]
            # count += 1
    # new_labels = np.array(new_labels)
    # new_labels serves as a map where the index = row id and value = label (1, ...)
    assert new_labels.shape[0] == len(df_centroids)
    # tree_new = KDTree(df_new[['northing', 'easting']])
    # ind_nn = tree_new.query_radius(df_new[['northing', 'easting']], r=5)
    # for i in tqdm(range(len(ind_nn))):
    #     query = int(df_new.iloc[i]["timestamp"])
    #     run = df_new.iloc[i]["date"]
    #     positives = np.setdiff1d(ind_nn[i], [i]).tolist()
    #     positives_yaw = []
    #     for pos in positives:
    #         diff = math.fabs(df_new.iloc[i]['yaw'] - df_new.iloc[pos]['yaw'])
    #         while diff >= 2 * math.pi:
    #             diff -= 2 * math.pi
    #         if diff >= math.pi:
    #             diff -= 2 * math.pi
    #         if math.fabs(diff) < 0.5:
    #             positives_yaw.append(pos)
    #     positives_yaw = np.array(positives_yaw)
    #     # queries_new_online[i] = {"query_img": df_new.iloc[i]['img_path'], "query_submap": df_new.iloc[i]['submap_path'], "date": run, "timestamp": query}

    # Store the valid label := Label that has more than 5 samples
    valid_labels = set(range(1, label)) - set(remove_label)
    valid_labels = np.array(list(valid_labels))
    valid_labels.sort()
    assert valid_labels.shape[0] == len(new_centroids)
    logger.info(f"Generate {valid_labels.shape[0]} labels")
    new_centroids = np.array(new_centroids)
    centroid_tree = KDTree(new_centroids[:, 0:2])
    close_labels = centroid_tree.query_radius(
        new_centroids[:, 0:2], r=positive_th)
    ignore_in_training = {}
    for i in range(len(close_labels)):
        positives = np.setdiff1d(close_labels[i], [i]).tolist()
        close_labels_yaw = []
        for pos in positives:
            diff = math.fabs(new_centroids[i][2] - new_centroids[pos][2])
            while diff >= 2 * math.pi:
                diff -= 2*math.pi
            if diff >= math.pi:
                diff -= 2 * math.pi
            if math.fabs(diff) < 0.5:
                close_labels_yaw.append(valid_labels[pos])
        close_labels_yaw = np.array(close_labels_yaw)
        ignore_in_training[valid_labels[i]] = close_labels_yaw

    # queries_new['base_folder'] = runs_folder
    # queries_new['pc_folder'] = pointcloud_fols
    # queries_new_online['base_folder'] = runs_folder
    # queries_new_online['pc_folder'] = pointcloud_fols

    # with open(filename, 'wb') as handle:
    #    pickle.dump(queries_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # online_filename = filename.replace('.pickle', '_online.pickle')

    # with open(online_filename, 'wb') as handle:
    #     pickle.dump(queries_new_online, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # logger.info(f"Query has been saved to {os.path.abspath(online_filename)}")

    label_filename = filename.replace('.pickle', '_label.pickle')
    with open(label_filename, 'wb') as handle:
        label_dict = {'valid_labels': valid_labels, 'labels': new_labels,
                      'ignore_in_training': ignore_in_training}
        pickle.dump(label_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(
        f"Label file has been saved to {os.path.abspath(label_filename)}")


def construct_query_dict(df_centroids, filename, positive_th, negative_th, yaw=False):
    logger.info(f"{df_centroids[['northing','easting']]}")
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(
        df_centroids[['northing', 'easting']], r=positive_th)
    ind_r = tree.query_radius(
        df_centroids[['northing', 'easting']], r=negative_th)

    queries = {}
    logger.info(f"{len(ind_nn)} places in total")
    for i in range(len(ind_nn)):
        logger.info(
            f'Append the {i}th place in the pickle file (Used {psutil.Process().memory_info().rss / (1024 * 1024)} Mb)')
        query_img = df_centroids.iloc[i]["img_path"]
        query_submap = df_centroids.iloc[i]["submap_path"]
        positives = np.setdiff1d(ind_nn[i], [i]).tolist()
        negatives = np.setdiff1d(
            df_centroids.index.values.tolist(), ind_r[i]).tolist()
        random.shuffle(negatives)
        if yaw:
            logger.info(f"Positives before processing yaw: {positives}")
            positives_yaw = []
            for pos in positives:
                diff = math.fabs(
                    df_centroids.iloc[i]['yaw'] - df_centroids.iloc[pos]['yaw'])
                while diff >= 2 * math.pi:
                    diff -= 2*math.pi
                if diff >= math.pi:
                    diff -= 2 * math.pi
                if math.fabs(diff) < 0.5:
                    positives_yaw.append(pos)
            if len(positives_yaw) == 0:
                logger.info(
                    f"There's no positive sample for this place {i}, make sure to pay attention to it.")
            positives = np.array(positives_yaw)
        else:
            if len(positives) == 0:
                logger.info(
                    f"There's no positive sample for this place {i}, make sure to pay attention to it.")
            positives = np.array(positives)
        queries[i] = {"query_img": query_img, "query_submap": query_submap,
                      "positives": positives, "negatives": negatives}

    logger.info(f"Saving the pickle file")
    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Done {filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate training / validation tuples for Oxford RobotCar dataset')
    parser.add_argument('--config', type=str,
                        default="config/setup_generate_tuple.yml")
    parser.add_argument('--mode', type=str, default='bruto-force',
                        help='Mode', choices=['bruto-force', 'label'])
    args = parser.parse_args()

    setup = load_setup_file(args.config)
    annotation = pd.read_csv(setup['parameter']['annotation'])
    # print(annotation[['northing', 'easting']])

    log_dir = os.path.join(setup['parameter']['save_dir'], "log")
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    logging.basicConfig(format='[%(levelname)s] [%(name)s] %(asctime)s: %(message)s',
                        filename=f'{os.path.join(log_dir, f"generate_tuple_{current_time}")}.log', level=logging.INFO)

    p = [setup['parameter']['p1'], setup['parameter']['p2'],
         setup['parameter']['p3'], setup['parameter']['p4']]

    logger.info(f"{len(annotation['timestamp'])} samples in total")

    df_train = pd.DataFrame({})
    df_test = pd.DataFrame({})
    # df_train = None
    # df_test = None
    removed_sequences = setup['parameter']['removed_sequences']

    for index, row in annotation.iterrows():
        if row['date'] not in removed_sequences:
            if (check_in_test_set(row['northing'], row['easting'], p, setup['parameter']['x_width'], setup['parameter']['y_width'])):
                # df_test=df_test.append(row, ignore_index=True)
                logger.info(
                    f'{index} row (Used {psutil.Process().memory_info().rss / (1024 * 1024)} Mb): TEST')
                # if df_test is None:
                #     df_test = row
                # else:
                #     df_test = pd.concat([df_test, row], ignore_index=True)
                new_row = pd.DataFrame([{'timestamp': row['timestamp'], 'date': row['date'], 'northing': row['northing'],
                                       'easting': row['easting'], 'yaw': row['yaw'], 'submap_path': row['submap_path'], 'img_path': row['img_path']}])
                # print(new_row)
                df_test = pd.concat([df_test, new_row], ignore_index=True)
                # print(df_test[['northing', 'easting']])
            else:
                # df_train=df_train.append(row, ignore_index=True)
                logger.info(
                    f'{index} row (Used {psutil.Process().memory_info().rss / (1024 * 1024)} Mb): Train')
                # if df_train is None:
                #     df_train = row
                # else:
                #     df_train = pd.concat([df_train, row], ignore_index=True)
                new_row = pd.DataFrame([{'timestamp': row['timestamp'], 'date': row['date'], 'northing': row['northing'],
                                       'easting': row['easting'], 'yaw': row['yaw'], 'submap_path': row['submap_path'], 'img_path': row['img_path']}])
                df_train = pd.concat([df_train, new_row], ignore_index=True)
                # print(df_train[['northing, easting']])

    logger.info(
        f"Number of training submaps: {str(len(df_train['timestamp']))}")
    logger.info(
        f"Number of non-disjoint test submaps: {str(len(df_test['timestamp']))}")
    pos_th = setup['parameter']['pos_th']
    neg_th = setup['parameter']['neg_th']
    if setup['parameter']['yaw']:
        yaw = '_yaw'
    else:
        yaw = ''
    if setup['parameter']['name'] is not None:
        tuple_name = setup['parameter']['name'] + '_'
    else:
        tuple_name = ''
    if args.mode.lower() == 'bruto-force':
        construct_query_dict(df_train, os.path.join(
            setup['parameter']['save_dir'], f"{tuple_name}training_queries_baseline_p{pos_th}_n{neg_th}{yaw}.pickle"), pos_th, neg_th, yaw=setup['parameter']['yaw'])
        construct_query_dict(df_test, os.path.join(
            setup['parameter']['save_dir'], f"{tuple_name}test_queries_baseline_p{pos_th}_n{neg_th}{yaw}.pickle"), pos_th, neg_th, yaw=setup['parameter']['yaw'])
    elif args.mode.lower() == 'label':
        construct_query_dict_new(df_train, os.path.join(
            setup['parameter']['save_dir'], f"{tuple_name}training_queries_p{pos_th}.pickle"), pos_th)
        construct_query_dict_new(df_test, os.path.join(
            setup['parameter']['save_dir'], f"{tuple_name}test_queries_p{pos_th}.pickle"), pos_th)
