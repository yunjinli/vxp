#
# Created on Wed May 25 2023
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
formatter = logging.Formatter('[%(levelname)s] [%(name)s] %(asctime)s: %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

def load_setup_file(path):
    with open(path) as f:
        data = yaml.load(f, Loader=FullLoader)
    return data

def check_in_test_set(seq, test_seq = ['campus_day2', 'city_day2']):
    if seq in test_seq:
        return True
    else:
        return False
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate training / test tuples')
    parser.add_argument('--config', type=str, default="config/setup_generate_tuple_vivid.yml")

    args = parser.parse_args()

    setup = load_setup_file(args.config)
    annotation = pd.read_csv(setup['parameter']['annotation'])
    # print(annotation[['northing', 'easting']])
    
    log_dir = os.path.join(setup['parameter']['save_dir'], "log")
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    logging.basicConfig(format='[%(levelname)s] [%(name)s] %(asctime)s: %(message)s', filename=f'{os.path.join(log_dir, f"generate_tuple_{current_time}")}.log', level=logging.INFO)

    logger.info(f"{len(annotation['timestamp'])} samples in total")

    df_train = pd.DataFrame({})
    df_test = pd.DataFrame({})
    # df_train = None
    # df_test = None
    img_submap_training = []
    img_submap_testing = []
    for index, row in annotation.iterrows():
        if 'day' in row['date']:
            if (check_in_test_set(row['date'])):
                logger.info(f'{index} row (Used {psutil.Process().memory_info().rss / (1024 * 1024)} Mb): TEST')
                img_submap_testing.append(index)
            else:
                logger.info(f'{index} row (Used {psutil.Process().memory_info().rss / (1024 * 1024)} Mb): Train')
                img_submap_training.append(index)

    logger.info(f"Number of training tuples: {len(img_submap_training)}")
    logger.info(f"Number of non-disjoint test tuples: {len(img_submap_testing)}")
    pos_th = setup['parameter']['pos_th']
    neg_th = setup['parameter']['neg_th']
    
    if setup['parameter']['name'] is not None:
        tuple_name = setup['parameter']['name'] + '_'
    else:
        tuple_name = ''
    with open(os.path.join(setup['parameter']['save_dir'], f"{tuple_name}training_pair_3d_student_p{pos_th}_n{neg_th}.pickle"), 'wb') as f:
        pickle.dump(img_submap_training, f)
    with open(os.path.join(setup['parameter']['save_dir'], f"{tuple_name}testing_pair_3d_student_p{pos_th}_n{neg_th}.pickle"), 'wb') as f:
        pickle.dump(img_submap_testing, f)