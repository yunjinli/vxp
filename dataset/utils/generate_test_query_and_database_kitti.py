#
# Created on Thu Nov 23 2023
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
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random
import psutil
from yaml.loader import FullLoader
import yaml
import math

process = psutil.Process()

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

def check_in_test_set_region(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if (point[0] - x_width < northing < point[0] + x_width and
                point[1] - y_width < easting < point[1] + y_width):
            in_test_set = True
            break
    return in_test_set

def output_to_file(output, filename):
	with open(filename, 'wb') as handle:
		pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
	logger.info(f"Done {filename}")
 
def construct_query_and_database_sets(annotation_dir, folders, th, yaw, save_dir, length, test_region=None, name=None):
	# query_name = query_base_dir.split('/')[-1]
	# db_name = db_base_dir.split('/')[-1]
	database_trees = []
	test_trees = []

	test_sets = []
	database_sets = []

 	# df_train = pd.DataFrame({})
    # df_test = pd.DataFrame({})
	annotation = pd.read_csv(annotation_dir)	
	for folder in folders:
		df_database = pd.DataFrame({})
		df_test = pd.DataFrame({})
		database = {}
		test = {} 
		df_q = pd.DataFrame({})
		df_db = pd.DataFrame({})
		
		# delta_sum = 0.0
		delta_sum_q = 5
		delta_sum_db = 0.0
		# interval = 20.0
		interval_q = 20.0
		interval_db = 20.0
		for index, row in annotation.iterrows():
			if row['date'] == folder:
				new_row = pd.DataFrame([{'timestamp': row['timestamp'], 'date': row['date'], 'northing': row['northing'],
										'easting': row['easting'], 'yaw': row['yaw'], 'submap_path': row['submap_path'], 'img_path': row['img_path']}])
				# if delta_sum < interval and index != 0:
				# 	delta_x = annotation.iloc[index]['northing'] - annotation.iloc[index - 1]['northing'] 
				# 	delta_x **= 2
				# 	delta_y = annotation.iloc[index]['easting'] - annotation.iloc[index - 1]['easting']
				# 	delta_y **= 2
				# 	try:
				# 		delta_sum += math.sqrt(delta_x + delta_y)
				# 	except:
				# 		logger.warn(
				# 			f'Compute square root fail: {delta_x} + {delta_y} = {delta_x + delta_y}')
				# 	continue
				# delta_sum = 0.0
				# df_q = pd.concat([df_q, new_row], ignore_index=True)
				# df_db = pd.concat([df_db, new_row], ignore_index=True)
				if delta_sum_q < interval_q and index != 0:
					delta_x = annotation.iloc[index]['northing'] - annotation.iloc[index - 1]['northing'] 
					delta_x **= 2
					delta_y = annotation.iloc[index]['easting'] - annotation.iloc[index - 1]['easting']
					delta_y **= 2
					try:
						delta_sum_q += math.sqrt(delta_x + delta_y)
					except:
						logger.warn(
							f'Compute square root fail: {delta_x} + {delta_y} = {delta_x + delta_y}')
				else:
					delta_sum_q = 0.0
					df_q = pd.concat([df_q, new_row], ignore_index=True)
     
				if delta_sum_db < interval_db and index != 0:
					delta_x = annotation.iloc[index]['northing'] - annotation.iloc[index - 1]['northing'] 
					delta_x **= 2
					delta_y = annotation.iloc[index]['easting'] - annotation.iloc[index - 1]['easting']
					delta_y **= 2
					try:
						delta_sum_db += math.sqrt(delta_x + delta_y)
					except:
						logger.warn(
							f'Compute square root fail: {delta_x} + {delta_y} = {delta_x + delta_y}')
				else:
					delta_sum_db = 0.0
					df_db = pd.concat([df_db, new_row], ignore_index=True)
				
		logger.info(f"Dataset size: {len(df_q)}")
		logger.info(f"Dataset size: {len(df_db)}")
		for index, row in df_q.iterrows():
			if(check_in_test_set_region(row['northing'], row['easting'], test_region['p'][row['date']], test_region['width'], test_region['width'])):
				# if(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
				new_row = pd.DataFrame([{'timestamp': row['timestamp'], 'date': row['date'], 'northing': row['northing'], 'easting': row['easting'], 'yaw': row['yaw'], 'submap_path': row['submap_path'], 'img_path': row['img_path']}])
				df_test = pd.concat([df_test, new_row], ignore_index=True)
				test[len(test.keys())] = {'submap_path': row['submap_path'], 'img_path': row['img_path'], 'northing': row['northing'], 'easting': row['easting'], 'yaw': row['yaw'], 'timestamp': row['timestamp']}

		for index, row in df_db.iterrows():
			new_row = pd.DataFrame([{'timestamp': row['timestamp'], 'date': row['date'], 'northing': row['northing'], 'easting': row['easting'], 'yaw': row['yaw'], 'submap_path': row['submap_path'], 'img_path': row['img_path']}])
			df_database = pd.concat([df_database, new_row], ignore_index=True)
			database[len(database.keys())] = {'submap_path': row['submap_path'], 'img_path': row['img_path'], 'northing': row['northing'], 'easting': row['easting'], 'yaw': row['yaw'], 'timestamp': row['timestamp']}
		
		database_sets.append(database)
		test_sets.append(test)	

		database_tree = KDTree(df_database[['northing', 'easting']])
		test_tree = KDTree(df_test[['northing', 'easting']])
		database_trees.append(database_tree)
		test_trees.append(test_tree)	

	for i in range(len(database_sets)):
		tree = database_trees[i]
		for j in range(len(test_sets)):
			# if(i==j):
			# 	continue
			for key in range(len(test_sets[j].keys())):
				logger.info(f"Current frame {key}")
				coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
				index = tree.query_radius(coor, r=th)
				positives = index[0].tolist()
				if yaw:
					positives_yaw = []
					for pos in positives:
						diff = math.fabs(test_sets[j][key]['yaw'] - database_sets[i][pos]['yaw'])
						while diff >= 2 * math.pi:
							diff -= 2*math.pi
						if diff >= math.pi:
							diff -= 2 * math.pi
						if math.fabs(diff) < 0.5:
							positives_yaw.append(pos)
					# positives = np.array(positives_yaw)
					positives = positives_yaw
				logger.info(f"Positive before removing recent frames {positives}")
				## Filter the recent frames
				positives_only_revisit = []
				print(f"Timestamp: {test_sets[j][key]['timestamp']}")
				for pos in positives:
					# if math.fabs(pos - key) != 0:
					# 	positives_rm_recent.append(pos)
					# diff = np.linalg.norm(coor - np.array([database_sets[i][pos]['northing'], database_sets[i][pos]['easting']]))
					# if diff == 0:
					# 	logger.warning("Remove same place from the database")
					# else:
					# 	positives_only_revisit.append(pos)
					if math.fabs(test_sets[j][key]["timestamp"] - database_sets[i][pos]['timestamp']) > 10 and test_sets[j][key]["timestamp"] > database_sets[i][pos]['timestamp']:
					# if math.fabs(test_sets[j][key]["timestamp"] - database_sets[i][pos]['timestamp']) > 10:
					# if test_sets[j][key]["timestamp"] > database_sets[i][pos]['timestamp']:
						positives_only_revisit.append(pos)
					else:
						print(f"Remove {pos} ({database_sets[i][pos]['timestamp']}) ")
				positives = positives_only_revisit
				logger.info(f"Positive after removing recent frames {positives}")
    
				## Fillter the future frames
				# positives_rm_future = []
				# for pos in positives:
				# 	if key > pos:
				# 		positives_rm_future.append(pos)
				# positives = positives_rm_future
				# logger.info(f"Positive after removing future frames {positives}")
				#indices of the positive matches in database i of each query (key) in test set j
				test_sets[j][key][i] = positives

	logger.info(f"Memory used: {process.memory_info().rss / (1024 * 1024 * 1024)} GB")
 
	if yaw:
		yaw = '_yaw'
	else:
		yaw = ''
	if name is None:
		output_to_file(database_sets, os.path.join(save_dir, f'evaluation_database_{th}m{yaw}_{length}runs.pickle'))
		output_to_file(test_sets, os.path.join(save_dir, f'evaluation_query_{th}m{yaw}_{length}runs.pickle'))
	else:
		output_to_file(database_sets, os.path.join(save_dir, f'{name}_evaluation_database_th{th}m{yaw}_{length}runs.pickle'))
		output_to_file(test_sets, os.path.join(save_dir, f'{name}_evaluation_query_th{th}m{yaw}_{length}runs.pickle'))
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate inference test tuples')
    parser.add_argument('--config', type=str, default="config/setup_generate_test_query_db_kitti.yml")

    args = parser.parse_args()

    setup = load_setup_file(args.config)
    
    log_dir = os.path.join(setup['general']['save_dir'], "log")
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    logging.basicConfig(format='[%(levelname)s] [%(name)s] %(asctime)s: %(message)s', filename=f'{os.path.join(log_dir, f"generate_test_query_db_{current_time}")}.log', level=logging.INFO)
    
    construct_query_and_database_sets(annotation_dir=setup['general']['annotation'],
                                        folders=setup['general']['sequence'],
										th=setup['parameter']['threshold'],
										yaw=setup['parameter']['yaw'],
										save_dir=setup['general']['save_dir'], 
										length=len(setup['general']['sequence']),
										name=setup['general']['name'],
										test_region=setup['general']['test_region'])