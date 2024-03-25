#
# Created on Fri Mar 22 2024
# The MIT License (MIT)
# Copyright (c) 2023 Mariia Gladkova, Yun-Jin Li (Jim)
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
import os
from create_all_annotation_oxford import create_all_annotation
from process_rosbag_vxp import main as process_raw_vividpp
from generate_annotations_vividpp import main as generate_vividpp_annotation

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate dataset from ViViD++')
    parser.add_argument('--dataset_root', type=str, default="/storage/group/dataset_mirrors/01_incoming/vividpp/driving_full")
    parser.add_argument('--sequences', nargs='+', default=[], help='Name of the sequence in ViViD++ e.g. campus_day1 campus_day2 city_day1 city_day2 city_day1...') 
    parser.add_argument('--save_dir', type=str, default="/storage/user/lyun/vivid")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    log_dir = os.path.join(args.save_dir, "log")
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        
    if args.debug:
        level=logging.DEBUG
    else:
        level=logging.INFO
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logging.basicConfig(format='[%(levelname)s] [%(name)s] %(asctime)s: %(message)s', filename=f'{os.path.join(log_dir, current_time)}.log', level=level)
    
    for seq_name in args.sequences:
        process_raw_vividpp(args.dataset_root, seq_name, os.path.join(args.save_dir, seq_name))
        generate_vividpp_annotation(os.path.join(args.save_dir, seq_name))
        
    create_all_annotation(dir=args.save_dir, save_dir=os.path.join(args.save_dir, 'all_annotation.csv'))
