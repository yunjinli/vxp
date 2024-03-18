#
# Created on Wed May 3 2023
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
import pandas as pd
import os
from datetime import datetime

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('[%(levelname)s] [%(name)s] %(asctime)s: %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

def create_all_annotation(dir='/storage/user/lyun/Oxford_Robocar/submap', save_dir='/storage/user/lyun/Oxford_Robocar/all_annotation.csv'):
    sub_annotations_path = []
    for root, dirs, files in os.walk(dir):
        for dirname in sorted(dirs):
            sub_annotation_path = os.path.join(dir, dirname, f"annotation_{dirname}.csv")
            sub_annotations_path.append(sub_annotation_path)
    for path in sub_annotations_path:
        logger.info(f"Sub-annotations to be combined: {path}")
    all_annotation = pd.DataFrame({})
    counter = 0
    for sub_annotation_path in sub_annotations_path:
        try:
            sub_annotation = pd.read_csv(sub_annotation_path)
            all_annotation = pd.concat([all_annotation, sub_annotation], ignore_index=True)
            counter += 1
        except:
            logger.warning(f"Cannot find sub annotation file from directory: {sub_annotation_path}")
    all_annotation.to_csv(save_dir)
    logger.info(f"{counter} sub-annotation files combined")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Combine annotation')
    parser.add_argument('--save_dir', type=str, default="/storage/user/lyun/Oxford_Robocar/")
    args = parser.parse_args()
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    logging.basicConfig(format='[%(levelname)s] [%(name)s] %(asctime)s: %(message)s', filename=f"generate_all_annotation_{current_time}.log", level=logging.INFO)
    create_all_annotation(dir=os.path.join(args.save_dir, 'submap'), save_dir=os.path.join(args.save_dir, 'all_annotation.csv'))