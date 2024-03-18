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
from datetime import datetime
from parser import Parser
import os
import yaml
from yaml.loader import FullLoader
from create_all_annotation_oxford import create_all_annotation

def load_setup_file(path):
    with open(path) as f:
        data = yaml.load(f, Loader=FullLoader)
    return data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--config', type=str, default="config/setup.yml")
    parser.add_argument('--date_list', nargs='+', default=[], help='E.g. 2014-05-19-13-20-57 2014-11-25-09-18-32... When not specify, parse all the sequences in the GPS folders. ') ## If empty, parse all
    parser.add_argument('--force_write', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    setup = load_setup_file(args.config)
    log_dir = os.path.join(setup['parameter']['save_dir'], "log")
    if not os.path.exists(setup['parameter']['save_dir']):
        os.mkdir(setup['parameter']['save_dir'])
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if args.debug:
        level=logging.DEBUG
    else:
        level=logging.INFO
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logging.basicConfig(format='[%(levelname)s] [%(name)s] %(asctime)s: %(message)s', filename=f'{os.path.join(log_dir, current_time)}.log', level=level)
    

    date_list = []

    if len(args.date_list) == 0 and len(setup['sequence']) == 0:
        if setup['parameter']['ins_dir'] is not None:
            # Specify INS directory
            parse_dir = setup['parameter']['ins_dir']
        else:
            parse_dir = os.path.join(setup['parameter']['base_dir'], "gps")
        
        for root, dirs, files in os.walk(parse_dir):
            for dirname in sorted(dirs):
                date_list.append(dirname)
        setup['sequence'] = date_list
    elif len(setup['sequence']) != 0 and len(args.date_list) == 0:
        date_list = setup['sequence']
    else:
        setup['sequence'] = args.date_list
        for date in args.date_list:
            date_list.append(date)
            # p.run(date, args.force_write)

    p = Parser(**setup['parameter'])

    for date in date_list:
        p.run(date, args.force_write)
    
    create_all_annotation(dir=os.path.join(setup['parameter']['save_dir'], 'submap'), save_dir=os.path.join(args.save_dir, 'all_annotation.csv'))
    