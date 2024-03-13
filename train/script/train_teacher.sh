#!/bin/bash
if [ "$#" -ne 2]; then
    echo "Usage: $0 <model_name> <dataset_name>"
    exit 1
fi

model_name="$1" ## vxp or cattaneo
dataset_name="$2" ## kitti, vivid, oxford
setup_path="setup/${dataset_name}/setup_${model_name}_teacher.yml"

python train_teacher.py --config ${setup_path}
