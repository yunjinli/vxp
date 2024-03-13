#!/bin/bash
if [ "$#" -ne 1]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

dataset_name="$1" ## kitti, vivid, oxford
setup_path="setup/${dataset_name}/setup_vxp_local_desc.yml"

python train_vxp_student_firststage.py --config ${setup_path}
