#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config>"
    exit 1
fi

config_name="$1"

python generate_training_tuples_kitti.py --config config/${config_name}.yml
python generate_3d_student_tuples_kitti.py --config config/${config_name}.yml
