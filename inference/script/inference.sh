#!/bin/bash
if [ "$#" -ne 2]; then
    echo "Usage: $0 <dataset_type> <test_query_database_type>"
    exit 1
fi

dataset_type="$1" ## kitti, vivid, oxford
test_query_database_type="$2"

if test "$dataset_type" = "kitti"
then
    echo "Testing on KITTI Odometry benchmark"
    python inference_kitti.py --kdtree --config setup/${test_query_database_type}.yml
else
    if test "$dataset_type" = "oxford"
    then
        echo "Testing on Oxford RobotCar dataset"
        python inference.py --kdtree --config setup/${test_query_database_type}.yml
    else
        echo "Testing on ViViD++ dataset"
        python inference.py --kdtree --config setup/${test_query_database_type}.yml
    fi
fi
