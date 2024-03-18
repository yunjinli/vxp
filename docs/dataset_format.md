# Dataset Format

## Dataset

<!-- ### Annotation -->

Our dataset is organized by a annotation.csv file with the following information

| timestamp        | date                | northing       | easting           | yaw                  | submap_path                                                                                               | img_path                                                                                                             |
| ---------------- | ------------------- | -------------- | ----------------- | -------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 1400505709783450 | 2014-05-19-13-20-57 | 5735941.728159 | 620061.5917779999 | -0.11223730717958637 | /storage/user/lyun/Oxford_Robocar/dataset_every_5m_45runs/submap/2014-05-19-13-20-57/1400505709783450.npy | /storage/user/lyun/Oxford_Robocar/dataset_every_5m_45runs/image/undistorted/2014-05-19-13-20-57/1400505709783450.png |
| 1400505723969093 | 2014-05-19-13-20-57 | 5735945.528253 | 620061.2238309999 | -0.11404230717958645 | /storage/user/lyun/Oxford_Robocar/dataset_every_5m_45runs/submap/2014-05-19-13-20-57/1400505723969093.npy | /storage/user/lyun/Oxford_Robocar/dataset_every_5m_45runs/image/undistorted/2014-05-19-13-20-57/1400505723969093.png |

### Oxford dataset

As the baseline suggested, points on the ground should be removed and images should be undistorted. Thus, we provide a data parser for parsing the Oxford RobotCar dataset with some codebase provided in their [SDK](https://github.com/ori-mrg/robotcar-dataset-sdk). The raw downloaded Oxford RobotCar dataset in the base directory should be in the following structure:

```
Oxford_RobotCar/
- gps
  - 2014-05-19-13-20-57
  - 2014-06-26-09-31-18
  - ...
  - 2015-11-13-10-28-08
- image
  - 2014-05-19-13-20-57
  - 2014-06-26-09-31-18
  - ...
  - 2015-11-13-10-28-08
- lm_front
  - 2014-05-19-13-20-57
  - 2014-06-26-09-31-18
  - ...
  - 2015-11-13-10-28-08
```

When the structure of the data meets the above requirement, the parser should be able to work off-the-shelf.

```
cd dataset/utils
python generate_dataset_oxford.py --config config/setup_oxford.yml
```

This would parse all the sequences in the base_dir specified in the [setup_oxford.yml](../dataset/utils/config/setup_oxford.yml)
After parsing each sequence, the processed data would be saved in the save_dir with its own annotation\_[squence_date].csv. Finally, all the existing sub-annotation.csv would be combined to form an all_annotation.csv.

### ViViD++ dataset

Please follow their [documentation](https://visibilitydataset.github.io/1_about.html) to extract point clouds and images, and organize (image, point cloud) pair in annotation.csv as suggested above.

### KITTI dataset

```
cd dataset/utils/
python generate_dataset_kitti.py --kitti_odom_color <PATH_TO_KITTI_ODOM_COLOR> --SemanticKitti <PATH_TO_SEMANTIC_KITTI> --pose_base_dir <PATH_TO_POSE_DIR> --save_dir <PATH_TO_SAVE_DIR>
```

Note that as pykitti package try to substract its origin for each sequence when computing poses, we'll have to slightly modify the [code](https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py) in line 141 as follow:

```
## T_w_imu = transform_from_rot_trans(R, t - origin)
T_w_imu = transform_from_rot_trans(R, t)
```

## Training Tuples

Once the annotation.csv file is created, we can start generate the training tuples by running the script. Note that the corresponding setup file under [here](../dataset/utils/config/) has to be modified as well.

```
cd dataset/utils
## Oxford RobotCar
bash script/generate_tuple_oxford.sh setup_generate_tuple_oxford
## ViViD++
bash script/generate_tuple_vivid.sh setup_generate_tuple_vivid
## KITTI
bash script/generate_tuple_kitti.sh setup_generate_tuple_kitti
```

## Inference Tuples

```
cd dataset/utils/
## Oxford RobotCar
python generate_test_query_and_database_oxford.py --config config/setup_generate_test_query_db_oxford.yml
## ViViD++
python generate_test_query_and_database_vivid.py --config config/setup_generate_test_query_db_vivid_{campus, city}_{day12, nightday2}.yml
## KITTI
python generate_test_query_and_database_kitti.py --config config/setup_generate_test_query_db_kitti.yml
```
