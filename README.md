# VXP: Voxel-Cross-Pixel Large-scale Image-LiDAR Place Recognition (3DV 2025)

## [Project page](https://yunjinli.github.io/projects-vxp/) | [Paper](https://arxiv.org/abs/2403.14594)
## News
- 2024/12/20: We are actively working on the improved version of VXP, stay tuned...
## Introduction
We propose a novel Voxel-Cross-Pixel (VXP) approach, which establishes voxel and pixel correspondences in a self-supervised manner and brings them into a shared feature space. We achieve state-of-the-art performance in cross-modal retrieval on the Oxford RobotCar, ViViD++ datasets and KITTI benchmark, while maintaining high uni-modal global localization accuracy.

|                                               |                                               |
| --------------------------------------------- | --------------------------------------------- |
| ![2d3d](/assets/day1_evening_video_2D-3D.gif) | ![3d2d](/assets/day1_evening_video_3D-2D.gif) |
| ![2d2d](/assets/day1_evening_video_2D-2D.gif) | ![3d3d](/assets/day1_evening_video_3D-3D.gif) |

![teaser](assets/teaser_figure.jpg)
![pipeline](assets/pipeline.jpg)

## Setup the environement

```
git clone https://github.com/yunjinli/vxp.git
cd vxp
conda create -n VXP python=3.10 -y
conda activate VXP
pip install torch==2.0.1 torchvision==0.15.2 numpy pandas tqdm tensorboard psutil scikit-learn==1.2.2 bitarray pytorch-metric-learning==0.9.94 torchinfo
pip install -U openmim
mim install mmengine==0.7.3 mmcv==2.0.0 mmdet==3.0.0 mmdet3d==1.1.0
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

For sparse 3D convolution, we're using spconv library. You can follow the detailed installation guide on their [repository](https://github.com/traveller59/spconv). Or you can simply run the following command with specific cuda version (I'm using CUDA 12.0).

```
pip install spconv-cu120
```

## Dataset Format / Creation

Please see [here](./docs/dataset_format.md).

## Training

Please see [here](./docs/training.md).

## Inference

Please see [here](./docs/inference.md).

## BibTex

```
@article{li2024vxp,
    title={VXP: Voxel-Cross-Pixel Large-scale Image-LiDAR Place Recognition},
    author={Li, Yun-Jin and Gladkova, Mariia and Xia, Yan and Wang, Rui and Cremers, Daniel},
    journal={arXiv preprint arXiv:2403.14594},
    year={2024}
}
```
