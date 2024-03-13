# VXP: Voxel-Cross-Pixel Large-scale Image-LiDAR Place Recognition

## Setup the environement

```
git clone https://github.com/yunjinli/vxp.git
cd vxp
conda create -n VXP python=3.10 -y
pip install -r requirements.txt
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