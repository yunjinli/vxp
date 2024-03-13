#
# Created on Wed Aug 23 2023
# The MIT License (MIT)
# Copyright (c) 2023 Yun-Jin Li (Jim), Technical University of Munich
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


from torch.utils.tensorboard import SummaryWriter
import os  # nopep8
import sys  # nopep8
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(''), '..')))  # nopep8
import dataset.preprocessing  # nopep8
from dataset import OxfordImagePointcloudDataset, make_collate_fn  # nopep8
from utility_functions import load_setup_file, model_factory, load_pretrained_weight, save_setup_file, model_factory_v2, load_data_augmentation  # nopep8
import model  # nopep8
# from model import ViTExtractor, V2IProjection  # nopep8
import logging
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datetime import datetime
import torchvision
import psutil
import spconv
import numpy as np
from torch.nn import MSELoss
import matplotlib.pyplot as plt

spconv.constants.SPCONV_ALLOW_TF32 = True

process = psutil.Process()

random.seed(0)
torch.manual_seed(0)

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter(
    '[%(levelname)s] [%(name)s] [%(process)d] %(asctime)s: %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Train VXP second-stage')
    parser.add_argument('--config', type=str,
                        default="../setup/student_setup_global_desc.yml")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    # parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    setup = load_setup_file(args.config)
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join(setup['general']['save_dir'], 'log')
    model_dir = os.path.join(setup['general']['save_dir'], 'models')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    logging.basicConfig(format='[%(levelname)s] [%(name)s] %(asctime)s: %(message)s', filename=os.path.join(
        log_dir, f'train_student_{current_time}.log'), level=level)

    voxel_size = setup['dataset']['pcd_preprocessing']['OnlineVoxelization']['parameters']['voxel_size']
    point_cloud_range = setup['dataset']['pcd_preprocessing']['OnlineVoxelization']['parameters']['point_cloud_range']

    if setup['model']['arch'] == 'SecondAsppNetvlad' or setup['model']['arch'] == 'SecondAsppNetvladV2' or setup['model']['arch'] == 'VoxelLocalFeatureExtractor':
        setup['model']['parameters']['voxel_size'] = setup['dataset']['pcd_preprocessing']['OnlineVoxelization']['parameters']['voxel_size']
        setup['model']['parameters']['pcd_range'] = setup['dataset']['pcd_preprocessing']['OnlineVoxelization']['parameters']['point_cloud_range']
        grid_xyz = (np.asarray(
            point_cloud_range[3:]) - np.asarray(point_cloud_range[:3])) / np.asarray(voxel_size)
        grid_xyz = np.round(grid_xyz).tolist()
        setup['model']['parameters']['grid_zyx'] = [
            grid_xyz[2], grid_xyz[1], grid_xyz[0]]

    if not args.cpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    logger.info(f"Using {device}")

    setup_teacher = load_setup_file(setup['model']['teacher_setup'])

    transforms_img = load_data_augmentation(
        setup_teacher['dataset']['preprocessing'])
    transforms_img = torchvision.transforms.Compose(transforms_img)

    teacher = model_factory_v2(model, setup_teacher['model'])
    teacher = load_pretrained_weight(
        teacher, setup['model']['teacher_model'], device)
    teacher = teacher.to(device)
    teacher.eval()
    # img_extractor = ViTExtractor(
    #     model=teacher.backbone.encoder, stride=8, device=device)

    transforms_pcd = load_data_augmentation(
        data_augmentations=setup['dataset']['pcd_preprocessing'], custom_data_augmentation_modules=dataset.preprocessing)
    transforms_pcd = torchvision.transforms.Compose(transforms_pcd)

    default_coordinate_frame_transformation = load_data_augmentation(
        data_augmentations=setup['dataset']['pcd_coordinate_transformation'], custom_data_augmentation_modules=dataset.preprocessing)
    default_coordinate_frame_transformation = torchvision.transforms.Compose(
        default_coordinate_frame_transformation)

    pcd_data_augmentation = load_data_augmentation(
        data_augmentations=setup['dataset']['pcd_data_augmentation'], custom_data_augmentation_modules=dataset.preprocessing)
    pcd_data_augmentation = torchvision.transforms.Compose(
        pcd_data_augmentation)

    img_data_augmentation = load_data_augmentation(
        data_augmentations=setup['dataset']['img_data_augmentation'])
    img_data_augmentation = torchvision.transforms.Compose(
        img_data_augmentation)

    logger.info("Loading the training dataset")
    train_dataset = OxfordImagePointcloudDataset(annotation_path=setup['general']['annotation_dir'],
                                                 sample_index_list_path=setup['general']['train_pickle_dir'],
                                                 transform_img=transforms_img,
                                                 transform_pcd=transforms_pcd,
                                                 img_data_augmentation=img_data_augmentation,
                                                 pcd_data_augmentation=pcd_data_augmentation,
                                                 random_horizontal_mirroring_p=setup['dataset'][
                                                     'random_horizontal_mirroring_p'],
                                                 default_coordinate_frame_transformation=default_coordinate_frame_transformation,
                                                 voxelization=setup['dataset']['voxelization'])

    logger.info(f"Successfully load {len(train_dataset)} data")
    train_dataloader = DataLoader(train_dataset, batch_size=setup['dataset']['batch_size'], shuffle=True, num_workers=setup[
                                  'dataset']['num_workers'], collate_fn=make_collate_fn(train_dataset), pin_memory=True, drop_last=True)

    logger.info("Loading the validation dataset")

    val_dataset = OxfordImagePointcloudDataset(annotation_path=setup['general']['annotation_dir'],
                                               sample_index_list_path=setup['general']['val_pickle_dir'],
                                               transform_img=transforms_img,
                                               transform_pcd=transforms_pcd,
                                               # img_data_augmentation=img_data_augmentation,
                                               # pcd_data_augmentation=pcd_data_augmentation,
                                               # random_horizontal_mirroring_p=setup['dataset']['random_horizontal_mirroring_p'],
                                               default_coordinate_frame_transformation=default_coordinate_frame_transformation,
                                               voxelization=setup['dataset']['voxelization'])

    logger.info(f"Successfully load {len(val_dataset)} data")
    val_dataloader = DataLoader(val_dataset, batch_size=setup['dataset']['batch_size'], shuffle=False, num_workers=setup['dataset']
                                ['num_workers'], collate_fn=make_collate_fn(val_dataset), pin_memory=True, drop_last=True)

    setup_voxel_feat_ext = load_setup_file(
        setup['model']['student_pretrained_setup'])
    voxel_feat_ext = model_factory(
        model_collection=model, model_setup=setup_voxel_feat_ext['model'])
    voxel_feat_ext = load_pretrained_weight(
        voxel_feat_ext, setup['model']['student_pretrained_model'], device)

    downsampling_factor = 110 / 28
    voxel_size_2x = [voxel_size[0] * downsampling_factor,
                     voxel_size[1] * downsampling_factor,
                     voxel_size[2] * downsampling_factor]
    # width = 28
    # height = 28
    setup['model']['parameters'].update(
        setup_voxel_feat_ext['model']['parameters'])
    setup['model']['parameters']['fx'] = setup_voxel_feat_ext['model']['proj_params']['fx']
    setup['model']['parameters']['fy'] = setup_voxel_feat_ext['model']['proj_params']['fy']
    setup['model']['parameters']['cx'] = setup_voxel_feat_ext['model']['proj_params']['cx']
    setup['model']['parameters']['cy'] = setup_voxel_feat_ext['model']['proj_params']['cy']
    setup['model']['parameters']['height'] = setup_voxel_feat_ext['model']['proj_params']['h']
    setup['model']['parameters']['width'] = setup_voxel_feat_ext['model']['proj_params']['w']
    setup['model']['parameters']['R'] = setup_voxel_feat_ext['model']['proj_params']['R']

    setup['model']['parameters']['up_voxel_size'] = voxel_size_2x
    setup['model']['parameters']['v2i_pcd_range'] = setup_voxel_feat_ext['model']['parameters']['pcd_range']
    setup['model']['parameters']['device'] = f"{device}"

    student = model_factory(model_collection=model, model_setup=setup['model'])

    if setup['model']['load_pretrained']:
        logger.info("Loading pretrained weight from the student network")
        student.voxel_feat_ext = voxel_feat_ext

    student = student.to(device)
    logger.info(student.proj)
    loss_class = getattr(torch.nn, setup['loss']['fn'])
    loss = loss_class(**setup['loss']['parameters'])
    loss = loss.to(device)
    logger.info(f"Loss function: {loss}")

    opt_class = getattr(torch.optim, setup['optimizer']['fn'])
    opt = opt_class(params=student.parameters(), **
                    setup['optimizer']['parameters'])

    if setup['scheduler']['fn'] is not None:
        scheduler_class = getattr(
            torch.optim.lr_scheduler, setup['scheduler']['fn'])
        if setup['scheduler']['fn'] == 'LambdaLR':
            gamma = setup['scheduler']['parameters']['gamma']
            step_size = setup['scheduler']['parameters']['step_size']

            def lambda_fn(epoch): return max(gamma ** (epoch // step_size),
                                             setup['optimizer']['min_lr'] / setup['optimizer']['parameters']['lr'])
            scheduler = scheduler_class(opt, lr_lambda=lambda_fn)
        else:
            scheduler = scheduler_class(
                opt, **setup['scheduler']['parameters'])
    else:
        scheduler = None

    if setup['general']['name'] is None:
        name = f'student3d_{current_time}'
    else:
        name = setup['general']['name'] + f'_{current_time}'

    setup['general']['name'] = name
    setup['model']['path'] = os.path.join(model_dir, name + '.pth')
    save_setup_file(setup=setup, path=os.path.join(
        model_dir, 'setup_' + name + '.yml'))

    writer = SummaryWriter(os.path.join('..', 'tb_runs', name))

    epoch_end = setup['optimizer']['epochs']
    # val_loss = MSELoss()
    best_val_loss = 10000

    for epoch in range(setup['optimizer']['epochs']):
        student.train()
        running_loss = 0.0
        pbar = tqdm(train_dataloader)
        pbar.set_description(f'[TRAINING] Epoch {epoch+1}/{epoch_end}')

        for i, data in enumerate(pbar):
            opt.zero_grad(set_to_none=True)
            img_anchor, voxels, coors, num_points = data
            img_anchor = img_anchor.to(device, non_blocking=True)
            voxels = voxels.to(device, non_blocking=True)
            coors = coors.to(device, non_blocking=True)
            num_points = num_points.to(device, non_blocking=True)
            batch_size = coors[-1, 0].item() + 1

            submap_emb = student(
                features=voxels, num_points=num_points, coors=coors, batch_size=batch_size)
            # print(submap_emb.shape)
            img_emb = teacher(img_anchor)
            losses = loss(img_emb, submap_emb)

            losses.backward()
            opt.step()

            running_loss += losses.item()
            pbar.set_postfix({'Running Loss': running_loss/(i+1)})

        if scheduler is not None:
            scheduler.step()
            logger.info(f"Current Learning Rate: {scheduler.get_last_lr()}")
        writer.add_scalar("Loss/train", running_loss / (i+1), epoch)
        logger.info(
            f"Epoch {epoch+1}/{epoch_end} Training Loss: {running_loss / (i+1)}")

        student.eval()

        with torch.no_grad():
            running_loss = 0.0
            pbar = tqdm(val_dataloader)
            pbar.set_description(f'[VALIDATION] Epoch {epoch+1}/{epoch_end}')

            for i, data in enumerate(pbar):
                img_anchor, voxels, coors, num_points = data
                img_anchor = img_anchor.to(device, non_blocking=True)
                voxels = voxels.to(device, non_blocking=True)
                coors = coors.to(device, non_blocking=True)
                num_points = num_points.to(device, non_blocking=True)
                batch_size = coors[-1, 0].item() + 1
                submap_emb = student(
                    features=voxels, num_points=num_points, coors=coors, batch_size=batch_size)
                img_emb = teacher(img_anchor)

                losses = loss(img_emb, submap_emb)

                running_loss += losses.item()
                pbar.set_postfix({'Running Loss': running_loss/(i+1)})

            writer.add_scalar("Loss/val", running_loss / (i + 1), epoch)

            logger.info(
                f"Epoch {epoch+1}/{epoch_end} Validation Loss: {running_loss / (i + 1)}")
            if running_loss < best_val_loss:
                best_val_loss = running_loss
                logger.info(
                    f"Found better model with val loss: {best_val_loss / (i + 1)}")
                if name is None:
                    torch.save(student.state_dict(), os.path.join(
                        setup['general']['save_dir'], 'models', 'student2d.pth'))
                else:
                    torch.save(student.state_dict(), os.path.join(
                        setup['general']['save_dir'], 'models', name + '.pth'))
            if name is None:
                torch.save(student.state_dict(), os.path.join(
                    setup['general']['save_dir'], 'models', 'student2d_last.pth'))
            else:
                torch.save(student.state_dict(), os.path.join(
                    setup['general']['save_dir'], 'models', name + '_last.pth'))
