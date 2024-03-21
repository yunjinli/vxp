#
# Created on May 30 2023
# The MIT License (MIT)
# Copyright (c) 2023 Yun-Jin Li (Jim), Technical University of Munich (TUM)
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
import sys  # nopep8
import os  # nopep8
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(''), '..')))  # nopep8
from utility_functions import load_setup_file, save_setup_file, model_factory_v2, model_factory, load_pretrained_weight, load_data_augmentation  # nopep8
import dataset.preprocessing  # nopep8
import model  # nopep8

import logging

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchvision
from dataset.dataset import PlaceRecognitionInferenceDb, PlaceRecognitionInferenceQuery, make_collate_fn
from datetime import datetime
import torch
import pickle as pkl
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize

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


def inference_single_q_db_2d_kdtree(enc_2d, q, db, q_data, db_data, num_clusters, encoder_dim, device, dates, writer=None, output_dim=None):
    with torch.no_grad():
        if output_dim is not None:
            pool_size = output_dim
        else:
            pool_size = num_clusters * encoder_dim
        db_submap_feat = np.empty((len(db_data), pool_size))
        db_img_feat = np.empty((len(db_data), pool_size))
        pbar = tqdm(db)

        for iteration, data in enumerate(pbar):
            img, _, indices = data
            img = img.to(device)
            emb_img = enc_2d(img)
            db_img_feat[indices.detach().numpy(
            ), :] = emb_img.detach().cpu().numpy()

        pbar = tqdm(q)

        q_img_feat = np.empty((len(q_data), pool_size))

        for iteration, data in enumerate(pbar):
            img, _, indices = data
            img = img.to(device)
            emb_img = enc_2d(img)
            q_img_feat[indices.detach().numpy(
            ), :] = emb_img.detach().cpu().numpy()

    gt = q_data.get_gt()

    # pair_recall, pair_similarity, pair_opr = get_recall(
    #     db_img_feat, q_img_feat, gt, writer, q_data=q_data, db_data=db_data)
    pair_recall, pair_similarity, pair_opr = get_recall(
        db_img_feat, q_img_feat, gt, writer, q_data=q_data, db_data=db_data)

    return pair_recall, pair_similarity, pair_opr


def inference_single_q_db_all_kdtree(enc_2d, enc_3d, q, db, q_data, db_data, device, dates, writer, output_dim=None, cross_normalize=False, num_neighbors=25):
    with torch.no_grad():
        pool_size = output_dim
        db_submap_feat = np.empty((len(db_data), pool_size))
        db_img_feat = np.empty((len(db_data), pool_size))
        pbar = tqdm(db)

        if db_data.get_dataset_type() == "InferenceImageVoxelDataset":
            for iteration, data in enumerate(pbar):
                img, voxels_out_list, coors_out_list, num_points_per_voxel_out_list, indices = data
                voxels_out_list = voxels_out_list.to(device)
                num_points_per_voxel_out_list = num_points_per_voxel_out_list.to(
                    device)
                coors_out_list = coors_out_list.to(device)
                img = img.to(device)
                emb_img = enc_2d(img)
                batch_size = coors_out_list[-1, 0].item() + 1
                emb_submap = enc_3d(features=voxels_out_list, num_points=num_points_per_voxel_out_list,
                                    coors=coors_out_list, batch_size=batch_size)
                db_img_feat[indices.detach().numpy(
                ), :] = emb_img.detach().cpu().numpy()
                db_submap_feat[indices.detach().numpy(
                ), :] = emb_submap.detach().cpu().numpy()

        pbar = tqdm(q)
        q_submap_feat = np.empty((len(q_data), pool_size))
        q_img_feat = np.empty((len(q_data), pool_size))

        if q_data.get_dataset_type() == "InferenceImageVoxelDataset":
            for iteration, data in enumerate(pbar):
                img, voxels_out_list, coors_out_list, num_points_per_voxel_out_list, indices = data
                voxels_out_list = voxels_out_list.to(device)
                num_points_per_voxel_out_list = num_points_per_voxel_out_list.to(
                    device)
                coors_out_list = coors_out_list.to(device)
                img = img.to(device)
                emb_img = enc_2d(img)
                batch_size = coors_out_list[-1, 0].item() + 1
                emb_submap = enc_3d(features=voxels_out_list, num_points=num_points_per_voxel_out_list,
                                    coors=coors_out_list, batch_size=batch_size)

                q_img_feat[indices.detach().numpy(
                ), :] = emb_img.detach().cpu().numpy()
                q_submap_feat[indices.detach().numpy(
                ), :] = emb_submap.detach().cpu().numpy()

    gt = q_data.get_gt()

    pair_recall_2d2d, pair_similarity_2d2d, pair_opr_2d2d = get_recall(
        db_img_feat, q_img_feat, gt, writer, mode='2D-2D', disp_img_tb=setup['general']['disp_img_tb'], q_data=q_data, db_data=db_data, num_neighbors=num_neighbors)

    if not cross_normalize:
        pair_recall_3d3d, pair_similarity_3d3d, pair_opr_3d3d = get_recall(
            db_submap_feat, q_submap_feat, gt, writer, mode='3D-3D', disp_img_tb=setup['general']['disp_img_tb'], q_data=q_data, db_data=db_data, num_neighbors=num_neighbors)
        pair_recall_2d3d, pair_similarity_2d3d, pair_opr_2d3d = get_recall(
            db_submap_feat, q_img_feat, gt, writer, mode='2D-3D', disp_img_tb=setup['general']['disp_img_tb'], q_data=q_data, db_data=db_data, num_neighbors=num_neighbors)
        pair_recall_3d2d, pair_similarity_3d2d, pair_opr_3d2d = get_recall(
            db_img_feat, q_submap_feat, gt, writer, mode='3D-2D', disp_img_tb=setup['general']['disp_img_tb'], q_data=q_data, db_data=db_data, num_neighbors=num_neighbors)
    else:
        pair_recall_3d3d, pair_similarity_3d3d, pair_opr_3d3d = get_recall(
            normalize(db_submap_feat), normalize(q_submap_feat), gt, writer, mode='3D-3D', disp_img_tb=setup['general']['disp_img_tb'], q_data=q_data, db_data=db_data, num_neighbors=num_neighbors)
        pair_recall_2d3d, pair_similarity_2d3d, pair_opr_2d3d = get_recall(
            normalize(db_submap_feat), normalize(q_img_feat), gt, writer, mode='2D-3D', disp_img_tb=setup['general']['disp_img_tb'], q_data=q_data, db_data=db_data, num_neighbors=num_neighbors)
        pair_recall_3d2d, pair_similarity_3d2d, pair_opr_3d2d = get_recall(
            normalize(db_img_feat), normalize(q_submap_feat), gt, writer, mode='3D-2D', disp_img_tb=setup['general']['disp_img_tb'], q_data=q_data, db_data=db_data, num_neighbors=num_neighbors)

    return pair_recall_2d2d, pair_similarity_2d2d, pair_opr_2d2d, pair_recall_3d3d, pair_similarity_3d3d, pair_opr_3d3d, pair_recall_2d3d, pair_similarity_2d3d, pair_opr_2d3d, pair_recall_3d2d, pair_similarity_3d2d, pair_opr_3d2d


def get_recall(database_output, queries_output, gt, writer=None, mode='2D-2D', disp_img_tb=False, q_data=None, db_data=None, num_neighbors=25):
    assert mode in ['2D-2D', '3D-3D', '2D-3D', '3D-2D']
    database_nbrs = KDTree(database_output)

    # num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(database_output.shape[0] / 100.0)), 1)
    # print(threshold)
    max_recall_to_tb = 1
    written_to_tb = 0
    num_evaluated = 0

    for i in range(len(queries_output)):
        true_neighbors = gt[i]
        if (len(true_neighbors) == 0):
            logger.warning("No re-visit for this query")
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]), k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if (j == 0):
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break
            else:
                if j == 0:  # only plot the image if recall@1 is not successful
                    if disp_img_tb:
                        if writer is not None:
                            if written_to_tb < max_recall_to_tb:
                                show_imgs = []
                                if mode == '2D-2D' or mode == '2D-3D':
                                    img_q = torchvision.io.read_image(
                                        q_data.q[q_data.test_index][i]['img_path'])
                                    # print(f"Image shape: {img_q.shape}")
                                elif mode == '3D-3D' or mode == '3D-2D':
                                    img_q = torchvision.io.read_image(q_data.q[q_data.test_index][i]['submap_path'].replace(
                                        'submap', 'submap_plot3d').replace('.npy', '.png'), torchvision.io.ImageReadMode.RGB)
                                    img_q = torchvision.transforms.Resize(
                                        size=(240, 320))(img_q)
                                    # print(f"Submap 3D plot shape: {img_q.shape}")

                                box = torch.tensor(
                                    [[0, 0, 320, 240]], dtype=torch.float)
                                color = ["yellow"]
                                img_q = torchvision.utils.draw_bounding_boxes(
                                    img_q, boxes=box, colors=color, width=10)
                                show_imgs.append(img_q)
                                for pred_idx in indices[0][:10]:
                                    if mode == '2D-2D' or mode == '3D-2D':
                                        img = torchvision.io.read_image(
                                            db_data.db[db_data.db_index][pred_idx]['img_path'])
                                        # print(f"Image shape: {img.shape}")
                                    elif mode == '3D-3D' or mode == '2D-3D':
                                        img = torchvision.io.read_image(db_data.db[db_data.db_index][pred_idx]['submap_path'].replace(
                                            'submap', 'submap_plot3d').replace('.npy', '.png'), torchvision.io.ImageReadMode.RGB)
                                        img = torchvision.transforms.Resize(
                                            size=(240, 320))(img)
                                        # print(f"Submap 3D plot shape: {img.shape}")
                                    if np.any(true_neighbors == pred_idx):
                                        box = torch.tensor(
                                            [[0, 0, 320, 240]], dtype=torch.float)
                                        color = ["green"]
                                        img = torchvision.utils.draw_bounding_boxes(
                                            img, boxes=box, colors=color, width=10)
                                    show_imgs.append(img)
                                img_grid = torchvision.utils.make_grid(
                                    show_imgs)
                                xq, yq = q_data.getUTM(i)
                                writer.add_image(
                                    f'{mode}_Recall@10_show/embed_dist:{distances[0][:10]}_true_dist:{get_distances2query(xq=xq, yq=yq, db_data=db_data, db_indices=indices[0][:10])}', img_grid)
                                written_to_tb += 1

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = one_percent_retrieved / float(num_evaluated)
    recall = np.cumsum(recall) / float(num_evaluated)

    for i in range(num_neighbors):
        # recalls[n] = recall_at_n[i]
        logger.info("{} Recall@{}: {:.4f} %".format(mode,
                    i + 1, recall[i] * 100))

    logger.info("{} Recall@1%: {:.4f} %".format(mode,
                one_percent_recall * 100))

    return recall, top1_similarity_score, one_percent_recall


def get_distances2query(xq, yq, db_data, db_indices):
    dists = []
    for i in db_indices:
        x, y = db_data.getUTM(i)
        d = get_distance(xq, yq, x, y)
        dists.append(d)
    return dists


def get_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Perform 2d-3d shared embedding space inference')
    parser.add_argument('--config', type=str,
                        default="../setup/inference_setup_v2.yml")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--only2d', action='store_true')
    parser.add_argument('--kdtree', action='store_true')
    # parser.add_argument('--use_tuple', action='store_true')
    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    setup = load_setup_file(args.config)
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join(setup['general']['save_dir'], 'log')

    logging.basicConfig(format='[%(levelname)s] [%(name)s] %(asctime)s: %(message)s', filename=os.path.join(
        log_dir, f'inference_{current_time}.log'), level=level)

    if not args.cpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    # transforms_img = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

    logger.info(setup)
    if args.only2d:
        if args.kdtree:
            recall = np.zeros(25)
            count = 0
            similarity = []
            one_percent_recall = []
            query_path = setup['general']['query']
            db_path = setup['general']['db']
            logger.info(f"Using {device}")
            setup_student = load_setup_file(setup['general']['student_setup'])
            setup_teacher = load_setup_file(setup['general']['teacher_setup'])

            transforms_img = load_data_augmentation(
                setup_teacher['dataset']['preprocessing'])
            transforms_img = torchvision.transforms.Compose(transforms_img)

            teacher = model_factory_v2(model, setup_teacher['model'])
            teacher = load_pretrained_weight(
                teacher, setup['general']['teacher_model'], device)
            teacher = teacher.to(device)
            teacher.eval()

            writer = SummaryWriter(os.path.join(
                '..', 'tb_runs', setup_teacher['general']['name']))

            query_data = PlaceRecognitionInferenceQuery(
                query_path, transform_img=transforms_img)
            db_data = PlaceRecognitionInferenceDb(
                db_path, transform_img=transforms_img)

            logger.info(f"{len(query_data.q)} sequences as queries in total")

            assert len(setup['general']['sequence']) == len(db_data.db)
            for qi in range(len(query_data.q)):
                # write_to_tb = False
                for dbj in range(len(query_data.q)):
                    if qi == dbj:
                        logger.info("Query and database is the same, skip")
                        continue
                    else:
                        query_data.set_test_index(qi)
                        query_data.set_db_index(dbj)
                        db_data.set_db_index(dbj)
                        query_dataloader = DataLoader(
                            query_data, batch_size=setup_teacher['dataset']['batch_size'], shuffle=False, num_workers=setup['general']['num_workers'], collate_fn=make_collate_fn(query_data))
                        db_dataloader = DataLoader(
                            db_data, batch_size=setup_teacher['dataset']['batch_size'], shuffle=False, num_workers=setup['general']['num_workers'], collate_fn=make_collate_fn(db_data))
                        logger.info(f"Query index: {qi}")
                        logger.info(f"Database index: {dbj}")
                        logger.info(
                            f"Query: {setup['general']['sequence'][qi]}")
                        logger.info(
                            f"Database: {setup['general']['sequence'][dbj]}")
                        # We want to randomly sampled query-database pair to
                        # put predicted images compared with ground-true into tensorboard.
                        # Once a query has already log images in the tb, we disable the writing
                        # option to make sure that for each query, there will be at most 1 query-database
                        # result logged into tb in order not to put too much information in the tb
                        pair_recall, pair_similarity, pair_opr = inference_single_q_db_2d_kdtree(
                            teacher, query_dataloader, db_dataloader, query_data, db_data, 64, 512, device, setup['general']['sequence'], writer, output_dim=setup_teacher['model']['output_dim'])
                        for i in range(25):
                            if setup['general']['name'] is not None:
                                inf_dataset_name = setup['general']['name']
                                writer.add_scalar(
                                    f"Recall_{inf_dataset_name}/Query:{setup['general']['sequence'][qi]} | Database: {setup['general']['sequence'][dbj]}", pair_recall[i], i + 1)
                            else:
                                writer.add_scalar(
                                    f"Recall/Query:{setup['general']['sequence'][qi]} | Database: {setup['general']['sequence'][dbj]}", pair_recall[i], i + 1)
                        recall += np.array(pair_recall)
                        count += 1
                        one_percent_recall.append(pair_opr)
                        for x in pair_similarity:
                            similarity.append(x)

                        logger.info("Running Average Recall")
                        for i in range(25):
                            logger.info(
                                "2D-2D Recall@{}: {:.4f} %".format(i + 1, (recall / count)[i] * 100))
                        logger.info(
                            "2D-2D Recall@1%: {:.4f} %".format(np.mean(one_percent_recall) * 100))

            ave_recall = recall / count
            average_similarity = np.mean(similarity)
            ave_one_percent_recall = np.mean(one_percent_recall)

            logger.info("Overall Average Recall")
            for i in range(25):
                if setup['general']['name'] is not None:
                    inf_dataset_name = setup['general']['name']
                    writer.add_scalar(
                        f"Recall_{inf_dataset_name}/Overall", ave_recall[i], i + 1)
                else:
                    writer.add_scalar(f"Recall/Overall", ave_recall[i], i + 1)
                logger.info(
                    "2D-2D Recall@{}: {:.4f} %".format(i + 1, ave_recall[i] * 100))
            if setup['general']['name'] is not None:
                if setup['general']['name'] is not None:
                    inf_dataset_name = setup['general']['name']
                    writer.add_scalar(
                        f"Recall_{inf_dataset_name}/Overall@1%", ave_one_percent_recall, i + 1)
            else:
                writer.add_scalar(f"Recall/Overall@1%",
                                  ave_one_percent_recall, i + 1)
            logger.info(
                "2D-2D Recall@1%: {:.4f} %".format(ave_one_percent_recall * 100))
    else:
        if args.kdtree:
            recall2d2d = np.zeros(setup['general']['k'])
            recall2d3d = np.zeros(setup['general']['k'])
            recall3d2d = np.zeros(setup['general']['k'])
            recall3d3d = np.zeros(setup['general']['k'])
            count = 0

            one_percent_recall_2d2d = []
            one_percent_recall_3d3d = []
            one_percent_recall_2d3d = []
            one_percent_recall_3d2d = []

            query_path = setup['general']['query']
            db_path = setup['general']['db']
            logger.info(f"Using {device}")
            setup_student = load_setup_file(setup['general']['student_setup'])
            setup['general']['teacher_setup'] = setup_student['model']['teacher_setup']
            setup['general']['teacher_model'] = setup_student['model']['teacher_model']
            setup_teacher = load_setup_file(setup['general']['teacher_setup'])

            transforms_img = load_data_augmentation(
                setup_teacher['dataset']['preprocessing'])
            transforms_img = torchvision.transforms.Compose(transforms_img)

            try:
                teacher = model_factory_v2(model, setup_teacher['model'])
                teacher = load_pretrained_weight(
                    teacher, setup['general']['teacher_model'], device)
            except:
                logger.warning("Load model with model factory v2 failed, try v1")
                teacher = model_factory(model, setup_teacher['model'])
                teacher = load_pretrained_weight(
                    teacher, setup['general']['teacher_model'], device)
            # teacher = model_factory_v2(model, setup_teacher['model'])
            # teacher = load_pretrained_weight(
            #     teacher, setup['general']['teacher_model'], device)
            teacher = teacher.to(device)
            teacher.eval()

            transforms_pcd = load_data_augmentation(
                data_augmentations=setup_student['dataset']['pcd_preprocessing'], custom_data_augmentation_modules=dataset.preprocessing)
            transforms_pcd = torchvision.transforms.Compose(transforms_pcd)

            default_coordinate_frame_transformation = load_data_augmentation(
                data_augmentations=setup_student['dataset']['pcd_coordinate_transformation'], custom_data_augmentation_modules=dataset.preprocessing)
            default_coordinate_frame_transformation = torchvision.transforms.Compose(
                default_coordinate_frame_transformation)

            student = model_factory(model, setup_student['model'])
            student = load_pretrained_weight(
                student, setup['general']['student_model'], device)
            if setup_student['model']['arch'] == 'MultiScaleVXPGeM':
                student.init()
            # student_arch = getattr(model, setup_student['model']['arch'])
            # student = student_arch(**setup_student['model']['parameters'])
            # student.load_state_dict(torch.load(setup['general']['student_model'], map_location=torch.device(device)))
            student = student.to(device)
            student.eval()

            writer = SummaryWriter(os.path.join(
                '..', 'tb_runs', setup_teacher['general']['name'] + '_' + setup_student['general']['name']))

            query_data = PlaceRecognitionInferenceQuery(query_path,
                                                        transform_img=transforms_img,
                                                        transform_submap=transforms_pcd,
                                                        default_coordinate_frame_transformation=default_coordinate_frame_transformation,
                                                        voxelization=setup_student['dataset']['voxelization'], rebase_dir=setup['general']['rebase_dir'])
            db_data = PlaceRecognitionInferenceDb(db_path,
                                                  transform_img=transforms_img,
                                                  transform_submap=transforms_pcd,
                                                  default_coordinate_frame_transformation=default_coordinate_frame_transformation,
                                                  voxelization=setup_student['dataset']['voxelization'], rebase_dir=setup['general']['rebase_dir'])

            logger.info(f"{len(query_data.q)} sequences as queries in total")
            assert len(setup['general']['sequence']) == len(db_data.db)

            for qi in range(len(query_data.q)):
                # write_to_tb = False
                for dbj in range(len(query_data.q)):
                    # if qi == dbj and len(query_data.q) != 1:
                    if qi == dbj:
                        logger.info("Query and database is the same, skip")
                        continue
                    else:
                        query_data.set_test_index(qi)
                        query_data.set_db_index(dbj)
                        db_data.set_db_index(dbj)
                        query_dataloader = DataLoader(
                            query_data, batch_size=setup['general']['batch_size'], shuffle=False, num_workers=setup['general']['num_workers'], collate_fn=make_collate_fn(query_data))
                        db_dataloader = DataLoader(
                            db_data, batch_size=setup['general']['batch_size'], shuffle=False, num_workers=setup['general']['num_workers'], collate_fn=make_collate_fn(db_data))
                        logger.info(f"Query index: {qi}")
                        logger.info(f"Database index: {dbj}")
                        logger.info(
                            f"Query: {setup['general']['sequence'][qi]}")
                        logger.info(
                            f"Database: {setup['general']['sequence'][dbj]}")
                        pair_recall_2d2d, pair_similarity_2d2d, pair_opr_2d2d, pair_recall_3d3d, pair_similarity_3d3d, pair_opr_3d3d, pair_recall_2d3d, pair_similarity_2d3d, pair_opr_2d3d, pair_recall_3d2d, pair_similarity_3d2d, pair_opr_3d2d = inference_single_q_db_all_kdtree(
                            enc_2d=teacher, enc_3d=student, q=query_dataloader, db=db_dataloader, q_data=query_data, db_data=db_data, device=device, dates=setup['general']['sequence'], writer=writer, output_dim=setup_teacher['model']['output_dim'], cross_normalize=setup['general']['cross_normalize'], num_neighbors=setup['general']['k'])

                        for i in range(setup['general']['k']):
                            if setup['general']['name'] is not None:
                                inf_dataset_name = setup['general']['name']
                                writer.add_scalar(
                                    f"Recall2D2D_{inf_dataset_name}/Query:{setup['general']['sequence'][qi]} | Database: {setup['general']['sequence'][dbj]}", pair_recall_2d2d[i], i + 1)
                                writer.add_scalar(
                                    f"Recall3D3D_{inf_dataset_name}/Query:{setup['general']['sequence'][qi]} | Database: {setup['general']['sequence'][dbj]}", pair_recall_3d3d[i], i + 1)
                                writer.add_scalar(
                                    f"Recall2D3D_{inf_dataset_name}/Query:{setup['general']['sequence'][qi]} | Database: {setup['general']['sequence'][dbj]}", pair_recall_2d3d[i], i + 1)
                                writer.add_scalar(
                                    f"Recall3D2D_{inf_dataset_name}/Query:{setup['general']['sequence'][qi]} | Database: {setup['general']['sequence'][dbj]}", pair_recall_3d2d[i], i + 1)
                            else:
                                writer.add_scalar(
                                    f"Recall2D2D/Query:{setup['general']['sequence'][qi]} | Database: {setup['general']['sequence'][dbj]}", pair_recall_2d2d[i], i + 1)
                                writer.add_scalar(
                                    f"Recall3D3D/Query:{setup['general']['sequence'][qi]} | Database: {setup['general']['sequence'][dbj]}", pair_recall_3d3d[i], i + 1)
                                writer.add_scalar(
                                    f"Recall2D3D/Query:{setup['general']['sequence'][qi]} | Database: {setup['general']['sequence'][dbj]}", pair_recall_2d3d[i], i + 1)
                                writer.add_scalar(
                                    f"Recall3D2D/Query:{setup['general']['sequence'][qi]} | Database: {setup['general']['sequence'][dbj]}", pair_recall_3d2d[i], i + 1)
                        recall2d2d += np.array(pair_recall_2d2d)
                        recall3d3d += np.array(pair_recall_3d3d)
                        recall2d3d += np.array(pair_recall_2d3d)
                        recall3d2d += np.array(pair_recall_3d2d)

                        count += 1

                        one_percent_recall_2d2d.append(pair_opr_2d2d)
                        one_percent_recall_3d3d.append(pair_opr_3d3d)
                        one_percent_recall_2d3d.append(pair_opr_2d3d)
                        one_percent_recall_3d2d.append(pair_opr_3d2d)

                        logger.info("Running Average Recall")
                        for i in range(setup['general']['k']):
                            logger.info(
                                "2D-2D Recall@{}: {:.4f} %".format(i + 1, (recall2d2d / count)[i] * 100))
                            logger.info(
                                "3D-3D Recall@{}: {:.4f} %".format(i + 1, (recall3d3d / count)[i] * 100))
                            logger.info(
                                "2D-3D Recall@{}: {:.4f} %".format(i + 1, (recall2d3d / count)[i] * 100))
                            logger.info(
                                "3D-2D Recall@{}: {:.4f} %".format(i + 1, (recall3d2d / count)[i] * 100))

                        logger.info(
                            "2D-2D Recall@1%: {:.4f} %".format(np.mean(one_percent_recall_2d2d) * 100))
                        logger.info(
                            "3D-3D Recall@1%: {:.4f} %".format(np.mean(one_percent_recall_3d3d) * 100))
                        logger.info(
                            "2D-3D Recall@1%: {:.4f} %".format(np.mean(one_percent_recall_2d3d) * 100))
                        logger.info(
                            "3D-2D Recall@1%: {:.4f} %".format(np.mean(one_percent_recall_3d2d) * 100))

            ave_recall2d2d = recall2d2d / count
            ave_recall3d3d = recall3d3d / count
            ave_recall2d3d = recall2d3d / count
            ave_recall3d2d = recall3d2d / count

            # average_similarity = np.mean(similarity)
            ave_one_percent_recall_2d2d = np.mean(one_percent_recall_2d2d)
            ave_one_percent_recall_3d3d = np.mean(one_percent_recall_3d3d)
            ave_one_percent_recall_2d3d = np.mean(one_percent_recall_2d3d)
            ave_one_percent_recall_3d2d = np.mean(one_percent_recall_3d2d)

            logger.info("Overall Average Recall")
            for i in range(setup['general']['k']):
                if setup['general']['name'] is not None:
                    inf_dataset_name = setup['general']['name']
                    writer.add_scalar(
                        f"Recall2D2D_{inf_dataset_name}/Overall", ave_recall2d2d[i], i + 1)
                    writer.add_scalar(
                        f"Recall3D3D_{inf_dataset_name}/Overall", ave_recall3d3d[i], i + 1)
                    writer.add_scalar(
                        f"Recall2D3D_{inf_dataset_name}/Overall", ave_recall2d3d[i], i + 1)
                    writer.add_scalar(
                        f"Recall3D2D_{inf_dataset_name}/Overall", ave_recall3d2d[i], i + 1)
                else:
                    writer.add_scalar(f"Recall2D2D/Overall",
                                      ave_recall2d2d[i], i + 1)
                    writer.add_scalar(f"Recall3D3D/Overall",
                                      ave_recall3d3d[i], i + 1)
                    writer.add_scalar(f"Recall2D3D/Overall",
                                      ave_recall2d3d[i], i + 1)
                    writer.add_scalar(f"Recall3D2D/Overall",
                                      ave_recall3d2d[i], i + 1)
                logger.info("2D-2D Recall@{}: {:.4f} %".format(i +
                            1, ave_recall2d2d[i] * 100))
                logger.info("3D-3D Recall@{}: {:.4f} %".format(i +
                            1, ave_recall3d3d[i] * 100))
                logger.info("2D-3D Recall@{}: {:.4f} %".format(i +
                            1, ave_recall2d3d[i] * 100))
                logger.info("3D-2D Recall@{}: {:.4f} %".format(i +
                            1, ave_recall3d2d[i] * 100))

            if setup['general']['name'] is not None:
                inf_dataset_name = setup['general']['name']
                writer.add_scalar(
                    f"Recall2D2D_{inf_dataset_name}/Overall@1%", ave_one_percent_recall_2d2d, i + 1)
                writer.add_scalar(
                    f"Recall3D3D_{inf_dataset_name}/Overall@1%", ave_one_percent_recall_3d3d, i + 1)
                writer.add_scalar(
                    f"Recall2D3D_{inf_dataset_name}/Overall@1%", ave_one_percent_recall_2d3d, i + 1)
                writer.add_scalar(
                    f"Recall3D2D_{inf_dataset_name}/Overall@1%", ave_one_percent_recall_3d2d, i + 1)
            else:
                writer.add_scalar(f"Recall2D2D/Overall@1%",
                                  ave_one_percent_recall_2d2d, i + 1)
                writer.add_scalar(f"Recall3D3D/Overall@1%",
                                  ave_one_percent_recall_3d3d, i + 1)
                writer.add_scalar(f"Recall2D3D/Overall@1%",
                                  ave_one_percent_recall_2d3d, i + 1)
                writer.add_scalar(f"Recall3D2D/Overall@1%",
                                  ave_one_percent_recall_3d2d, i + 1)

            logger.info(
                "2D-2D Recall@1%: {:.4f} %".format(ave_one_percent_recall_2d2d * 100))
            logger.info(
                "3D-3D Recall@1%: {:.4f} %".format(ave_one_percent_recall_3d3d * 100))
            logger.info(
                "2D-3D Recall@1%: {:.4f} %".format(ave_one_percent_recall_2d3d * 100))
            logger.info(
                "3D-2D Recall@1%: {:.4f} %".format(ave_one_percent_recall_3d2d * 100))
