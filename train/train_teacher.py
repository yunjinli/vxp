#
# Created on May 16 2023
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
import sys  # nopep8
import os  # nopep8
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(''), '..')))  # nopep8
import logging
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import logging
import numpy as np

import torchvision
from torch.utils.tensorboard import SummaryWriter
import psutil

from sklearn.neighbors import KDTree
from dataset import OxfordImageDataset, BatchSampler, make_collate_fn  # nopep8
import model  # nopep8
from loss import make_loss  # nopep8
from utility_functions import load_setup_file, model_factory_v2, load_pretrained_weight, save_setup_file, load_data_augmentation  # nopep8

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


def print_stats(stats, phase):
    if 'num_pairs' in stats:
        # For batch hard contrastive loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Pairs per batch (all/non-zero pos/non-zero neg): {:.1f}/{:.1f}/{:.1f}'
        logger.info(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pairs'],
                             stats['pos_pairs_above_threshold'], stats['neg_pairs_above_threshold']))
    elif 'num_triplets' in stats:
        # For triplet loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Triplets per batch (all/non-zero): {:.1f}/{:.1f}'
        logger.info(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_triplets'],
                             stats['num_non_zero_triplets']))
    elif 'num_pos' in stats:
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   #positives/negatives: {:.1f}/{:.1f}'
        logger.info(s.format(
            phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pos'], stats['num_neg']))

    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.8f}/{:.8f}/{:.8f}   Neg dist (min/mean/max): {:.8f}/{:.8f}/{:.8f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if 'pos_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Pos loss: {:.8f}  Neg loss: {:.8f}'
        l += [stats['pos_loss'], stats['neg_loss']]
    if len(l) > 0:
        logger.info(s.format(*l))


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e])
             else stats[e] for e in stats}
    return stats


def train(teacher, optimizer, loss_fn, scheduler, setup, train_dataloader, val_dataloader, val_dataset, device, name=None, writer=None, recall_interval=5, debug=False, batch_expansion_th=None):
    if not os.path.exists(os.path.join(setup['general']['save_dir'], 'models')):
        os.mkdir(os.path.join(setup['general']['save_dir'], 'models'))
    epoch_end = setup['optimizer']['epochs']
    best_recall_acc = 0
    best_val = 100000
    best_val_nz = 100000

    # Training statistics
    stats = {'train': [], 'val': []}
    if debug:
        setup['optimizer']['epochs'] = 1
    for epoch in range(setup['optimizer']['epochs']):
        logger.info(f'Epoch {epoch + 1}/{epoch_end}')
        running_stats = []
        teacher.train()
        pbar = tqdm(train_dataloader)
        pbar.set_description(f'Epoch {epoch + 1}/{epoch_end}')
        batch_count = 0
        for i, (batch, positives_mask, negatives_mask) in enumerate(pbar):
            if debug:
                batch_count += 1
                if batch_count > 10:
                    break
            batch_stats = {}
            # print(positives_mask)
            # print(negatives_mask)
            # Move everything to the device except 'coords' which must stay on CPU
            batch = batch.to(device, non_blocking=True)

            n_positives = torch.sum(positives_mask).item()
            n_negatives = torch.sum(negatives_mask).item()

            if n_positives == 0 or n_negatives == 0:
                # Skip a batch without positives or negatives
                logger.warning(
                    'Skipping batch without positive or negative examples')
                continue

            optimizer.zero_grad()

            # Compute embeddings of all elements
            embeddings = teacher(batch)
            loss, temp_stats, _ = loss_fn(
                embeddings, positives_mask, negatives_mask)

            temp_stats = tensors_to_numbers(temp_stats)
            batch_stats.update(temp_stats)
            batch_stats['loss'] = loss.item()

            loss.backward()
            optimizer.step()

            running_stats.append(batch_stats)
            # torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

        # ******* Training Epoch END *******
        # Compute mean stats for the epoch
        epoch_stats = {}
        for key in running_stats[0].keys():
            temp = [e[key] for e in running_stats]
            epoch_stats[key] = np.mean(temp)

        stats['train'].append(epoch_stats)
        print_stats(epoch_stats, 'train')
        # ******* Validation Epoch START *******
        running_stats = []  # running stats for the current epoch
        teacher.eval()
        pbar = tqdm(val_dataloader)
        pbar.set_description(f'Epoch {epoch+1}/{epoch_end}')
        batch_count = 0
        for i, (batch, positives_mask, negatives_mask) in enumerate(pbar):
            if debug:
                batch_count += 1
                if batch_count > 10:
                    break
            with torch.no_grad():
                batch_stats = {}
                # Move everything to the device except 'coords' which must stay on CPU
                batch = batch.to(device, non_blocking=True)

                n_positives = torch.sum(positives_mask).item()
                n_negatives = torch.sum(negatives_mask).item()

                if n_positives == 0 or n_negatives == 0:
                    # Skip a batch without positives or negatives
                    logger.warning(
                        'Skipping batch without positive or negative examples')
                    continue

                # Compute embeddings of all elements
                embeddings = teacher(batch)
                loss, temp_stats, _ = loss_fn(
                    embeddings, positives_mask, negatives_mask)

                temp_stats = tensors_to_numbers(temp_stats)
                batch_stats.update(temp_stats)
                batch_stats['loss'] = loss.item()

            running_stats.append(batch_stats)
            torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

        # Compute mean stats for the epoch
        epoch_stats = {}
        for key in running_stats[0].keys():
            temp = [e[key] for e in running_stats]
            epoch_stats[key] = np.mean(temp)

        stats['val'].append(epoch_stats)
        print_stats(epoch_stats, 'val')

        # ******* EPOCH END *******
        if scheduler is not None:
            scheduler.step()

        if writer is not None:
            loss_metrics = {'train': stats['train'][-1]['loss']}
            loss_metrics['val'] = stats['val'][-1]['loss']
            writer.add_scalars('Loss', loss_metrics, epoch)

            if 'num_triplets' in stats['train'][-1]:
                nz_metrics = {'train': stats['train']
                              [-1]['num_non_zero_triplets']}
                nz_metrics['val'] = stats['val'][-1]['num_non_zero_triplets']
                writer.add_scalars('Non-zero triplets', nz_metrics, epoch)

            elif 'num_pairs' in stats['train'][-1]:
                nz_metrics = {'train_pos': stats['train'][-1]['pos_pairs_above_threshold'],
                              'train_neg': stats['train'][-1]['neg_pairs_above_threshold']}
                nz_metrics['val_pos'] = stats['val'][-1]['pos_pairs_above_threshold']
                nz_metrics['val_neg'] = stats['val'][-1]['neg_pairs_above_threshold']
                writer.add_scalars('Non-zero pairs', nz_metrics, epoch)

            writer.add_scalar('Train/Loss of non-zero triplets',
                              stats['train'][-1]['loss'] * nz_metrics['train'] / train_dataloader.batch_sampler.batch_size, epoch)
            writer.add_scalar('Validation/Loss of non-zero triplets',
                              stats['val'][-1]['loss'] * nz_metrics['val'] / val_dataloader.batch_sampler.batch_size, epoch)

        if batch_expansion_th is not None:
            # Dynamic batch expansion
            epoch_train_stats = stats['train'][-1]
            if 'num_non_zero_triplets' not in epoch_train_stats:
                logger.warning(
                    'Batch size expansion is enabled, but the loss function is not supported')
            else:
                # Ratio of non-zero triplets
                rnz = epoch_train_stats['num_non_zero_triplets'] / \
                    epoch_train_stats['num_triplets']
                if rnz < batch_expansion_th:
                    train_dataloader.batch_sampler.expand_batch()

        if stats['val'][-1]['loss'] * nz_metrics['val'] < best_val:
            logger.info(
                f"Found better model with non-zero triplet sum: {stats['val'][-1]['loss'] * nz_metrics['val']}, saving it...")
            best_val = stats['val'][-1]['loss'] * nz_metrics['val']
            if name is None:
                torch.save(teacher.state_dict(), os.path.join(
                    setup['general']['save_dir'], 'models', 'teacher2d_v.pth'))
            else:
                torch.save(teacher.state_dict(), os.path.join(
                    setup['general']['save_dir'], 'models', name + '_v.pth'))

        if stats['val'][-1]['loss'] < best_val_nz:
            logger.info(
                f"Found better model with non-zero triplet mean: {stats['val'][-1]['loss']}, saving it...")
            best_val_nz = stats['val'][-1]['loss']
            if name is None:
                torch.save(teacher.state_dict(), os.path.join(
                    setup['general']['save_dir'], 'models', 'teacher2d_vnz.pth'))
            else:
                torch.save(teacher.state_dict(), os.path.join(
                    setup['general']['save_dir'], 'models', name + '_vnz.pth'))

        if name is None:
            torch.save(teacher.state_dict(), os.path.join(
                setup['general']['save_dir'], 'models', 'teacher2d_last.pth'))
        else:
            torch.save(teacher.state_dict(), os.path.join(
                setup['general']['save_dir'], 'models', name + '_last.pth'))


def test(model, setup, val_dataloader, val_dataset, device, writer=None):
    model.eval()
    with torch.no_grad():
        if setup['model']['parameters']['output_dim'] is not None:
            pool_size = setup['model']['parameters']['output_dim']
        else:
            pool_size = setup['model']['parameters']['num_clusters'] * \
                setup['model']['parameters']['encoder_dim']
        dbFeat = np.empty((len(val_dataset), pool_size))

        pbar = tqdm(val_dataloader)
        pbar.set_description(f'Inference')
        for iteration, data in enumerate(pbar):
            img_anchor, indices = data
            img_anchor = img_anchor.to(device)
            emb_a = model(img_anchor)
            dbFeat[indices.detach().numpy(), :] = emb_a.detach().cpu().numpy()

    gt = []
    qids_to_tb = random.choices(range(len(val_dataset)), k=25)
    for i in range(len(val_dataset)):
        gt.append(val_dataset.get_positives_ndx(i))

    recall, top1_similarity_score, one_percent_recall = get_recall(
        dbFeat, dbFeat, gt, delete_first=True, writer=writer, qids_to_tb=qids_to_tb, val_dataset=val_dataset)


def get_recall(database_output, queries_output, gt, writer=None, delete_first=False, qids_to_tb=None, val_dataset=None):
    database_nbrs = KDTree(database_output)

    num_neighbors = 25

    recall = [0] * num_neighbors

    if delete_first:
        num_neighbors += 1

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(database_output.shape[0] / 100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        true_neighbors = gt[i]
        if (len(true_neighbors) == 0):
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]), k=num_neighbors)

        if delete_first:
            indices = indices[:, 1:]

        if i in qids_to_tb:
            if writer is not None:
                # for random_idx in qids_to_tb:
                random_idx = i
                # Get the first 5 predictions
                random_idx_preds = indices[0][:10]
                # random_dist = dist[0][:10]
                show_imgs = []
                img_q = torchvision.io.read_image(
                    val_dataset.queries[random_idx]['query_img'])
                box = torch.tensor([[0, 0, 320, 240]], dtype=torch.float)
                color = ["yellow"]
                img_q = torchvision.utils.draw_bounding_boxes(
                    img_q, boxes=box, colors=color, width=10)
                show_imgs.append(img_q)
                # print(f"Database length: {len(db_data)}")
                # is_correct = False
                for pred_idx in random_idx_preds:
                    # print(f"Prediction idx: {pred_idx}")
                    img = torchvision.io.read_image(
                        val_dataset.queries[pred_idx]['query_img'])
                    if np.any(gt[random_idx] == pred_idx):
                        # is_correct = True
                        box = torch.tensor(
                            [[0, 0, 320, 240]], dtype=torch.float)
                        color = ["green"]
                        img = torchvision.utils.draw_bounding_boxes(
                            img, boxes=box, colors=color, width=10)
                    show_imgs.append(img)

                # if len(gt[random_idx]) != 0:
                #     for gt_idx in gt[random_idx]:
                #         box = torch.tensor([[0, 0, 320, 240]], dtype=torch.float)
                #         color = ["red"]
                #         img = torchvision.io.read_image(db_data.db[db_data.db_index][gt_idx]['img_path'])
                #         img = torchvision.utils.draw_bounding_boxes(img, boxes=box, colors=color, width=10)
                #         show_imgs.append(img)

                img_grid = torchvision.utils.make_grid(show_imgs)
                writer.add_image(
                    f'{val_dataset.name}/ID:{random_idx} Recall@10', img_grid)
                # if is_correct:
                #     writer.add_image(f'SUCCESS/Q_{dates[q_data.test_index]}/DB_{dates[db_data.db_index]}/Dist_{random_dist}', img_grid)
                # else:
                #     writer.add_image(f'FAIL/Q_{dates[q_data.test_index]}/DB_{dates[db_data.db_index]}/Dist_{random_dist}', img_grid)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if (j == 0):
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = one_percent_retrieved / float(num_evaluated)
    recall = np.cumsum(recall) / float(num_evaluated)

    for i in range(25):
        # recalls[n] = recall_at_n[i]
        writer.add_scalar(
            f"{val_dataset.name}_Recall/Overall", recall[i], i + 1)
        logger.info("2D-2D Recall@{}: {:.4f} %".format(i + 1, recall[i] * 100))

    logger.info("2D-2D Recall@1%: {:.4f} %".format(one_percent_recall * 100))

    return recall, top1_similarity_score, one_percent_recall


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Train the 2d-teacher network')
    parser.add_argument('--config', type=str,
                        default="../setup/teacher_setup.yml")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    # parser.add_argument('--train_pickle_dir', type=str, default="/storage/user/lyun/Oxford_Robocar/dataset_every_5m_45runs/vxp_training_queries_baseline_p10_n25_yaw.pickle", help="Path to the training pickle file.")
    # parser.add_argument('--val_pickle_dir', type=str, default="/storage/user/lyun/Oxford_Robocar/dataset_every_5m_45runs/vxp_test_queries_baseline_p10_n25_yaw.pickle", help="Path to the validation pickle file.")
    # parser.add_argument('--save_dir', type=str, default="/storage/user/lyun/Oxford_Robocar/dataset_every_5m_45runs/", help="Path to the saving directory, model.pth and setup.yml would be saved into a sub-directory named /models.")
    # parser.add_argument('--name', type=str, default="dinogem", help="Name of the model")
    

    args = parser.parse_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    setup = load_setup_file(args.config)
    # setup['general']['train_pickle_dir'] = args.train_pickle_dir
    # setup['general']['val_pickle_dir'] = args.val_pickle_dir
    # setup['general']['save_dir'] = args.save_dir
    # setup['general']['name'] = args.name
    
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join(setup['general']['save_dir'], 'log')
    model_dir = os.path.join(setup['general']['save_dir'], 'models')\

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    logging.basicConfig(format='[%(levelname)s] [%(name)s] %(asctime)s: %(message)s', filename=os.path.join(
        log_dir, f'train_teacher_{current_time}.log'), level=level)

    transforms = load_data_augmentation(setup['dataset']['preprocessing'])
    transforms = torchvision.transforms.Compose(transforms)

    if not args.cpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    logger.info(f"Using {device}")

    logger.info("Loading the training dataset")

    if setup['dataset']['data_augmentation'] is not None:
        data_augmentation = load_data_augmentation(
            setup['dataset']['data_augmentation'])
        data_augmentation_tranforms = torchvision.transforms.Compose(
            data_augmentation
        )
        train_dataset = OxfordImageDataset(query_filepath=setup['general']['train_pickle_dir'],
                                            transform=transforms,
                                            max_elems=None, use_undistorted=setup['dataset']['use_undistorted'],
                                            data_augmentation=data_augmentation_tranforms)
    else:
        train_dataset = OxfordImageDataset(query_filepath=setup['general']['train_pickle_dir'],
                                            transform=transforms,
                                            max_elems=None, use_undistorted=setup['dataset']['use_undistorted']
                                            )

    train_sampler = BatchSampler(train_dataset, batch_size=setup['dataset']['batch_size'],
                                    batch_size_limit=setup['dataset']['batch_size_limit'],
                                    batch_expansion_rate=setup['dataset']['batch_expansion_rate'])

    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=make_collate_fn(train_dataset),
                                    num_workers=setup['dataset']['num_workers'], pin_memory=True)

    logging.info(f"Successfully load {len(train_dataset)} data")

    logging.info("Loading the validation dataset")

    val_dataset = OxfordImageDataset(query_filepath=setup['general']['val_pickle_dir'],
                                        transform=transforms,
                                        max_elems=None, use_undistorted=setup['dataset']['use_undistorted'])

    val_sampler = BatchSampler(
        val_dataset, batch_size=setup['dataset']['batch_size'])

    val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=make_collate_fn(val_dataset),
                                num_workers=setup['dataset']['num_workers'], pin_memory=True)

    logging.info(f"Successfully load {len(val_dataset)} data")

    teacher = model_factory_v2(model_collection=model,
                                model_setup=setup['model'])
    if setup['model']['pretrained'] is not None:
        teacher = load_pretrained_weight(
            teacher, os.path.join('..', 'pretrained', setup['model']['pretrained']), device)
    teacher = teacher.to(device)

    loss = make_loss(setup['loss']['loss_fn'], setup['loss']['parameters'])

    opt_class = getattr(torch.optim, setup['optimizer']['fn'])
    opt = opt_class(params=teacher.parameters(), **
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
            # setup['scheduler']['parameters'] = {'lr_lambda': lambda_fn}
        else:
            scheduler = scheduler_class(
                opt, **setup['scheduler']['parameters'])
    else:
        scheduler = None

    if setup['general']['name'] is None:
        name = f'teacher2d_{current_time}'
    else:
        name = setup['general']['name'] + f'_{current_time}'

    setup['general']['name'] = name
    setup['model']['path'] = os.path.join(model_dir, name + '.pth')
    save_setup_file(setup=setup, path=os.path.join(
        model_dir, 'setup_' + name + '.yml'))
    # with open(os.path.join(model_dir, 'setup_' + name + '.yml'), 'w') as yaml_file:
    #     yaml.dump(setup, yaml_file, default_flow_style=False)

    writer = SummaryWriter(os.path.join('..', 'tb_runs', name))

    train(teacher, opt, loss, scheduler, setup, train_dataloader, val_dataloader, val_dataset, device, name,
            writer, setup['optimizer']['recall_interval'], args.debug, setup['dataset']['batch_expansion_th'])
