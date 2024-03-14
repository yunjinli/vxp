# Refer https://github.com/KamilZywanowski/MinkLoc3D-SI/blob/master/models/loss.py
# Original author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import LpDistance
import logging
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


def make_loss(loss_type, params):
    if loss_type == 'BatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = BatchHardTripletLossWithMasks(**params)
    elif loss_type == 'CrossBatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = CrossBatchHardTripletLossWithMasks(**params)
    elif loss_type == 'ContrastiveLoss':
        loss_fn = ContrastiveLoss(**params)
    elif loss_type == 'BatchHardContrastiveLoss':
        loss_fn = BatchHardContrastiveLossWithMasks(**params)
    else:
        logger.error('Unknown loss: {}'.format(loss_type))
        raise NotImplementedError
    return loss_fn


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        # self.metric = metric
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, out0, out1, label):
        gt = label.float()
        D = self.distance(out0, out1).float().squeeze()
        loss = gt * 0.5 * torch.pow(D, 2) + (1 - gt) * 0.5 * \
            torch.pow(torch.clamp(self.margin - D, min=0.0), 2)
        return loss


class HardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance):
        self.distance = distance
        # Stats
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, embeddings, positives_mask, negatives_mask):
        assert embeddings.dim() == 2
        d_embeddings = embeddings.detach()
        with torch.no_grad():
            hard_triplets = self.mine(
                d_embeddings, positives_mask, negatives_mask)
        return hard_triplets

    def mine(self, embeddings, positives_mask, negatives_mask):
        # Based on pytorch-metric-learning implementation
        # print(embeddings.shape)
        dist_mat = self.distance(embeddings)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(
            dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(
            dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(
            hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        self.max_pos_pair_dist = torch.max(hardest_positive_dist).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist).item()
        return a, p, n


class CrossHardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance):
        self.distance = distance
        # Stats
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, embeddings, positives_mask, negatives_mask, batch_size):
        assert embeddings.dim() == 2
        d_embeddings = embeddings.detach()
        with torch.no_grad():
            hard_triplets = self.mine(
                d_embeddings, positives_mask, negatives_mask, batch_size)
        return hard_triplets

    def mine(self, embeddings, positives_mask, negatives_mask, batch_size):
        # Based on pytorch-metric-learning implementation
        dist_mat = self.distance(embeddings)
        logger.debug(f"Distance matrix: {dist_mat}")

        logger.debug(f"positives_mask: {positives_mask}")
        (hardest_positive_dist_2d3d, hardest_positive_indices_2d3d), a1p_keep = get_max_per_row(
            dist_mat[:batch_size, batch_size:], positives_mask)
        (hardest_positive_dist_3d2d, hardest_positive_indices_3d2d), _ = get_max_per_row(
            dist_mat[batch_size:, :batch_size], positives_mask)
        logger.debug(
            f"hardest_positive_indices_2d3d: {hardest_positive_indices_2d3d}")
        logger.debug(
            f"hardest_positive_indices_3d2d: {hardest_positive_indices_3d2d}")
        logger.debug(f"a1p_keep: {a1p_keep}")

        hardest_positive_dist = torch.concat(
            [hardest_positive_dist_2d3d, hardest_positive_dist_3d2d], dim=0)
        logger.debug(f"hardest_positive_dist: {hardest_positive_dist}")
        logger.debug(f"negatives_mask: {negatives_mask}")
        (hardest_negative_dist_2d3d, hardest_negative_indices_2d3d), a2n_keep = get_min_per_row(
            dist_mat[:batch_size, batch_size:], negatives_mask)
        (hardest_negative_dist_3d2d, hardest_negative_indices_3d2d), _ = get_min_per_row(
            dist_mat[batch_size:, :batch_size], negatives_mask)
        logger.debug(
            f"hardest_negative_indices_2d3d: {hardest_negative_indices_2d3d}")
        logger.debug(
            f"hardest_negative_indices_3d2d: {hardest_negative_indices_3d2d}")
        logger.debug(f"a2n_keep: {a2n_keep}")

        hardest_negative_dist = torch.concat(
            [hardest_negative_dist_2d3d, hardest_negative_dist_3d2d], dim=0)
        logger.debug(f"hardest_negative_dist: {hardest_negative_dist}")
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        logger.debug(f"a_keep_idx: {a_keep_idx}")

        a_2d3d = torch.arange(dist_mat.size(0)).to(
            hardest_positive_indices_2d3d.device)[a_keep_idx]
        p_2d3d = hardest_positive_indices_2d3d[a_keep_idx] + batch_size
        n_2d3d = hardest_negative_indices_2d3d[a_keep_idx] + batch_size

        a_3d2d = torch.arange(dist_mat.size(0)).to(
            hardest_positive_indices_3d2d.device)[a_keep_idx] + batch_size
        p_3d2d = hardest_positive_indices_3d2d[a_keep_idx]
        n_3d2d = hardest_negative_indices_3d2d[a_keep_idx]

        a = torch.concat([a_2d3d, a_3d2d], dim=0)
        p = torch.concat([p_2d3d, p_3d2d], dim=0)
        n = torch.concat([n_2d3d, n_3d2d], dim=0)

        self.max_pos_pair_dist = torch.max(hardest_positive_dist).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist).item()
        return a, p, n


def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows


class BatchHardTripletLossWithMasks:
    def __init__(self, margin, normalize_embeddings):
        self.margin = margin
        self.normalize_embeddings = normalize_embeddings
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings)
        # We use triplet loss with Euclidean distance
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        self.loss_fn = losses.TripletMarginLoss(
            margin=self.margin, swap=True, distance=self.distance)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(
            embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
                 'num_triplets': len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats, hard_triplets


class CrossBatchHardTripletLossWithMasks:
    def __init__(self, margin, normalize_embeddings):
        self.margin = margin
        self.normalize_embeddings = normalize_embeddings
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings)
        # We use triplet loss with Euclidean distance
        self.miner_fn = CrossHardTripletMinerWithMasks(distance=self.distance)
        self.loss_fn = losses.TripletMarginLoss(
            margin=self.margin, swap=True, distance=self.distance)

    def __call__(self, img_embs, submap_embs, positives_mask, negatives_mask):
        assert img_embs.shape[0] == submap_embs.shape[0]
        batch_size = img_embs.shape[0]
        embeddings = torch.concat([img_embs, submap_embs], dim=0)
        logger.debug(f"Concatenated embeddings size = {embeddings.shape}")
        hard_triplets = self.miner_fn(
            embeddings, positives_mask, negatives_mask, batch_size)
        logger.debug(f"Computed hard triplets = {hard_triplets}")
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
                 'num_triplets': len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats, hard_triplets


class BatchHardContrastiveLossWithMasks:
    def __init__(self, pos_margin, neg_margin, normalize_embeddings):
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        # We use contrastive loss with squared Euclidean distance
        self.loss_fn = losses.ContrastiveLoss(pos_margin=self.pos_margin, neg_margin=self.neg_margin,
                                              distance=self.distance)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(
            embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'pos_pairs_above_threshold': self.loss_fn.reducer.reducers['pos_loss'].pos_pairs_above_threshold,
                 'neg_pairs_above_threshold': self.loss_fn.reducer.reducers['neg_loss'].neg_pairs_above_threshold,
                 'pos_loss': self.loss_fn.reducer.reducers['pos_loss'].pos_loss.item(),
                 'neg_loss': self.loss_fn.reducer.reducers['neg_loss'].neg_loss.item(),
                 'num_pairs': 2*len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats, hard_triplets
