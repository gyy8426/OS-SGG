from lib.word_vectors import obj_edge_vectors
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
from config import DATA_PATH
import os
from lib.get_dataset_counts import get_counts


class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, eps=1e-3, graph=False, bias = 1.0):
        super(FrequencyBias, self).__init__()
        fg_matrix, bg_matrix = get_counts(must_overlap=True)
        print('before fg_matrix min: ', fg_matrix.min(), 'fg_matrix max: ', fg_matrix.max())
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        print('after fg_matrix min: ', fg_matrix.min(), 'fg_matrix max: ', fg_matrix.max())
        if graph:
            fg_matrix = np.concatenate([fg_matrix[:, :, 0][:,:,None], (fg_matrix[:, :, 1:].sum(-1))[:,:,None]], -1)
        pred_dist = np.log( bias * fg_matrix / ( fg_matrix.sum(2)[:, :, None] + 1e-8 ) + eps )
        #if not with_bg:
            #pred_dist[0] = 0.0
        self.num_objs = pred_dist.shape[0]
        pred_dist = torch.FloatTensor(pred_dist).view(-1, pred_dist.shape[2])

        self.obj_baseline = nn.Embedding(pred_dist.size(0), pred_dist.size(1))
        self.obj_baseline.weight.data = pred_dist
        #self.obj_baseline.weight.requires_grad = False

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """

        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def forward(self, obj_cands0, obj_cands1):
        """
        :param obj_cands0: [batch_size, 151] prob distibution over cands.
        :param obj_cands1: [batch_size, 151] prob distibution over cands.
        :return: [batch_size, #predicates] array, which contains potentials for
        each possibility
        """
        # [batch_size, 151, 151] repr of the joint distribution
        joint_cands = obj_cands0[:, :, None] * obj_cands1[:, None]

        # [151, 151, 51] of targets per.
        baseline = joint_cands.view(joint_cands.size(0), -1) @ self.obj_baseline.weight

        return baseline
