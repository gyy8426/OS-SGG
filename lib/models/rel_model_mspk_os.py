"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM
from lib.fpn.nms.functions.nms import apply_nms

# from lib.decoder_rnn import DecoderRNN, lstm_factory, LockedDropout
from lib.lstm.decoder_rnn import DecoderRNN
from lib.lstm.highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM
from lib.fpn.box_utils import bbox_overlaps, center_size, nms_overlaps
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg, load_resnet
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, \
    Flattener
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors, obj_edge_rel_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
import math

from lib.attention.bert import BERT
from lib.gcn.pygcn import GraphConvolution as GCN
from lib.utils.prepare_feat_bert import prepare_feat, postdiso_feat
from lib.get_dataset_counts import get_counts
from lib.utils.gelu import GELU
import json
from lib.pytorch_misc import to_variable


def _sort_by_score(im_inds, scores):
    """
    We'll sort everything scorewise from Hi->low, BUT we need to keep images together
    and sort LSTM from l
    :param im_inds: Which im we're on
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
    :return: Permutation to put everything in the right order for the LSTM
             Inverse permutation
             Lengths for the TxB packed sequence.
    """
    num_im = im_inds[-1] + 1
    rois_per_image = scores.new(num_im)
    lengths = []
    for i, s, e in enumerate_by_image(im_inds):
        rois_per_image[i] = 2 * (s - e) * num_im + i
        lengths.append(e - s)
    lengths = sorted(lengths, reverse=True)
    inds, ls_transposed = transpose_packed_sequence_inds(lengths)  # move it to TxB form
    inds = torch.LongTensor(inds).cuda(im_inds.get_device())

    # ~~~~~~~~~~~~~~~~
    # HACKY CODE ALERT!!!
    # we're sorting by confidence which is in the range (0,1), but more importantly by longest
    # img....
    # ~~~~~~~~~~~~~~~~
    roi_order = scores - 2 * rois_per_image[im_inds]
    _, perm = torch.sort(roi_order, 0, descending=True)
    perm = perm[inds]
    _, inv_perm = torch.sort(perm)

    return perm, inv_perm, ls_transposed


MODES = ('sgdet', 'sgcls', 'predcls')


class LinearizedContext(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, classes, rel_classes, mode='sgdet',
                 embed_dim=200, hidden_dim=256, obj_dim=2048,
                 nl_obj=12, nh_obj=12, nl_edge=12, nh_edge=12,
                 nl_gcn_obj=2, nl_gcn_edge=2,
                 dropout_rate=0.2, order='confidence',
                 pass_in_obj_feats_to_decoder=True,
                 pass_in_obj_feats_to_edge=True):
        super(LinearizedContext, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        assert mode in MODES
        self.mode = mode

        self.nl_obj = nl_obj
        self.nl_edge = nl_edge

        self.dim_obj_hidden = hidden_dim
        self.nl_obj = nl_obj
        self.nh_obj = nh_obj
        self.nl_gcn_obj = nl_gcn_obj

        self.dim_edge_hidden = hidden_dim
        self.nl_edge = nl_edge
        self.nh_edge = nh_edge
        self.nl_gcn_edge = nl_gcn_edge
        self.embed_dim = embed_dim
        self.obj_dim = obj_dim
        self.dropout_rate = dropout_rate
        self.pass_in_obj_feats_to_decoder = pass_in_obj_feats_to_decoder
        self.pass_in_obj_feats_to_edge = pass_in_obj_feats_to_edge

        self.nl_gcn_concept = 2
        assert order in ('size', 'confidence', 'random', 'leftright')
        self.order = order

        # EMBEDDINGS
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)

        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed_in_edge = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed_in_edge.weight.data = embed_vecs.clone()
        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        ])
        obj_ctx_indim = self.obj_dim + self.embed_dim + 128
        if self.nl_obj > 0 and self.mode != 'predcls_nongtbox':
            self.obj_ctx_bert = BERT(
                input_dim=obj_ctx_indim,
                hidden_dim=self.dim_obj_hidden,
                n_layers=self.nl_obj,
                attn_heads=self.nh_obj,
                dropout=self.dropout_rate)

            self.obj_ctx_classifier = nn.Linear(self.dim_obj_hidden, self.num_classes)

        if self.nl_gcn_edge > 0 or self.nl_gcn_obj > 0:
            gcn_embed_dim = 300
            obj_rel_embed_vecs = obj_edge_vectors(self.classes + self.rel_classes, wv_dim=gcn_embed_dim)
            fg_matrix, bg_matrix = get_counts(must_overlap=True)
            fg_matrix[fg_matrix > 0] = 1
            num_gcn_node = self.num_classes + self.num_rels

            obj_adj_mat = np.zeros([num_gcn_node, num_gcn_node]).astype('float32')
            obj_adj_mat[:self.num_classes, :self.num_classes] = 1.0 * fg_matrix.max(-1)
            fg_ind = np.where(obj_adj_mat == 1)
            obj_adj_mat[fg_ind[1], fg_ind[0]] = 1.0

            rel_adj_mat = np.zeros([num_gcn_node, num_gcn_node]).astype('float32')
            rel_ind = np.where(fg_matrix == 1)
            rel_adj_mat[rel_ind[0], self.num_classes + rel_ind[2]] = 1.0
            rel_adj_mat[rel_ind[1], self.num_classes + rel_ind[2]] = 1.0
            rel_adj_mat[self.num_classes + rel_ind[2], rel_ind[0]] = 1.0
            rel_adj_mat[self.num_classes + rel_ind[2], rel_ind[1]] = 1.0
            self.obj_adj_mat = obj_adj_mat
            self.edge_adj_mat = rel_adj_mat

        if self.nl_gcn_obj > 0:
            self.obj_ctx_gcn = []
            obj_gcn_hidden_dim = self.dim_obj_hidden
            self.obj_gcn_input = obj_rel_embed_vecs.clone()
            self.obj_gcn_input_layer = nn.Linear(gcn_embed_dim, obj_gcn_hidden_dim)
            for i in range(self.nl_gcn_obj):
                self.obj_ctx_gcn.append(GCN(obj_gcn_hidden_dim, obj_gcn_hidden_dim))
            self.obj_ctx_gcn = nn.ModuleList(self.obj_ctx_gcn)
            # self.obj_ctx_gcn =  nn.Sequential(*self.obj_ctx_gcn)

        if self.mode == 'detclass':
            return
        # print('input_dim: ',input_dim)

        if self.nl_edge > 0:
            edge_ctx_indim = self.obj_dim + self.embed_dim + 128
            # edge_ctx_indim = self.dim_obj_hidden + self.embed_dim + 128
            self.edge_ctx_bert = BERT(
                input_dim=edge_ctx_indim,
                hidden_dim=self.dim_edge_hidden,
                n_layers=self.nl_edge,
                attn_heads=self.nh_edge,
                dropout=self.dropout_rate)
        if self.nl_gcn_edge > 0:
            self.edge_ctx_gcn = []
            edge_gcn_hidden_dim = self.dim_obj_hidden
            self.edge_gcn_input = obj_rel_embed_vecs.clone()
            self.edge_gcn_input_layer = nn.Linear(gcn_embed_dim, edge_gcn_hidden_dim)
            for i in range(self.nl_gcn_edge):
                self.edge_ctx_gcn.append(GCN(edge_gcn_hidden_dim, edge_gcn_hidden_dim))
            self.edge_ctx_gcn = nn.ModuleList(self.edge_ctx_gcn)
            self.gcn_act = GELU()
        if self.nl_gcn_concept > 0:
            concept_vocab_path = './OS-SGG_file/ConceptNet/concept.txt'
            relation_vocab_path = './OS-SGG_file/ConceptNet/relation.txt'
            concept2id = {}
            id2concept = {}
            with open(concept_vocab_path, "r", encoding="utf8") as f:
                for w in f.readlines():
                    concept2id[w.strip()] = len(concept2id)
                    id2concept[len(id2concept)] = w.strip()
            self.concept2id = concept2id
            self.id2concept = id2concept
            print("concept2id done")
            concept_id2relation = {}
            concept_relation2id = {}
            with open(relation_vocab_path, "r", encoding="utf8") as f:
                for w in f.readlines():
                    concept_id2relation[len(concept_id2relation)] = w.strip()
                    concept_relation2id[w.strip()] = len(concept_relation2id)
            self.concept_id2relation = concept_id2relation
            self.concept_relation2id = concept_relation2id
            print("relation2id done")
            pruned_pckle_file = "./OS-SGG_file/ConceptNet/visual_genome_concept_pruned_path.pf"
            self.pruned_path = json.load(open(pruned_pckle_file, 'r'))
            conceptvg_dict = json.load(
                open('./visual_genome/data/genome/concept_graph/conceptvg_dict.json',
                     'r'))
            self.concept2obj_ind = conceptvg_dict['concept2obj_ind']
            self.obj2concept_ind = conceptvg_dict['obj2concept_ind']
            self.concept2pred_ind = conceptvg_dict['concept2pred_ind']
            self.pred2concept_ind = conceptvg_dict['pred2concept_ind']
            all_concept = conceptvg_dict['all_concept']

            vg_sgg_dicts = json.load(
                open('./OS-SGG_file/ConceptNet/VG-SGG-dicts.json', 'r'))

            id2obj = vg_sgg_dicts['idx_to_label']
            obj2id = vg_sgg_dicts['label_to_idx']
            id2pred = vg_sgg_dicts['idx_to_predicate']
            pred2id = vg_sgg_dicts['predicate_to_idx']
            obj_class_str = list(id2obj.values())
            rel_class_str = list(id2pred.values())
            obj_class_inds = list([int(i) for i in id2obj.keys()])
            rel_class_inds = list([int(i) for i in id2pred.keys()])
            all_class_str = obj_class_str + rel_class_str
            # print(all_class_str)
            all_class_inds = obj_class_inds + rel_class_inds
            # print(all_class_inds)
            # print(all_class_inds)
            # obj_class2path_id = []
            all_class2path_id = {}
            self.num_all_path_class = len(all_class_str)
            count = 0
            count_f = 0
            self.obj2allcls = {}
            for i in range(len(all_class_str)):
                self.obj2allcls[all_class_inds[i]] = count_f
                count_f = count_f + 1
                for j in range(len(all_class_str)):
                    if i != j:
                        obji_ind = all_class_inds[i]
                        objj_ind = all_class_inds[j]
                        all_class2path_id[obji_ind * self.num_all_path_class + objj_ind] = count
                        count = count + 1
            self.obj_class2path_id = json.load(
                open('./OS-SGG_file/ConceptNet/obj2path.json', 'r'))

            kgnet_path = './OS-SGG_file/ConceptNet/'
            concept_embs = np.load(
                kgnet_path + "concept_glove.max.npy")
            concept_embs = torch.from_numpy(concept_embs)
            # concept_embed_vecs = obj_edge_vectors(all_concept, wv_dim=gcn_embed_dim)
            concept_ctx_gcn = []
            concept_gcn_embed_dim = concept_embs.size(-1)
            concpt_gcn_hidden_dim = self.dim_obj_hidden
            self.concpt_gcn_input = concept_embs.clone()
            self.concpt_gcn_input_layer = nn.Linear(concept_gcn_embed_dim, concpt_gcn_hidden_dim)
            for i in range(self.nl_gcn_concept):
                concept_ctx_gcn.append(GCN(concpt_gcn_hidden_dim, concpt_gcn_hidden_dim))
            self.concept_ctx_gcn = nn.ModuleList(concept_ctx_gcn)
            self.cpt_gcn_act = nn.ReLU()  # GELU()

    def sort_rois(self, batch_idx, confidence, box_priors):
        """
        :param batch_idx: tensor with what index we're on
        :param confidence: tensor with confidences between [0,1)
        :param boxes: tensor with (x1, y1, x2, y2)
        :return: Permutation, inverse permutation, and the lengths transposed (same as _sort_by_score)
        """
        cxcywh = center_size(box_priors)
        if self.order == 'size':
            sizes = cxcywh[:, 2] * cxcywh[:, 3]
            # sizes = (box_priors[:, 2] - box_priors[:, 0] + 1) * (box_priors[:, 3] - box_priors[:, 1] + 1)
            assert sizes.min() > 0.0
            scores = sizes / (sizes.max() + 1)
        elif self.order == 'confidence':
            scores = confidence
        elif self.order == 'random':
            scores = torch.FloatTensor(np.random.rand(batch_idx.size(0))).cuda(batch_idx.get_device())
        elif self.order == 'leftright':
            centers = cxcywh[:, 0]
            scores = centers / (centers.max() + 1)
        else:
            raise ValueError("invalid mode {}".format(self.order))
        return _sort_by_score(batch_idx, scores)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def get_concept_graph(self, obj_labels):
        obj_labels_np = obj_labels.data.cpu().numpy()
        obj_labels_unique = set(obj_labels_np)
        sub_graph_concpt = []
        sub_graph_concpt_cont = 0
        sub_graph_concpt_dict = {}
        sub_graph_concpt_edge = []

        if len(obj_labels_unique) == 1:
            sub_graph = np.zeros([1, 1], dtype=np.int32)
            unique_concept_id = self.obj2concept_ind[str(list(obj_labels_unique)[0])]
            sub_graph_concpt_dict[unique_concept_id] = len(sub_graph_concpt)
            sub_graph_concpt.append(unique_concept_id)
        ifprint = False
        if len(obj_labels_unique) > 1:
            for i in obj_labels_unique:
                for j in obj_labels_unique:
                    if i == j:
                        continue
                    # all_cls_ind_i = self.obj2allcls[i]
                    # all_cls_ind_j = self.obj2allcls[j]
                    path_id = self.obj_class2path_id[str(i * (
                        self.num_all_path_class) + j)]  # all_cls_ind_i * (self.num_all_path_class - 1 ) +  all_cls_ind_j #
                    path_i = self.pruned_path[path_id]["pf_res"]  # all paths between i and j
                    if ifprint:
                        print('obj i: ', self.classes[i], 'obj j: ', self.classes[j])
                        print('obj source: ', self.pruned_path[path_id]["source"], 'obj target: ',
                              self.pruned_path[path_id]["target"])
                    # print('path_i: ',path_i)
                    source_id = self.concept2id[self.pruned_path[path_id]["source"]]
                    target_id = self.concept2id[self.pruned_path[path_id]["target"]]
                    if source_id not in sub_graph_concpt:
                        # print(source_id)
                        sub_graph_concpt_dict[source_id] = len(sub_graph_concpt)
                        sub_graph_concpt.append(source_id)
                        sub_graph_concpt_cont = sub_graph_concpt_cont + 1
                    if target_id not in sub_graph_concpt:
                        sub_graph_concpt_dict[target_id] = len(sub_graph_concpt)
                        sub_graph_concpt.append(target_id)
                        sub_graph_concpt_cont = sub_graph_concpt_cont + 1
                    if path_i is not None:
                        sub_graph_concpt_edge.append(
                            [sub_graph_concpt_dict[source_id], sub_graph_concpt_dict[target_id]])
                        for path_indj in range(len(path_i)):
                            paths_j = path_i[path_indj]['path']  # the concenpts in the path_j
                            rel_j = path_i[path_indj]['rel']
                            for path_indk in range(len(paths_j) - 1):
                                sourcej_id = paths_j[path_indk]
                                targetj_id = paths_j[path_indk + 1]

                                if sourcej_id not in sub_graph_concpt:
                                    # print(source_id)
                                    sub_graph_concpt_dict[sourcej_id] = len(sub_graph_concpt)
                                    sub_graph_concpt.append(sourcej_id)
                                    sub_graph_concpt_cont = sub_graph_concpt_cont + 1
                                if targetj_id not in sub_graph_concpt:
                                    sub_graph_concpt_dict[targetj_id] = len(sub_graph_concpt)
                                    sub_graph_concpt.append(targetj_id)
                                    sub_graph_concpt_cont = sub_graph_concpt_cont + 1
                                sub_graph_concpt_edge.append(
                                    [sub_graph_concpt_dict[sourcej_id], sub_graph_concpt_dict[targetj_id]])
                                rel_list = rel_j[path_indk]

                                if ifprint:
                                    rel_list_str = []
                                    for rel in rel_list:
                                        if rel < len(self.concept_id2relation):
                                            rel_list_str.append(self.concept_id2relation[rel])
                                        else:
                                            rel_list_str.append(
                                                self.concept_id2relation[rel - len(self.concept_id2relation)] + "*")
                                    print(self.id2concept[sourcej_id], "----[%s]---> " % ("/".join(rel_list_str)),
                                          end="")
                                    if path_indk + 1 == len(paths_j) - 1:
                                        print(self.id2concept[targetj_id], end="")
                            if ifprint:
                                print(' ')
            # print('sub_graph_concpt: ',sub_graph_concpt)

            sub_graph = np.zeros([len(sub_graph_concpt), len(sub_graph_concpt)], dtype=np.int32)
            sub_graph_concpt = np.array(sub_graph_concpt, dtype=np.int32)
            for edge_i in sub_graph_concpt_edge:
                sub_graph[edge_i[0], edge_i[1]] = 1
        obj2subgraph_ind_np = np.zeros([len(obj_labels_np)], dtype=np.int32)
        # print(sub_graph_concpt_dict)
        for ind, obj_labels_ind_i in enumerate(obj_labels_np):
            cpt_ind = self.obj2concept_ind[str(obj_labels_ind_i)]
            if cpt_ind in sub_graph_concpt_dict.keys():
                obj2subgraph_ind_np[ind] = sub_graph_concpt_dict[cpt_ind]

        return torch.LongTensor(sub_graph).cuda(obj_labels.get_device(), async=True), \
               torch.LongTensor(sub_graph_concpt).cuda(obj_labels.get_device(), async=True), \
               torch.LongTensor(obj2subgraph_ind_np).cuda(obj_labels.get_device(), async=True)

    def concept_gcn_func(self, obj_preds, im_inds):
        sub_graph_mat = []
        sub_graph_concpt = []
        obj_in_subgraph = []
        graph_offset = 0
        for i, s, e in enumerate_by_image(im_inds.data):
            sub_graph_mat_i, sub_graph_concpt_i, obj_in_subgraph_i = self.get_concept_graph(obj_preds[s:e])
            if len(sub_graph_mat_i) == 0:
                continue
            sub_graph_mat.append(sub_graph_mat_i)
            sub_graph_concpt.append(sub_graph_concpt_i)
            obj_in_subgraph.append(obj_in_subgraph_i + graph_offset)
            graph_offset = graph_offset + sub_graph_concpt_i.size(0)
        sub_graph_concpt = torch.cat(sub_graph_concpt)
        obj_in_subgraph = torch.cat(obj_in_subgraph)
        batch_graph_mat = torch.zeros([graph_offset, graph_offset])
        start_ind = 0
        for sub_graph_mat_i in sub_graph_mat:
            sub_graph_len_i = sub_graph_mat_i.size(0)
            end_ind = start_ind + sub_graph_len_i
            batch_graph_mat[start_ind:end_ind, start_ind:end_ind] = sub_graph_mat_i
            start_ind = end_ind
        batch_graph_mat = (batch_graph_mat + batch_graph_mat.permute(1, 0)) > 0
        batch_graph_mat = batch_graph_mat.float()
        # np.set_printoptions(threshold=np.inf)
        # print('im_inds')
        # print(im_inds)
        # print('obj_preds: ')
        # print(obj_preds.data.cpu().numpy())
        # print('sub_graph_concpt: ')
        # print( sub_graph_concpt.cpu().numpy())
        # print('obj_in_subgraph: ')
        # print( obj_in_subgraph.cpu().numpy())
        # print('batch_graph_mat: ')
        # print(batch_graph_mat[:20,:20].cpu().numpy())
        sub_graph_concpt = Variable(sub_graph_concpt).cuda(obj_preds.get_device())
        obj_in_subgraph = Variable(obj_in_subgraph).cuda(obj_preds.get_device())
        batch_graph_mat = Variable(batch_graph_mat).cuda(obj_preds.get_device())
        concpt_gcn_input = Variable(self.concpt_gcn_input).cuda(obj_preds.get_device())
        sub_graph_concpt_embed = concpt_gcn_input[sub_graph_concpt.data]
        sub_graph_concpt_embed = self.concpt_gcn_input_layer(sub_graph_concpt_embed)
        concept_gcn_layer_input = F.dropout(sub_graph_concpt_embed, self.dropout_rate, training=self.training)
        # concept_gcn_layer_input = sub_graph_concpt_embed
        count_gcn = 0
        for m in self.concept_ctx_gcn:
            # if count_gcn < 2:
            #     input_adj_mat = sub_graph_mat
            #     #print('input_adj_lap obj_adj_lap')
            # else:
            #     input_adj_mat = sub_graph_mat
            #     #print('input_adj_lap edge_adj_lap')
            input_adj_mat = batch_graph_mat
            input_adj_mat = F.dropout(input_adj_mat, 0.6, training=self.training)
            input_adj_lap = self.adj_to_Laplacian(input_adj_mat)
            concept_gcn_layer_output = m(concept_gcn_layer_input, input_adj_lap)
            concept_gcn_output = concept_gcn_layer_output + concept_gcn_layer_input
            # edge_gcn_output = self.cpt_gcn_act(edge_gcn_output)
            concept_gcn_output = F.dropout(concept_gcn_output, self.dropout_rate, training=self.training)
            count_gcn = count_gcn + 1
            concept_gcn_layer_input = concept_gcn_output

        return concept_gcn_output[obj_in_subgraph.data]

    def adj_to_Laplacian(self, adj_mat, type='hard'):
        eye_mat = torch.eye(adj_mat.size(0)).cuda(adj_mat.get_device(), async=True)
        eye_mat = Variable(eye_mat)
        adj_mat = adj_mat + eye_mat
        degree_mat = adj_mat.sum(-1)
        degree_mat_re = 1 / (degree_mat + 1e-8)
        dot_1 = degree_mat_re[:, None] * adj_mat

        # print(dot_1.size(),degree_mat_re.size(),adj_mat.size())
        return dot_1

    def exchange_obj_edge_gcn_func(self, obj_preds, obj_dists):
        edge_gcn_input = Variable(self.edge_gcn_input).cuda(obj_preds.get_device())
        edge_gcn_input = self.edge_gcn_input_layer(edge_gcn_input)

        edge_adj_mat = torch.from_numpy(self.edge_adj_mat).cuda(obj_preds.get_device())
        edge_adj_mat = Variable(edge_adj_mat)
        # edge_adj_lap = self.adj_to_Laplacian(edge_adj_mat)

        obj_adj_mat = torch.from_numpy(self.obj_adj_mat).cuda(obj_preds.get_device())
        obj_adj_mat = Variable(obj_adj_mat)
        # obj_adj_lap = self.adj_to_Laplacian(obj_adj_mat)

        edge_gcn_layer_input = F.dropout(edge_gcn_input, self.dropout_rate, training=self.training)
        count_gcn = 0
        for m in self.edge_ctx_gcn:
            if count_gcn < 2:
                input_adj_mat = obj_adj_mat
                # print('input_adj_lap obj_adj_lap')
            else:
                input_adj_mat = edge_adj_mat
                # print('input_adj_lap edge_adj_lap')
            # input_adj_mat = edge_adj_mat
            input_adj_mat = F.dropout(input_adj_mat, 0.3, training=self.training)
            input_adj_lap = self.adj_to_Laplacian(input_adj_mat)
            edge_gcn_layer_output = m(edge_gcn_layer_input, input_adj_lap)
            edge_gcn_output = edge_gcn_layer_output + edge_gcn_layer_input
            # edge_gcn_output = self.gcn_act(edge_gcn_output)
            edge_gcn_output = F.dropout(edge_gcn_output, self.dropout_rate, training=self.training)
            count_gcn = count_gcn + 1
            edge_gcn_layer_input = edge_gcn_output
        # return obj_dists @ edge_gcn_output[:self.num_classes, :]
        return edge_gcn_output[obj_preds.data, :]

    def edge_gcn(self, obj_preds, obj_dists):
        edge_gcn_input = Variable(self.edge_gcn_input).cuda(obj_preds.get_device())
        edge_gcn_input = self.edge_gcn_input_layer(edge_gcn_input)
        edge_adj_mat = torch.from_numpy(self.edge_adj_mat).cuda(obj_preds.get_device())
        edge_adj_mat = Variable(edge_adj_mat)
        edge_adj_lap = self.adj_to_Laplacian(edge_adj_mat)
        edge_gcn_layer_input = edge_gcn_input.clone()
        count_gcn = 0
        for m in self.edge_ctx_gcn:
            edge_gcn_layer_output = m(edge_gcn_layer_input, edge_adj_lap)
            edge_gcn_output = edge_gcn_layer_output + edge_gcn_layer_input
            count_gcn = count_gcn + 1
            edge_gcn_layer_input = edge_gcn_output
        return obj_dists @ edge_gcn_output[:self.num_classes, :]

    def edge_ctx(self, obj_feats, im_inds, obj_preds):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :return: edge_ctx: [num_obj, #feats] For later!
        """

        # Only use hard embeddings
        obj_feats_t, mask = prepare_feat(obj_feats, im_inds.data)
        edge_reps = self.edge_ctx_bert(obj_feats_t, mask)
        edge_reps = postdiso_feat(edge_reps, im_inds.data)
        # now we're good! unperm
        return edge_reps

    def obj_gcn(self, obj_preds, obj_dists):
        obj_gcn_input = Variable(self.obj_gcn_input).cuda(obj_preds.get_device())
        obj_gcn_input = self.obj_gcn_input_layer(obj_gcn_input)
        obj_adj_mat = torch.from_numpy(self.obj_adj_mat).cuda(obj_preds.get_device())
        obj_adj_mat = Variable(obj_adj_mat)
        obj_adj_lap = self.adj_to_Laplacian(obj_adj_mat)
        obj_gcn_input_m = obj_gcn_input.clone()
        for m in self.obj_ctx_gcn:
            obj_gcn_input_m = m(obj_gcn_input_m, obj_adj_lap)
        obj_gcn_output = obj_gcn_input_m
        return obj_dists @ (obj_gcn_output + obj_gcn_input)[:self.num_classes, :]

    def obj_ctx(self, obj_feats, obj_dists, obj_labels=None,
                boxes_per_cls=None, im_inds=None, obj_preds=None,
                ):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :param obj_labels: [num_obj] the GT labels of the image
        :param boxes: [num_obj, 4] boxes. We'll use this for NMS
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # Sort by the confidence of the maximum detection.
        prior_obj_dists = obj_dists.clone()
        obj_feats_t, mask = prepare_feat(obj_feats, im_inds.data)
        obj_ctx_rep = self.obj_ctx_bert(obj_feats_t, mask)
        obj_ctx_rep = postdiso_feat(obj_ctx_rep, im_inds.data)
        if self.nl_gcn_obj > 0:
            obj_gcn_output = self.obj_gcn(
                obj_preds=obj_preds,
                obj_dists=F.softmax(prior_obj_dists, dim=1),
            )
            obj_ctx_rep = obj_gcn_output * obj_ctx_rep
        obj_dists_refine = self.obj_ctx_classifier(obj_ctx_rep)

        if prior_obj_dists is not None:
            # prior_obj_dists_drop = F.dropout(prior_obj_dists, self.dropout_rate, training=self.training)
            # obj_dists = obj_dists_refine + prior_obj_dists_drop
            obj_dists_2 = obj_dists_refine + prior_obj_dists
        else:
            obj_dists_2 = obj_dists_refine
        boxes_for_nms = boxes_per_cls

        if self.training:
            nms_labels = obj_labels
            nonzero_pred = obj_dists_2[:, 1:].max(1)[1] + 1
            is_bg = (nms_labels.data == 0).nonzero()
            if len(is_bg) > 0:
                nms_labels[is_bg.squeeze(1)] = nonzero_pred[is_bg.squeeze(1)]
        else:
            out_dist_sample = F.softmax(obj_dists_2, dim=1)
            nms_labels = out_dist_sample[:, 1:].max(1)[1] + 1

        if (boxes_for_nms is not None and not self.training):
            is_overlap = nms_overlaps(boxes_for_nms.data).view(
                boxes_for_nms.size(0), boxes_for_nms.size(0), boxes_for_nms.size(1)
            ).cpu().numpy() >= 0.3

            out_dists = obj_dists_2
            out_dists_sampled = F.softmax(out_dists, 1).data.cpu().numpy()
            out_dists_sampled[:, 0] = 0

            nms_labels = nms_labels[0].data.new(len(nms_labels)).fill_(0)

            for i in range(nms_labels.size(0)):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                nms_labels[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            output_labels = Variable(nms_labels)
        else:
            output_labels = nms_labels
        output_dists = obj_dists_2
        if self.mode == 'predcls':
            output_labels = obj_labels
            output_dists = Variable(to_onehot(output_labels.data, self.num_classes))
        return obj_ctx_rep, output_dists, output_labels, obj_dists_2

    def get_union_box(self, rois, union_inds):
        im_inds = rois[:, 0][union_inds[:, 0]]

        union_rois = torch.cat((
            im_inds[:, None],
            torch.min(rois[:, 1:3][union_inds[:, 0]], rois[:, 1:3][union_inds[:, 1]]),
            torch.max(rois[:, 3:5][union_inds[:, 0]], rois[:, 3:5][union_inds[:, 1]]),
        ), 1)
        return union_rois

    def max_pooling_image(self, obj_dist, num_box):
        output = []
        pre_i = 0
        for i in num_box.data.cpu().numpy():
            i = int(i)
            output.append(((obj_dist[pre_i:pre_i + i].max(0)[0])).clone())
            pre_i = i
        return torch.stack(output)

    def forward(self, obj_fmaps, obj_logits, im_inds, obj_labels=None,
                box_priors=None, boxes_per_cls=None, obj_preds=None):
        """
        Forward pass through the object and edge context
        :param obj_fmaps: shape: [num_boxes, dim_feature]
        :param obj_logits: shape: [num_boxes, num_classes]  before softmax
        :param im_inds: shape: [num_boxes, 1]  each is img_ind
        :param obj_labels: shape: [num_boxes, 1]  each is box class
        :param box_priors: shape: [num_boxes, 4]  each is box position
        :return:
        """
        obj_logits_softmax = F.softmax(obj_logits, dim=1)
        # if self.mode == 'predcls':
        #   obj_logits = Variable(to_onehot(obj_labels.data, self.num_classes))
        #   obj_logits_softmax = obj_logits

        obj_embed = obj_logits_softmax @ self.obj_embed.weight
        obj_embed = F.dropout(obj_embed, self.dropout_rate, training=self.training)
        pos_embed = self.pos_embed(Variable(center_size(box_priors)))

        obj_pre_rep = torch.cat((obj_fmaps, obj_embed, pos_embed), 1)

        if self.nl_obj > 0:
            if self.training or self.mode == 'predcls':
                obj_ctx_preds_in = obj_labels
            elif self.mode == 'sgcls':
                obj_ctx_preds_in = obj_logits[:, 1:].max(1)[1] + 1
            else:
                obj_ctx_preds_in = obj_preds
            if self.mode == 'precls':
                obj_gcn_dis = Variable(to_onehot(obj_ctx_preds_in.data, self.num_classes))
            else:
                obj_gcn_dis = obj_logits
            obj_ctx, obj_dists2, obj_preds, obj_dists_pred = self.obj_ctx(
                obj_feats=obj_pre_rep,
                obj_dists=obj_gcn_dis,
                obj_labels=obj_labels,
                boxes_per_cls=boxes_per_cls,
                im_inds=im_inds,
                obj_preds=obj_ctx_preds_in,
            )
        else:
            # UNSURE WHAT TO DO HERE
            if self.mode == 'predcls':
                obj_dists2 = Variable(to_onehot(obj_labels.data, self.num_classes))
            else:
                obj_dists2 = obj_logits

            if self.mode == 'sgdet' and not self.training:
                # NMS here for baseline
                probs = F.softmax(obj_dists2, 1)
                nms_mask = obj_dists2.data.clone()
                nms_mask.zero_()
                for c_i in range(1, obj_dists2.size(1)):
                    scores_ci = probs.data[:, c_i]
                    boxes_ci = boxes_per_cls.data[:, c_i]

                    keep = apply_nms(scores_ci, boxes_ci,
                                     pre_nms_topn=scores_ci.size(0), post_nms_topn=scores_ci.size(0),
                                     nms_thresh=0.3)
                    nms_mask[:, c_i][keep] = 1

                obj_preds = Variable(nms_mask * probs.data, volatile=True)[:, 1:].max(1)[1] + 1
            else:
                obj_preds = obj_labels if obj_labels is not None else obj_dists2[:, 1:].max(1)[1] + 1
            obj_ctx = obj_pre_rep

        if self.nl_edge > 0:
            # hard embedding
            obj_embed_to_edge = self.obj_embed_in_edge(obj_preds)
            edge_pre_rep = torch.cat((obj_fmaps, obj_embed_to_edge, pos_embed), 1)
            # edge_pre_rep = torch.cat((obj_ctx, obj_embed_to_edge, pos_embed), 1)
            edge_ctx = self.edge_ctx(
                obj_feats=edge_pre_rep,
                obj_preds=obj_preds,  # obj_preds_zeros, #obj_preds_zeros obj_preds_nozeros
                im_inds=im_inds,
            )
        if self.training or self.mode == 'predcls':
            edge_obj_preds_in = obj_labels
        else:
            edge_obj_preds_in = obj_preds
        if self.mode == 'precls':
            edge_gcn_dis = Variable(to_onehot(obj_preds.data, self.num_classes))  # obj_dists_pred
            # edge_gcn_dis = obj_dists_pred
        else:
            edge_gcn_dis = obj_dists2
            #
        if self.nl_gcn_edge > 0:
            # edge_gcn_output = self.edge_gcn(
            # obj_preds = edge_obj_preds_in,
            # obj_dists = F.softmax(edge_gcn_dis, dim=1),
            # )
            edge_gcn_output = self.exchange_obj_edge_gcn_func(
                obj_preds=edge_obj_preds_in,
                obj_dists=F.softmax(edge_gcn_dis, dim=1),
            )
            edge_ctx = edge_gcn_output + edge_ctx

        if self.nl_gcn_concept > 0:
            # edge_gcn_output = self.edge_gcn(
            # obj_preds = edge_obj_preds_in,
            # obj_dists = F.softmax(edge_gcn_dis, dim=1),
            # )
            concept_gcn_output = self.concept_gcn_func(
                obj_preds=edge_obj_preds_in,
                im_inds=im_inds,
            )
            edge_ctx = concept_gcn_output + edge_ctx

        return obj_dists2, obj_preds, edge_ctx


class RelModel(nn.Module):
    """
    RELATIONSHIPS
    """

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, use_vision=True, require_overlap_det=True,
                 embed_dim=200, hidden_dim=256, pooling_dim=2048,
                 nl_obj=6, nl_edge=6, nh_obj=6, nh_edge=6,
                 nl_obj_gcn=2, nl_edge_gcn=2,
                 use_resnet=False, thresh=0.01,
                 use_proposals=False, pass_in_obj_feats_to_decoder=True,
                 pass_in_obj_feats_to_edge=True, rec_dropout=0.0, use_bias=True, use_tanh=True,
                 limit_vision=True, bg_num_rel=2, test_alpha=1.0):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param use_vision: Whether to use vision in the final product
        :param require_overlap_det: Whether two objects must intersect
        :param embed_dim: Dimension for all embeddings
        :param hidden_dim: LSTM hidden size
        :param obj_dim:
        """
        print("-----------Init ablation rel model exgan extgraph 0!-----------")
        super(RelModel, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode

        self.pooling_size = 7
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 2048 if use_resnet else 4096
        self.pooling_dim = pooling_dim

        self.use_bias = use_bias
        self.use_vision = use_vision
        self.use_tanh = use_tanh
        self.limit_vision = limit_vision
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'
        self.use_resnet = use_resnet
        self.nl_obj = nl_obj
        self.nh_obj = nh_obj
        self.nl_obj_gcn = nl_obj_gcn
        self.nl_edge = nl_edge
        self.nh_edge = nh_edge
        self.nl_edge_gcn = nl_edge_gcn
        self.bg_num_rel = bg_num_rel
        self.test_alpha = test_alpha
        self.dropout_rate = rec_dropout
        self.with_graph = True

        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64,
            bg_num_rel=self.bg_num_rel
        )

        self.context = LinearizedContext(self.classes, self.rel_classes, mode=self.mode,
                                         embed_dim=self.embed_dim, hidden_dim=self.hidden_dim,
                                         obj_dim=self.obj_dim,
                                         nl_obj=nl_obj, nl_edge=nl_edge, nh_obj=self.nh_obj, nh_edge=self.nh_edge,
                                         nl_gcn_obj=self.nl_obj_gcn, nl_gcn_edge=self.nl_edge_gcn,
                                         dropout_rate=rec_dropout,
                                         pass_in_obj_feats_to_decoder=pass_in_obj_feats_to_decoder,
                                         pass_in_obj_feats_to_edge=pass_in_obj_feats_to_edge)

        # Image Feats (You'll have to disable if you want to turn off the features from here)
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                              dim=1024 if use_resnet else 512)

        if use_resnet:
            roi_fmap = load_resnet(pretrained=False)[1]
            if pooling_dim != 2048:
                roi_fmap.append(nn.Linear(2048, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_resnet(pretrained=False)[1]
        else:
            roi_fmap = [
                Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096,
                         pretrained=False).classifier,
            ]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier

        ###################################
        post_context_dim = self.hidden_dim  # * 2
        self.post_context_obj = nn.Linear(post_context_dim, self.pooling_dim)
        self.post_context_sub = nn.Linear(post_context_dim, self.pooling_dim)

        # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1.
        # (Half contribution comes from LSTM, half from embedding.

        # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
        self.post_context_obj.weight = torch.nn.init.xavier_normal(self.post_context_obj.weight, gain=1.0)
        self.post_context_sub.weight = torch.nn.init.xavier_normal(self.post_context_sub.weight, gain=1.0)

        rel_bilinear_input_dim = self.pooling_dim
        self.rel_bilinear_layer = nn.Linear(rel_bilinear_input_dim, self.num_rels, bias=False)
        self.rel_bilinear_layer.weight = torch.nn.init.xavier_normal(self.rel_bilinear_layer.weight, gain=1.0)
        if self.with_graph:
            self.rel_bilinear_graph_layer = nn.Linear(rel_bilinear_input_dim, 2, bias=False)
            self.rel_bilinear_graph_layer.weight = torch.nn.init.xavier_normal(self.rel_bilinear_graph_layer.weight,
                                                                               gain=1.0)
        if self.use_bias:
            self.freq_bias = FrequencyBias()
            if self.with_graph:
                self.freq_bias_graph = FrequencyBias(graph=True)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def visual_rep(self, features, rois, pair_inds):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        if not self.use_resnet:
            return self.roi_fmap(uboxes)
        else:
            # print('uboxes: ',uboxes.size())
            roi_fmap_t = self.roi_fmap(uboxes)
            # print('roi_fmap_t: ',roi_fmap_t.size())
            return roi_fmap_t.mean(3).mean(2)

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        # Get the relationship candidates
        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)

                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        if not self.use_resnet:
            feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
                features, rois)
            return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))
        else:
            feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
                features, rois)
            return self.roi_fmap_obj(feature_pool).mean(3).mean(2)

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels

            if test:
            prob dists, boxes, img inds, maxscores, classes

        """
        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)
        # if not self.training:
        #     if self.mode == 'sgdet' or self.mode == 'sgcls':
        #         result.rm_obj_labels = None
        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels, gt_adj_mat_graph, rel_labels_offset_fg \
                = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                  gt_boxes.data, gt_classes.data, gt_rels.data,
                                  image_offset, filter_non_overlap=True,
                                  num_sample_per_gt=1,
                                  time_bg=self.bg_num_rel, time_fg=1)

        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)

        rois = torch.cat((im_inds[:, None].float(), boxes), 1)

        result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)
        if self.training or self.mode == 'predcls':
            obj_gt_labels = result.rm_obj_labels
        else:
            obj_gt_labels = None
        # Prevent gradients from flowing back into score_fc from elsewhere
        vr_rel = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:])
        result.rm_obj_dists, result.obj_preds_nozeros, edge_ctx = self.context(
            obj_fmaps=result.obj_fmap,
            obj_logits=result.rm_obj_dists.detach(),
            im_inds=im_inds,
            obj_labels=obj_gt_labels,
            box_priors=boxes.data,
            boxes_per_cls=result.boxes_all,
            obj_preds=result.obj_preds,
        )

        subj_rep = self.post_context_sub(edge_ctx)
        obj_rep = self.post_context_obj(edge_ctx)

        subj_rep = F.dropout(subj_rep, self.dropout_rate, training=self.training)
        obj_rep = F.dropout(obj_rep, self.dropout_rate, training=self.training)

        vr_obj = vr_rel
        subj_rep_rel = subj_rep[rel_inds[:, 1]]
        obj_rep_rel = obj_rep[rel_inds[:, 2]]
        # Split into subject and object representations
        if self.use_vision:

            if self.limit_vision:
                # exact value TBD
                subj_rep_rel = torch.cat((subj_rep_rel[:, :2048] * subj_rep_rel[:, :2048], subj_rep_rel[:, 2048:]), 1)
                obj_rep_rel = torch.cat((obj_rep_rel[:, :2048] * obj_rep_rel[:, :2048], obj_rep_rel[:, 2048:]), 1)
            else:
                subj_rep_rel = subj_rep_rel * vr_obj
                obj_rep_rel = obj_rep_rel * vr_obj

        if self.use_tanh:
            subj_rep_rel = F.tanh(subj_rep_rel)
            obj_rep_rel = F.tanh(obj_rep_rel)

        prod_rep_graph = subj_rep_rel * obj_rep_rel
        result.rel_dists = self.rel_bilinear_layer(prod_rep_graph)
        if self.with_graph:
            result.rel_dists_graph = self.rel_bilinear_graph_layer(prod_rep_graph)

        if self.use_bias:
            rel_obj_preds = result.obj_preds_nozeros.clone()
            freq_bias_so = self.freq_bias.index_with_labels(torch.stack((
                rel_obj_preds[rel_inds[:, 1]],
                rel_obj_preds[rel_inds[:, 2]],
            ), 1))
            # freq_bias_so = F.dropout(freq_bias_so, self.dropout_rate, training=self.training)
            freq_bias_so = F.dropout(freq_bias_so, 0.3, training=self.training)
            result.rel_dists = result.rel_dists + freq_bias_so
            if self.with_graph:
                freq_bias_so_graph = self.freq_bias_graph.index_with_labels(torch.stack((
                    rel_obj_preds[rel_inds[:, 1]],
                    rel_obj_preds[rel_inds[:, 2]],
                ), 1))
                # freq_bias_so_graph = F.dropout(freq_bias_so_graph, self.dropout_rate, training=self.training)
                freq_bias_so_graph = F.dropout(freq_bias_so_graph, 0.3, training=self.training)
                result.rel_dists_graph = result.rel_dists_graph + freq_bias_so_graph
        if self.training:
            return result

        twod_inds = arange(result.obj_preds_nozeros.data) * self.num_classes + result.obj_preds_nozeros.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors

        rel_scores = F.softmax(result.rel_dists, dim=1)
        if self.with_graph:
            rel_scores_graph = F.softmax(result.rel_dists_graph, dim=1)
            rel_scores = rel_scores * (rel_scores_graph[:, 1])[:, None]
            '''
            rel_scores1 = rel_scores[:, 1:] * ((rel_scores_graph[:, 1])[:, None]**alpha)
            rel_scores0 = rel_scores[:, 0] * (rel_scores_graph[:, 0]**(1.0-alpha))
            rel_scores =torch.cat([rel_scores0[:,None],rel_scores1],-1)
            '''            
        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds_nozeros, rel_inds[:, 1:], rel_scores)

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs
