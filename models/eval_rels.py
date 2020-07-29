
from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
import torch

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
import dill as pkl
import os
import random
conf = ModelConfig()
if conf.model == 'motifnet':
    from lib.rel_model import RelModel
elif conf.model == 'gkg':
    from lib.models.rel_model_mspk_os import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
else:
    raise ValueError()
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')
if conf.test:
    val = test
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet,
                    nl_edge=conf.nl_edge, nl_obj=conf.nl_obj,
                    nh_edge=conf.nh_edge, nh_obj=conf.nh_obj,
                    nl_edge_gcn=conf.nl_edge_gcn, nl_obj_gcn=conf.nl_obj_gcn,
                    hidden_dim=conf.hidden_dim,
                    use_proposals=conf.use_proposals,
                    pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
                    pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
                    pooling_dim=conf.pooling_dim,
                    rec_dropout=conf.rec_dropout,
                    use_bias=conf.use_bias,
                    use_tanh=conf.use_tanh,
                    limit_vision=conf.limit_vision,
                    bg_num_rel=conf.bg_num_rel,
                    test_alpha=conf.test_alpha,
                    )


detector.cuda()
ckpt = torch.load(conf.ckpt)
print('Loading Everything')
optimistic_restore(detector, ckpt['state_dict'])


all_pred_entries = []
def val_batch(batch_num, b, evaluator, thrs=(20, 50, 100)):
    num_obj_cls_correct = 0
    num_obj_sample = 0
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:,0]] > 0) and np.all(objs_i[rels_i[:,1]] > 0)
        # assert np.all(rels_i[:,2] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,
        }
        all_pred_entries.append(pred_entry)

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )
        if conf.mode != 'sgdet':
            num_obj_cls_correct = num_obj_cls_correct + np.sum((val.gt_classes[batch_num + i].copy() - objs_i) == 0)
            num_obj_sample = num_obj_sample + objs_i.shape[0]
    return  num_obj_cls_correct, num_obj_sample

evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=conf.multi_pred)
if conf.cache is not None and os.path.exists(conf.cache):
    print("Found {}! Loading from it".format(conf.cache))
    with open(conf.cache,'rb') as f:
        all_pred_entries = pkl.load(f)
    for i, pred_entry in enumerate(tqdm(all_pred_entries)):
        gt_entry = {
            'gt_classes': val.gt_classes[i].copy(),
            'gt_relations': val.relationships[i].copy(),
            'gt_boxes': val.gt_boxes[i].copy(),
        }
        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )
    evaluator[conf.mode].print_stats()
else:
    detector.eval()
    num_correct = 0
    num_sample = 0
    for val_b, batch in enumerate(tqdm(val_loader)):
        num_correct_i, num_sample_i \
            = val_batch(conf.num_gpus*val_b, batch, evaluator)
        num_correct = num_correct + num_correct_i
        num_sample = num_sample + num_sample_i
    evaluator[conf.mode].print_stats()
    print('num_correct ',num_correct)
    print('num_sample',num_sample)
    print('obj acc:', (num_correct*1.0)/(num_sample*1.0 + 1e-8))
    if conf.cache is not None:
        with open(conf.cache,'wb') as f:
            pkl.dump(all_pred_entries, f)
