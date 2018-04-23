#!/usr/bin/env python

import argparse
import numpy as np
import os
import os.path as osp
import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import torchfcn
import torchfcn.datasets.voc
import torchfcn.datasets.synthetic
from torchfcn import script_utils, instance_utils, visualization_utils, trainer
import skimage.io
from torchfcn import losses


here = osp.dirname(osp.abspath(__file__))


def loss_function(score, sem_lbl, inst_lbl, instance_problem, matching_loss=True, size_average=True, **kwargs):
    if not (sem_lbl.size() == inst_lbl.size() == (score.size(0), score.size(2),
                                                  score.size(3))):
        import ipdb; ipdb.set_trace()
        raise Exception('Sizes of score, targets are incorrect')
    permutations, loss = losses.cross_entropy2d(
        score, sem_lbl, inst_lbl,
        semantic_instance_labels=instance_problem.semantic_instance_class_list,
        instance_id_labels=instance_problem.instance_count_id_list,
        matching=matching_loss, size_average=size_average, break_here=False, recompute_optimal_loss=False, **kwargs)
    return permutations, loss


def semantic_label_gt_as_instance_prediction(sem_lbl, problem_config):
    n_instance_classes = problem_config.n_classes
    semantic_instance_class_list = problem_config.semantic_instance_class_list
    # compute the semantic label score
    score_shape = (sem_lbl.size(0), n_instance_classes, sem_lbl.size(1), sem_lbl.size(2))
    sem_lbl_score = Variable(torch.zeros(score_shape))
    for inst_idx, sem_val in enumerate(semantic_instance_class_list):
        sem_lbl_score[:, inst_idx, ...] = sem_lbl == sem_val
    return sem_lbl_score


def semantic_instance_label_gts_as_instance_prediction(sem_lbl, inst_lbl, problem_config):
    n_instance_classes = problem_config.n_classes
    instance_ids = problem_config.instance_count_id_list
    semantic_instance_class_list = problem_config.semantic_instance_class_list
    # compute the semantic label score
    score_shape = (sem_lbl.size(0), n_instance_classes, sem_lbl.size(1), sem_lbl.size(2))
    inst_score = Variable(torch.zeros(score_shape))
    for inst_idx, sem_val, inst_id in enumerate(zip(semantic_instance_class_list, instance_ids)):
        inst_score[:, inst_idx, ...] = (sem_lbl == sem_val) & (inst_lbl == inst_id)
    return inst_score


def compute_scores(img, sem_lbl, inst_lbl, problem_config, quality=1.0):
    assert sem_lbl.shape[0] == 1, NotImplementedError('Only handling the case of one image for now')
    n_instance_classes = problem_config.n_classes
    perfect_semantic_score = semantic_label_gt_as_instance_prediction(sem_lbl, problem_config)
    correct_instance_score = semantic_instance_label_gts_as_instance_prediction(sem_lbl, inst_lbl, problem_config)
    score = correct_instance_score
    return score


def main():
    script_utils.check_clean_work_tree()

    synthetic_generator_n_instances_per_semantic_id = 2

    out = script_utils.get_log_dir(osp.basename(__file__).replace(
        '.py', ''), parent_directory=osp.dirname(osp.abspath(__file__)))

    cuda = False  # torch.cuda.is_available()
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    dataset_kwargs = dict(transform=True, n_max_per_class=synthetic_generator_n_instances_per_semantic_id,
                          map_to_single_instance_problem=False)
    train_dataset = torchfcn.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs)
    val_dataset = torchfcn.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs)
    try:
        img, (sl, il) = train_dataset[0]
    except:
        import ipdb; ipdb.set_trace()
        raise Exception('Cannot load an image from your dataset')
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    train_loader_for_val = torch.utils.data.DataLoader(train_dataset.copy(modified_length=3), batch_size=1,
                                                       shuffle=False, **loader_kwargs)

    # 0. Problem setup (instance segmentation definition)
    class_names = val_dataset.class_names
    n_semantic_classes = len(class_names)
    cfg = {'n_instances_per_class': 3}
    n_instances_per_class = cfg['n_instances_per_class'] or \
                            (1 if cfg['single_instance'] else synthetic_generator_n_instances_per_semantic_id)
    n_instances_by_semantic_id = [1] + [n_instances_per_class for sem_cls in range(1, n_semantic_classes)]
    problem_config = instance_utils.InstanceProblemConfig(n_instances_by_semantic_id=n_instances_by_semantic_id)
    problem_config.set_class_names(class_names)

    writer = SummaryWriter(log_dir=out)

    for data_idx, (data, (sem_lbl, inst_lbl)) in enumerate(train_loader):
        if data_idx >= 1:
            break
        if cuda:
            data, (sem_lbl, inst_lbl) = data.cuda(), (sem_lbl.cuda(), inst_lbl.cuda())
        data, sem_lbl, inst_lbl = Variable(data, volatile=True), \
                                  Variable(sem_lbl), Variable(inst_lbl)
        for prediction_number, prediction_quality in enumerate(np.linspace(0, 1, 1000)):
            semantic_quality = 1.0
            instance_confidence = 1.0
            instance_mixing = 0.0
            score = compute_scores(data, sem_lbl, inst_lbl, problem_config, prediction_quality)
            pred_permutations, loss = loss_function(score, sem_lbl, inst_lbl, problem_config)
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while validating')
            softmax_scores = F.softmax(score, dim=1).data.cpu().numpy()
            inst_lbl_pred = score.data.max(dim=1)[1].cpu().numpy()[:, :, :]
            writer.add_scalar('instance_confidence', instance_confidence, prediction_number)
            writer.add_scalar('instance_mixing', instance_mixing, prediction_number)
            writer.add_scalar('semantic_quality', semantic_quality, prediction_number)


if __name__ == '__main__':
    main()
