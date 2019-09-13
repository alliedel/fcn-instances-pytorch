#!/usr/bin/env python

import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import instanceseg
import instanceseg.analysis.visualization_utils
import instanceseg.datasets.synthetic
import instanceseg.datasets.voc
import instanceseg.losses.loss
import instanceseg.utils.logs
from instanceseg.utils import instance_utils
from instanceseg.analysis import visualization_utils

here = osp.dirname(osp.abspath(__file__))


def compute_scores(img, sem_lbl, inst_lbl, problem_config, scoring_method='instance', smearing=0,
                   assignment_mixing=0, max_confidence=10000, cuda=True):
    assert sem_lbl.shape[0] == 1, NotImplementedError('Only handling the case of one image for now')
    n_instance_classes = problem_config.n_classes
    correct_semantic_score = semantic_label_gt_as_instance_prediction(sem_lbl, problem_config)

    if scoring_method == 'semantic':
        score = correct_semantic_score
    elif scoring_method == 'instance':
        correct_instance_score = semantic_instance_label_gts_as_instance_prediction(sem_lbl, inst_lbl, problem_config)
        correct_instance_score_swapped = swap_channels(correct_instance_score, channel0=1, channel1=3)
        score = (1.0 * correct_instance_score + smearing * correct_instance_score_swapped) / (1.0 + smearing)
    else:
        raise ValueError("Didn't recognize the scoring method {}".format(scoring_method))
    if cuda:
        score = score.cuda()
    return score * max_confidence


def swap_channels(tensor_4d, channel0, channel1):
    """
    Requires making a copy since we can't do in-place operations
    """
    permuted_channels = list(range(tensor_4d.size(1)))
    permuted_channels[channel0] = channel1
    permuted_channels[channel1] = channel0
    return permute_channels(tensor_4d, permuted_channels)


def permute_channels(tensor_4d, permuted_channels):
    """
    Requires making a copy since we can't do in-place operations
    """
    return torch.cat([tensor_4d[:, i:(i + 1), :, :] for i in permuted_channels], dim=1)


def semantic_label_gt_as_instance_prediction(sem_lbl, problem_config):
    n_instance_classes = problem_config.n_classes
    model_channel_semantic_ids = problem_config.model_channel_semantic_ids
    # compute the semantic label score
    score_shape = (sem_lbl.size(0), n_instance_classes, sem_lbl.size(1), sem_lbl.size(2))
    sem_lbl_score = Variable(torch.zeros(score_shape))
    for inst_idx, sem_val in enumerate(model_channel_semantic_ids):
        sem_lbl_score[:, inst_idx, ...] = sem_lbl == sem_val
    return sem_lbl_score


def semantic_instance_label_gts_as_instance_prediction(sem_lbl, inst_lbl, problem_config):
    n_instance_classes = problem_config.n_classes
    instance_ids = problem_config.instance_count_id_list
    # compute the semantic label score
    score_shape = (sem_lbl.size(0), n_instance_classes, sem_lbl.size(1), sem_lbl.size(2))
    inst_score = Variable(torch.zeros(score_shape))
    for inst_idx, (sem_val, inst_id) in enumerate(zip(problem_config.model_channel_semantic_ids, instance_ids)):
        inst_score[:, inst_idx, ...] = (sem_lbl == sem_val) & (inst_lbl == inst_id)
    return inst_score


def loss_function(score, sem_lbl, inst_lbl, instance_problem, matching_loss=True, size_average=True,
                  return_loss_components=False, **kwargs):
    if not (sem_lbl.size() == inst_lbl.size() == (score.size(0), score.size(2),
                                                  score.size(3))):
        import ipdb;
        ipdb.set_trace()
        raise Exception('Sizes of score, targets are incorrect')
    rets = instanceseg.losses.loss.cross_entropy2d(
        score, sem_lbl, inst_lbl,
        semantic_instance_labels=instance_problem.model_channel_semantic_ids,
        instance_id_labels=instance_problem.instance_count_id_list,
        matching=matching_loss, size_average=size_average, break_here=False, recompute_optimal_loss=False,
        return_loss_components=return_loss_components, **kwargs)
    if return_loss_components:
        permutations, loss, loss_components = rets
        return permutations, loss, loss_components
    else:
        permutations, loss = rets
        return permutations, loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--resume', help='Checkpoint path')
    parser.add_argument('--scoring_method', help='scoring method', default='instance')
    parser.add_argument('--smearing', type=float, default=0)
    parser.add_argument('--assignment_mixing', type=int, default=0)
    parser.add_argument('--max_confidence', type=float, default=1)
    parser.add_argument('--xaxis', default='max_confidence')
    args = parser.parse_args()
    return args


def write_visualizations(sem_lbl, inst_lbl, score, pred_permutations, problem_config, outdir, writer, iteration,
                         basename):
    inst_lbl_pred = score.max(dim=1)[1].data.cpu().numpy()[:, :, :]
    sem_lbl = sem_lbl.data.cpu().numpy()
    inst_lbl = inst_lbl.data.cpu().numpy()
    score = score.data.cpu().numpy()
    model_channel_semantic_ids = problem_config.model_channel_semantic_ids
    instance_count_id_list = problem_config.instance_count_id_list
    n_combined_class = problem_config.n_classes

    lt_combined = instance_utils.combine_semantic_and_instance_labels(sem_lbl, inst_lbl, model_channel_semantic_ids,
                                                                      instance_count_id_list)
    channel_labels = problem_config.get_channel_labels('{} {}')
    viz = visualization_utils.visualize_heatmaps(scores=score[0, ...],
                                                 lbl_true=lt_combined[0, ...],
                                                 lbl_pred=inst_lbl_pred[0, ...],
                                                 pred_permutations=pred_permutations[0, ...],
                                                 n_class=n_combined_class,
                                                 score_vis_normalizer=score.max(),
                                                 channel_labels=channel_labels,
                                                 channels_to_visualize=None)
    instanceseg.analysis.visualization_utils.export_visualizations([viz], out_dir=outdir, tensorboard_writer=writer,
                                                                   iteration=iteration, basename=basename, tile=True)


def main():
    synthetic_generator_n_instances_per_semantic_id = 2
    args = parse_args()
    cfg = {
        'max_confidence': args.max_confidence,
        'scoring_method': args.scoring_method,
        'smearing': args.smearing,
        'assignment_mixing': args.assignment_mixing,
        'xaxis': args.xaxis,
    }

    out = instanceseg.utils.logs.get_log_dir(osp.basename(__file__).replace('.py', ''), config_id=None,
                                             cfg=cfg,
                                             parent_directory=os.path.join(here, 'logs', 'synthetic'))
    print('Log in {}'.format(out))
    cuda = True  # torch.cuda.is_available()
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) if isinstance(gpu, int) else ','.join(str(gpu))

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    dataset_kwargs = dict(transform=True, n_max_per_class=synthetic_generator_n_instances_per_semantic_id,
                          map_to_single_instance_problem=False, blob_size=(80, 80))
    train_dataset = instanceseg.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs)
    val_dataset = instanceseg.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs)
    try:
        img, (sl, il) = train_dataset[0]
    except:
        import ipdb;
        ipdb.set_trace()
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
    labels_table = instance_utils.create_default_labels_table_from_instance_problem_config(
        semantic_class_names=class_names, semantic_class_vals=list(range(len(class_names))))
    problem_config = instance_utils.InstanceProblemConfig(n_instances_by_semantic_id=n_instances_by_semantic_id,
                                                          labels_table=labels_table)

    writer = SummaryWriter(log_dir=out)

    for data_idx, (data, (sem_lbl, inst_lbl)) in enumerate(train_loader):
        if data_idx >= 1:
            break
        if cuda:
            data, (sem_lbl, inst_lbl) = data.cuda(), (sem_lbl.cuda(), inst_lbl.cuda())
        data, sem_lbl, inst_lbl = Variable(data, volatile=True), \
                                  Variable(sem_lbl), Variable(inst_lbl)
        if args.xaxis == 'max_confidence':
            max_confidences = [1, 2, 3, 4, 5, 10, 100, 1000, 10000]
            smearings = [args.smearing for _ in max_confidences]
            assignment_mixings = [args.assignment_mixing for _ in max_confidences]
        elif args.xaxis == 'smearing':
            smearings = list(np.linspace(0, 1, 10))
            max_confidences = [args.max_confidence for _ in smearings]
            assignment_mixings = [args.assignment_mixing for _ in smearings]
        else:
            raise ValueError('xaxis == {} unrecognized.'.format(args.xaxis))
        for prediction_number, (max_confidence, smearing, assignment_mixing) in enumerate(zip(max_confidences,
                                                                                              smearings,
                                                                                              assignment_mixings)):
            scoring_cfg = {
                'max_confidence': max_confidence,
                'scoring_method': args.scoring_method,
                'smearing': float(smearing),
                'assignment_mixing': assignment_mixing,
            }
            print('prediction {}/{}'.format(prediction_number + 1, len(max_confidences)))
            score = compute_scores(data, sem_lbl, inst_lbl, problem_config, cuda=True, **scoring_cfg)
            assignments, loss, loss_components = loss_function(score, sem_lbl, inst_lbl, problem_config,
                                                                     return_loss_components=True)
            if np.isnan(float(loss.data[0])):
                raise ValueError('losses is nan while validating')
            softmax_scores = F.softmax(score, dim=1)
            inst_lbl_pred = score.data.max(dim=1)[1].cpu().numpy()[:, :, :]

            # Write scalars
            writer.add_scalar('max_confidence', max_confidence, prediction_number)
            writer.add_scalar('smearing', smearing, prediction_number)
            # writer.add_scalar('instance_mixing', instance_mixing, prediction_number)
            writer.add_scalar('losses', loss, prediction_number)
            channel_labels = problem_config.get_channel_labels('{} {}')
            for i, label in enumerate(channel_labels):
                tag = 'loss_components/{}'.format(label.replace(' ', '_'))
                writer.add_scalar(tag, loss_components[i], prediction_number)

            # Write images
            write_visualizations(sem_lbl, inst_lbl, softmax_scores, pred_permutations, problem_config, outdir=out,
                                 writer=writer, iteration=prediction_number, basename='scores')


if __name__ == '__main__':
    main()
