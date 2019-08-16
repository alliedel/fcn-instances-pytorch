import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import tqdm
import yaml
from PIL import Image

from instanceseg.ext.panopticapi.utils import rgb2id
from instanceseg.losses import loss
from instanceseg.utils import display as display_pyutils
from instanceseg.utils import instance_utils
from instanceseg.utils.misc import rgb2hex, TermColors
from scripts import evaluate
from instanceseg.utils.script_setup import get_cache_dir_from_test_logdir
from scripts.visualizations import visualize_pq_stats, export_prediction_vs_gt_vis_sorted


def load_gt_img_in_panoptic_form(gt_img_file):
    with Image.open(gt_img_file) as img:
        pan_gt = np.array(img, dtype=np.uint32)
    pan_gt = rgb2id(pan_gt)
    return pan_gt


def scatter_loss_vs_pq(loss_npz_file, pq_npz_file, save_to_workspace=True, reverse_x=True):
    display_pyutils.set_my_rc_defaults()

    output_dir = os.path.dirname(loss_npz_file)
    loss_npz_d = np.load(loss_npz_file)
    pq_npz_d = np.load(pq_npz_file)
    eval_iou_threshold = pq_npz_d['iou_threshold'].item()
    loss_arr = loss_npz_d['losses_compiled_per_img_per_cls']
    eval_stat_types = ('pq', 'sq', 'rq')
    pq_arrs = {
        stat_type: pq_npz_d['collated_stats_per_image_per_cat'].item()[stat_type] for stat_type in eval_stat_types
    }
    problem_config = loss_npz_d['problem_config'].item()
    assert problem_config.semantic_instance_class_list == pq_npz_d[
        'problem_config'].item().semantic_instance_class_list, \
        'Sanity check failed: Problem configurations should have matched up'
    assert problem_config.n_semantic_classes == loss_arr.shape[1]
    subplot_idx = 1
    sem_names = problem_config.semantic_class_names
    sem_idxs_to_visualize = [i for i, nm in enumerate(sem_names) if nm != 'background']
    sem_totals_to_visualize = ['total_with_bground', 'total_without_bground']
    subR, subC = len(eval_stat_types), len(sem_idxs_to_visualize) + len(sem_totals_to_visualize)
    scale = max(np.ceil(subR / 2), np.ceil(subC / 4))
    plt.figure(figsize=[scale * s for s in display_pyutils.BIG_FIGSIZE]);
    plt.clf()

    markers = display_pyutils.MARKERS
    default_colors = display_pyutils.GOOD_COLOR_CYCLE
    try:
        labels_table = problem_config.labels_table
        colors_by_idx = {idx: np.array(labels_table[idx].color) for idx in range(len(labels_table))}
    except:
        raise
    size = 30
    for eval_stat_idx, eval_stat_type in enumerate(eval_stat_types):
        x_arr = pq_arrs[eval_stat_type]
        y_arr = loss_arr
        assert x_arr.shape == y_arr.shape
        eval_identifier = eval_stat_type + '-iou_{}'.format(eval_iou_threshold)
        for sem_idx, sem_name in zip(sem_idxs_to_visualize, [sem_names[ii] for ii in sem_idxs_to_visualize]):
            # pq_d[xlabel]
            x, y = x_arr[:, sem_idx], y_arr[:, sem_idx]
            plt.subplot(subR, subC, subplot_idx)
            ylabel = 'loss'
            xlabel = eval_identifier + (' (reversed)' if reverse_x else '')
            label = 'loss vs {}, component: {}'.format(eval_stat_type, sem_name)
            scatter(x, y, colors_by_idx[sem_idx], label, markers, size, xlabel, ylabel)
            subplot_idx += 1
            plt.xlim([0, 1])
            if reverse_x:
                plt.gca().invert_xaxis()

        for agg_i, aggregate_type in enumerate(sem_totals_to_visualize):
            if aggregate_type == 'total_with_bground':
                sem_idxs = list(range(len(sem_names)))
            elif aggregate_type == 'total_without_bground':
                sem_idxs = [idx for idx, nm in enumerate(sem_names) if nm != 'background']
            else:
                raise NotImplementedError
            plt.subplot(subR, subC, subplot_idx)
            ylabel = 'total loss'
            xlabel = 'sum ' + eval_identifier + (' (reversed)' if reverse_x else '')
            label = 'loss vs {}, {}'.format(eval_stat_type, aggregate_type)
            x, y = x_arr[:, sem_idxs].sum(axis=1), y_arr[:, sem_idxs].sum(axis=1)
            scatter(x, y, default_colors[agg_i], label, markers, size, xlabel, ylabel)
            subplot_idx += 1
            plt.xlim([0, max(x)])
            if reverse_x:
                plt.gca().invert_xaxis()

    plt.tight_layout()
    onehot_tag = '_onehot' if 'losses_onehot' in loss_npz_file else ''

    figname = 'all_loss_vs_eval_iouthresh_{}{}.png'.format(eval_iou_threshold, onehot_tag)
    if save_to_workspace:
        display_pyutils.save_fig_to_workspace(figname)
    figpath = os.path.join(output_dir, figname)
    plt.savefig(figpath)
    print(TermColors.OKGREEN + 'Scatterplot saved to {}'.format(os.path.abspath(figpath)) + TermColors.ENDC)


def scatter(x, y, color, label, markers, size, xlabel, ylabel):
    if len(color) == 3:
        color = rgb2hex(color[0], color[1], color[2])
    plt.scatter(x, y, alpha=0.5, marker=markers[0], s=size,
                c=color, edgecolors=color, label=label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(label)


def main(test_logdir, overwrite=False, eval_iou_threshold=None, which_models='best',
         visualize_pq_hists=True, export_sorted_perf_images=None, scores_to_onehot=False):
    # with open(os.path.join(args.test_logdir, 'train_logdir.txt'), 'r') as fid:
    #     train_logdir = fid.read()
    # model_paths = get_list_of_model_paths(train_logdir, which_models)
    export_sorted_perf_images = export_sorted_perf_images if export_sorted_perf_images is not None else \
        visualize_pq_hists

    loss_npz_file = compute_losses(test_logdir, overwrite, scores_to_onehot=scores_to_onehot)
    analysis_cache_outdir = get_cache_dir_from_test_logdir(test_logdir=test_logdir)
    eval_pq_npz_file = evaluate.main(analysis_cache_outdir, overwrite=overwrite, iou_threshold=eval_iou_threshold)
    scatter_loss_vs_pq(loss_npz_file=loss_npz_file, pq_npz_file=eval_pq_npz_file)
    analysis_cache_outdir = os.path.dirname(eval_pq_npz_file)
    if visualize_pq_hists:
        visualize_pq_stats.main(eval_pq_npz_file)

    if export_sorted_perf_images:
        export_prediction_vs_gt_vis_sorted.main(collated_stats_npz=eval_pq_npz_file)

    return analysis_cache_outdir


def label_to_one_hot(input_label, n_classes, output_onehot=None, dtype=None, device=None):
    if dtype is None:
        dtype = input_label.dtype
    if device is None:
        device = input_label.device
    void_class = 1  # If void class could exist (-1), we'll leave room for it and then remove it.
    assert len(input_label.shape) == 2
    if output_onehot is None:
        out_size = (n_classes + void_class, input_label.shape[0], input_label.shape[1])
        output_onehot = torch.zeros(out_size, dtype=dtype, device=device)
    else:
        output_onehot = output_onehot.zero_()
    channel_dim = 0
    input_labels_expanded = input_label.view(1, input_label.shape[0], input_label.shape[1]) + void_class
    try:
        output_onehot.scatter_(channel_dim, input_labels_expanded, 1)
    except:
        import ipdb; ipdb.set_trace()
        raise
    output_onehot = output_onehot[void_class:, :, :]
    return output_onehot


def compute_losses(test_logdir, overwrite, scores_to_onehot=False):
    scores_outdir = os.path.join(test_logdir, 'scores')
    groundtruth_outdir = os.path.join(test_logdir, 'groundtruth')
    test_cfg_file = os.path.join(test_logdir, 'config.yaml')
    problem_config_file = os.path.join(test_logdir, 'instance_problem_config.yaml')
    with open(os.path.join(test_logdir, 'train_logdir.txt'), 'r') as fid:
        train_logdir = fid.read().strip()
    analysis_cache_outdir = get_cache_dir_from_test_logdir(test_logdir=test_logdir)
    dataset_cache_dir = os.path.dirname(os.path.dirname(analysis_cache_outdir.rstrip('/')))
    if not os.path.exists(dataset_cache_dir):
        raise Exception('Dataset cache doesnt exist.  Maybe the dataset name is wrong? {}'.format(dataset_cache_dir))
    if not os.path.exists(analysis_cache_outdir):
        os.makedirs(analysis_cache_outdir)
    compiled_loss_arr_outfile = os.path.join(analysis_cache_outdir, 'losses.npz' if not scores_to_onehot else
    'losses_onehot.npz')
    create_loss_npz = True
    if os.path.exists(compiled_loss_arr_outfile):
        if not overwrite:
            print('Already found {}.  If you would like to overwrite, add the flag --overwrite'.format(
                compiled_loss_arr_outfile))
            create_loss_npz = False
    else:
        print('Generating {}'.format(compiled_loss_arr_outfile))
    if create_loss_npz:
        cfg = yaml.safe_load(open(test_cfg_file, 'rb'))
        problem_config = instance_utils.InstanceProblemConfig.load(problem_config_file)

        score_files = sorted(glob.glob(os.path.join(scores_outdir, '*.pt')))
        gt_files = sorted(glob.glob(os.path.join(groundtruth_outdir, '*.png')))
        assert len(score_files) == len(gt_files)

        my_loss_object = loss.loss_object_factory(
            loss_type=cfg['loss_type'], semantic_instance_class_list=problem_config.semantic_instance_class_list,
            instance_id_count_list=problem_config.instance_count_id_list, matching=cfg['matching'],
            size_average=cfg['size_average'])

        n_images = len(score_files)
        losses_compiled_per_img_per_cls = -1 * np.ones((n_images, problem_config.n_semantic_classes))
        losses_compiled_per_img_per_channel = -1 * np.ones((n_images, problem_config.n_classes))

        for idx, (score_file, gt_file) in tqdm.tqdm(enumerate(zip(score_files, gt_files)), total=n_images,
                                                    desc='Getting losses for saved scores, GT'):
            score_3d = torch.load(score_file)
            if scores_to_onehot:
                labels_2d = score_3d.max(dim=0)[1]
                score_3d = label_to_one_hot(input_label=labels_2d, n_classes=score_3d.shape[0], dtype=score_3d.dtype)

            gt_im = load_gt_img_in_panoptic_form(gt_file).astype('int')
            gt_sem_3d, gt_inst_3d = problem_config.decompose_semantic_and_instance_labels(gt_im)
            score = score_3d.view(1, *score_3d.shape)
            gt_sem, gt_inst = torch.Tensor(gt_sem_3d[np.newaxis, ...]).cuda(device=score.device), \
                              torch.Tensor(gt_inst_3d[np.newaxis, ...]).cuda(device=score.device)
            pred_permutations, total_loss, loss_components = my_loss_object.loss_fcn(score, gt_sem, gt_inst)
            per_channel_loss = loss_components.cpu().numpy()
            per_cls_loss = problem_config.aggregate_across_same_sem_cls(per_channel_loss)
            losses_compiled_per_img_per_channel[idx, :] = per_channel_loss
            losses_compiled_per_img_per_cls[idx, :] = per_cls_loss

        d = {
            'losses_compiled_per_img_per_cls': losses_compiled_per_img_per_cls,
            'losses_compiled_per_img_per_channel': losses_compiled_per_img_per_channel,
            'problem_config': problem_config,
            'scores_outdir': scores_outdir,
            'groundtruth_outdir': groundtruth_outdir,
            'my_loss_object': my_loss_object
        }

        np.savez(compiled_loss_arr_outfile, **d)
        print('Losses saved to {}'.format(compiled_loss_arr_outfile))
    else:
        print('Losses already existed in {}'.format(compiled_loss_arr_outfile))

    return compiled_loss_arr_outfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_logdir',
                        help='Must contain \'scores\' and \'groundtruth\' directories' +
                             'Example: scripts/logs/synthetic/test_2019-08-12-131034_VCS-c8923f1__test_split-val')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--eval_iou_threshold', type=float, default=None)
    parser.add_argument('--which_models', type=str, default='best')
    parser.add_argument('--visualize_pq_hists', type=bool, default=True)
    parser.add_argument('--export_sorted_perf_images', type=bool, default=None)
    parser.add_argument('--scores_to_onehot', type=int, default=0)
    return parser.parse_args()


def get_list_of_model_paths(train_logdir, which_models):
    if which_models == 'best':
        list_of_model_paths = [os.path.join(train_logdir, 'model_best.pth.tar')]
    elif which_models == 'final':
        list_of_model_paths = [os.path.join(train_logdir, 'checkpoint.pth.tar')]
    else:
        raise NotImplementedError
    return list_of_model_paths


if __name__ == '__main__':
    args = parse_args()
    print(args.scores_to_onehot)
    compiled_loss_arr_outfile = main(args.test_logdir, args.overwrite, args.eval_iou_threshold,
                                     which_models=args.which_models, visualize_pq_hists=args.visualize_pq_hists,
                                     export_sorted_perf_images=args.export_sorted_perf_images,
                                     scores_to_onehot=args.scores_to_onehot)
