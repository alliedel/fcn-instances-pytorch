import argparse
import os

import yaml

from instanceseg.utils import display as display_pyutils

default_markers = display_pyutils.MARKERS
default_colors = display_pyutils.GOOD_COLOR_CYCLE

from instanceseg.datasets.dataset_generator_registry import get_default_datasets_for_instance_counts
from instanceseg.factory.samplers import get_samplers
from scripts import evaluate
from scripts.analysis import loss_vs_eval_metrics
import numpy as np
import json

from scripts.analysis.get_upsnet_pq_eval import upsnet_evaluate
from scripts.configurations import sampler_cfg_registry
import matplotlib.pyplot as plt
from scripts.visualizations.visualize_pq_stats import extract_variable, nanscatter_x_y, \
    swap_outer_and_inner_keys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_logdir',
                        help='Must contain \'scores\' and \'groundtruth\' directories' +
                             'Example: scripts/logs/synthetic/test_2019-08-12-131034_VCS-c8923f1__test_split-val')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--eval_iou_threshold', type=float, default=None)
    parser.add_argument('--visualize_pq_hists', type=bool, default=True)
    parser.add_argument('--export_sorted_perf_images', type=bool, default=None)
    return parser.parse_args()


def assemble_arr_from_dict_with_categories_by_name(stats_dict, orig_category_names_by_key, subsampled_category_names):
    cat_ids = []
    assert len(stats_dict.keys()) == len(orig_category_names_by_key.keys())
    name_to_key = {v: k for k, v in orig_category_names_by_key.items()}
    for cat_name in subsampled_category_names:
        assert cat_name in orig_category_names_by_key.values(), '{} not in original array with categories {}'.format(
            cat_name, orig_category_names_by_key)
        cat_ids.append(name_to_key[cat_name])
    stats_cols = [stats_dict[cat_id] for cat_id in cat_ids]
    if all([np.isscalar(s) for s in stats_cols]):
        stats_cols = [np.array(s).reshape(1,1) for s in stats_cols]
    elif all([len(s.shape) == 1 for s in stats_cols]):
        pass
    else:
        assert all(s.shape[1] == 1 for s in stats_cols)
    return np.concatenate(stats_cols, axis=1)


def subsample_arr_with_categories_by_name(stats_arr, orig_category_names, subsampled_category_names):
    cat_idxs = []
    assert stats_arr.shape[1] == len(orig_category_names)
    for cat_name in subsampled_category_names:
        assert cat_name in orig_category_names, '{} not in original array with categories {}'.format(
            cat_name, orig_category_names)
        cat_idxs.append(orig_category_names.index(cat_name))

    return stats_arr[:, np.array(cat_idxs)]


def compare_two_eval_npz_files(npz_1, npz_2, output_dir, identifier_1='1', identifier_2='2', save_to_workspace=True):
    ds = {
        identifier_1: dict(np.load(npz_1)),
        identifier_2: dict(np.load(npz_2))
    }
    for _, d in ds.items():
        for key in d.keys():
            d[key] = extract_variable(d, key)

    labels_tables = {
        k: d['problem_config'].labels_table for k, d in ds.items()
    }

    category_names_by_col = {}
    category_ids_to_name = {}
    for identifier, d in ds.items():
        ids_by_col = d['categories']
        labels_table = d['problem_config'].labels_table
        labels_table_ids = [l['id'] for l in labels_table]
        labels_table_idxs_by_col = [labels_table_ids.index(cat_id) for cat_id in ids_by_col]
        category_names_by_col[identifier] = [labels_table[table_idx]['name'] for table_idx in labels_table_idxs_by_col]
        category_ids_to_name[identifier] = {labels_table[idx].id: labels_table[idx].name
                                            for idx in labels_table_idxs_by_col}

    overlapping_category_names = \
        [c for c in category_names_by_col[identifier_1] if c in category_names_by_col[identifier_2]]
    overlapping_labels_table = [labels_tables[identifier_1][category_names_by_col[identifier_1].index(cat_name)]
                                for cat_name in overlapping_category_names]

    stats_arrays_total_dataset_per_subdiv = {}
    for identifier, d in ds.items():
        tot_dataset = extract_variable(d, 'dataset_totals')
        stats_arrays_total_dataset_per_subdiv[identifier] = \
            swap_outer_and_inner_keys({k: tot_dataset[k] for k in tot_dataset.keys() if k != 'per_class'})

    stats_arrays_per_img_same_cols = {}
    stats_arrays_total_dataset_per_class_by_stat_same_cols = {}
    for identifier, d in ds.items():
        per_image = extract_variable(d, 'collated_stats_per_image_per_cat')
        # get rid of nan by image
        for stat_type, stat_array_by_type in per_image.items():
            n_inst_arrs = per_image['n_inst']
            if stat_type != 'n_inst':
                stat_array_by_type[n_inst_arrs == 0] = np.nan

        stats_arrays_per_img_same_cols[identifier] = {k: subsample_arr_with_categories_by_name(
            v, category_names_by_col[identifier], overlapping_category_names) for k, v in
            per_image.items()}
        per_class = swap_outer_and_inner_keys(extract_variable(d, 'dataset_totals')['per_class'])
        stats_arrays_total_dataset_per_class_by_stat_same_cols[identifier] = {
            stat_type: assemble_arr_from_dict_with_categories_by_name(
                stat_arr, category_ids_to_name[identifier], overlapping_category_names)
            for stat_type, stat_arr in per_class.items()
        }

    stat_types = list(stats_arrays_per_img_same_cols[identifier_1].keys())
    R, C = len(stat_types), len(overlapping_category_names)
    assert C == stats_arrays_per_img_same_cols[identifier_1][stat_types[0]].shape[1]
    scatter_kwargs = dict(
        x_arrs_R_C=[[stats_arrays_per_img_same_cols[identifier_1][stat_type][:, col] for col in range(C)]
                    for stat_type in stat_types],
        y_arrs_R_C=[[stats_arrays_per_img_same_cols[identifier_2][stat_type][:, col] for col in range(C)]
                    for stat_type in stat_types],
        x_labels_c=[identifier_1 for _ in range(C)],
        y_labels_r=[identifier_2 for _ in range(R)],
        colors_c=[l.color for l in overlapping_labels_table],
        labels_R_C=[['{} {}'.format(stat_type, cat_name) for cat_name in overlapping_category_names]
                    for stat_type in stat_types]
    )
    all_scatters(**scatter_kwargs)
    figname = 'per_image_{}_vs_{}.png'.format(identifier_1, identifier_2)
    if save_to_workspace:
        display_pyutils.save_fig_to_workspace(figname)
    plt.savefig(os.path.join(output_dir, figname), dpi=500)

    fig_names = [figname]

    return fig_names


def all_scatters(x_arrs_R_C, y_arrs_R_C, x_labels_c=None, y_labels_r=None, colors_c=None, labels_R_C=None,
                 subtitles_c=None):
    R = len(x_arrs_R_C)
    C = len(x_arrs_R_C[0])
    for arr in [x_arrs_R_C, y_arrs_R_C, labels_R_C]:
        assert len(arr) == R and all(len(row) == C for row in arr)
    if x_labels_c is None:
        x_labels_c = [None for _ in range(C)]
    else:
        assert len(x_labels_c) == C
    if colors_c is None:
        colors_c = [None for _ in range(C)]
    else:
        assert len(colors_c) == C
    if subtitles_c is None:
        subtitles_c = [None for _ in range(C)]
    else:
        assert len(subtitles_c) == C
    if y_labels_r is None:
        y_labels_r = [None for _ in range(R)]
    else:
        assert len(y_labels_r) == R

    subplot_idx = 1
    figsize = (display_pyutils.BIG_FIGSIZE[0] * C / 10, display_pyutils.BIG_FIGSIZE[1] * R / 3)
    fig = plt.figure(1)
    plt.clf()
    plt.figure(1, figsize=figsize)
    for r in range(R):
        ax1 = plt.subplot(R, C, subplot_idx)
        ax1_subplot_num = subplot_idx
        plt.ylabel(y_labels_r[r], labelpad=20)
        for c in range(C):
            if subplot_idx == ax1_subplot_num:
                plt.subplot(R, C, subplot_idx)
            else:
                ax = plt.subplot(R, C, subplot_idx, sharex=ax1, sharey=ax1)
            if r == 0:
                plt.title(subtitles_c[c], rotation=45, y=1.4)
            if r == (R-1):
                plt.xlabel(x_labels_c[c])

            nanscatter_x_y(x_arrs_R_C[r][c], y_arrs_R_C[r][c], marker=None, alpha=1.0, c=colors_c[c],
                           label=labels_R_C[r][c], s=None, remove_nan=True)
            if c != 0:
                plt.setp(plt.gca().get_yticklabels(), visible=False)
            subplot_idx += 1

    fig.set_size_inches(figsize)


def main(test_logdir, overwrite=False, eval_iou_threshold=None, visualize_pq_hists=True,
         export_sorted_perf_images=True):
    analysis_cache_outdir = loss_vs_eval_metrics.get_cache_dir_from_test_logdir(test_logdir=test_logdir)
    us_eval_pq_npz_file = evaluate.main(analysis_cache_outdir, overwrite=overwrite,
                                        iou_threshold=eval_iou_threshold)
    upsnet_eval_pq_npz_file = upsnet_evaluate(eval_iou_threshold=eval_iou_threshold, overwrite=overwrite)
    ds = {
        'us': dict(np.load(us_eval_pq_npz_file)),
        'upsnet': dict(np.load(upsnet_eval_pq_npz_file))
    }

    gt_files = {
        k: json.load(open(d['gt_json_file'].item(), 'r')) for k, d in ds.items()
    }
    upsnet_reduced, reduction_tag = reduce_upsnet_to_image_set(ds, gt_files, test_logdir)
    upsnet_eval_pq_npz_file_reduced = upsnet_eval_pq_npz_file.replace('.npz', '_{}.npz'.format(reduction_tag))
    np.savez(upsnet_eval_pq_npz_file_reduced, **upsnet_reduced)
    output_dir = os.path.join(os.path.dirname(us_eval_pq_npz_file), 'figs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig_names = compare_two_eval_npz_files(npz_1=upsnet_eval_pq_npz_file_reduced, npz_2=us_eval_pq_npz_file,
                                           identifier_1='upsnet', identifier_2='us',
                                           output_dir=output_dir)
    print('Figures {} saved to {}'.format(fig_names, output_dir))
    return analysis_cache_outdir


def reduce_upsnet_to_image_set(ds, gt_files, test_logdir):
    train_cfg = yaml.load(open(os.path.join(test_logdir, 'config.yaml'), 'rb'))
    sampler_name = train_cfg['sampler']
    if 'val' in test_logdir:
        split = 'val'
    else:
        raise Exception
    dataset_type = 'cityscapes'
    assert dataset_type in test_logdir, 'Are we still working with Cityscapes?  Im assuming it in this code.'
    full_datasets, transformer_tag = get_default_datasets_for_instance_counts(dataset_type=dataset_type,
                                                                              splits=(split,))
    full_dataset = full_datasets[split]
    sampler_cfg = sampler_cfg_registry.get_sampler_cfg_set(sampler_name)
    sampler_cfg = {
        k: v for k, v in sampler_cfg.items() if k == split
    }
    samplers = get_samplers(dataset_type=dataset_type, sampler_cfg=sampler_cfg, datasets={split: full_dataset},
                            splits=(split,))
    upsnet_image_list = gt_files['upsnet']['images']
    assert len(upsnet_image_list) == len(samplers[split].initial_indices)
    sampler_idxs = samplers[split].indices
    upsnet_reduced = ds['upsnet']
    orig_collated_stats = ds['upsnet']['collated_stats_per_image_per_cat'].item()
    upsnet_reduced['collated_stats_per_image_per_cat'] = {k: v[sampler_idxs, :]
                                                          for k, v in orig_collated_stats.items()}
    upsnet_reduced['reduced_idxs'] = sampler_idxs
    upsnet_reduced['reduction_sampler'] = (dataset_type, sampler_cfg)
    reduction_tag = 'sampler_{}_{}'.format(dataset_type, sampler_cfg)
    return upsnet_reduced, reduction_tag


if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)
    # compiled_loss_arr_outfile = loss_vs_eval_metrics.main(args.test_logdir, args.overwrite, args.eval_iou_threshold,
    #                                                       which_models=args.which_models,
    #                                                       visualize_pq_hists=args.visualize_pq_hists,
    #                                                       export_sorted_perf_images=args.export_sorted_perf_images,
    #                                                       scores_to_onehot=args.scores_to_onehot)

    # d = {
    #     'losses_compiled_per_img_per_cls': losses_compiled_per_img_per_cls,
    #     'losses_compiled_per_img_per_channel': losses_compiled_per_img_per_channel,
    #     'problem_config': problem_config,
    #     'scores_outdir': scores_outdir,
    #     'groundtruth_outdir': groundtruth_outdir,
    #     'my_loss_object': my_loss_object
    # }
