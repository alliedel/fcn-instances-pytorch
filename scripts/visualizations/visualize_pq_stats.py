import argparse

import matplotlib.pyplot as plt
import numpy as np
import os

from instanceseg.utils import display as display_pyutils


def bar_graph(x, height, width=1.0):
    plt.bar(x, height, width=width)


def extract_variable(loaded_dict, key_name):
    value = loaded_dict[key_name]
    try:
        value_shape = value.shape
    except AttributeError:
        return value
    if value_shape == ():
        return value.item()
    else:
        return value


def plot_averages_with_error_bars(stats_arrays, problem_config, category_idxs_to_display=None, save_to_workspace=True,
                                  output_dir='/tmp/'):
    category_idxs_to_display = category_idxs_to_display if category_idxs_to_display is not None \
        else range(stats_arrays[stats_arrays.keys()[0]].shape[1])
    average_per_cat = {}
    std_per_cat = {}

    for stat_type, stat_array in stats_arrays.items():
        print(stat_type, np.average(stat_array, axis=0))
        average_per_cat[stat_type] = np.average(stat_array, axis=0)
        std_per_cat[stat_type] = np.std(stat_array, axis=0)

    category_names_to_display = [problem_config.semantic_class_names[idx] for idx in category_idxs_to_display]
    for stat_type in average_per_cat.keys():
        plt.figure(1)
        plt.clf()
        fig, ax = plt.subplots()
        ax.bar(category_idxs_to_display, average_per_cat[stat_type][category_idxs_to_display],
               yerr=std_per_cat[stat_type][category_idxs_to_display], align='center',
               alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel(stat_type)
        ax.set_xticks(category_idxs_to_display)
        ax.set_xticklabels(list(category_names_to_display))
        ax.set_title(stat_type)
        ax.yaxis.grid(True)
        figname = 'avg_' + stat_type + '.png'
        if save_to_workspace:
            display_pyutils.save_fig_to_workspace(figname)
        plt.savefig(os.path.join(output_dir, figname))


def plot_scatterplot_sq_rq(stats_arrays, problem_config, category_idxs_to_display=None, save_to_workspace=True,
                           output_dir='/tmp/'):
    category_idxs_to_display = category_idxs_to_display if category_idxs_to_display is not None \
        else range(stats_arrays[stats_arrays.keys()[0]].shape[1])
    category_names_to_display = [problem_config.semantic_class_names[idx] for idx in category_idxs_to_display]
    sqs = [stats_arrays['sq'][:, i] for i in category_idxs_to_display]
    rqs = [stats_arrays['rq'][:, i] for i in category_idxs_to_display]

    plt.figure(1)
    plt.clf()
    scatter_list_of_xs_and_ys(sqs, rqs, labels=category_names_to_display, xlabel='SQ', ylabel='RQ')
    figname = 'sq_vs_rq' + '.png'
    if save_to_workspace:
        display_pyutils.save_fig_to_workspace(figname)
    plt.savefig(os.path.join(output_dir, figname))


def plot_hists_pq_rq_sq(stats_arrays, problem_config, category_idxs_to_display=None, save_to_workspace=True,
                        output_dir='/tmp/'):
    category_idxs_to_display = category_idxs_to_display if category_idxs_to_display is not None \
        else range(stats_arrays[stats_arrays.keys()[0]].shape[1])
    category_names_to_display = [problem_config.semantic_class_names[idx] for idx in category_idxs_to_display]

    for stat_type, stat_array in stats_arrays.items():
        for catidx, catname in zip(category_idxs_to_display, category_names_to_display):
            plt.figure(1)
            plt.clf()
            display_pyutils.histogram(stat_array[:, catidx], label=catname)
            figname = '{}_{}_hist'.format(stat_type, catname) + '.png'
            plt.title(figname.replace('.png','').replace('_', ' '))
            if save_to_workspace:
                display_pyutils.save_fig_to_workspace(figname)
            plt.savefig(os.path.join(output_dir, figname))


def scatter_list_of_xs_and_ys(xs, ys, labels=None, xlabel=None, ylabel=None):
    if labels is None:
        labels = ['{}'.format(i) for i in range(len(xs))]
    markers = display_pyutils.MARKERS
    colors = display_pyutils.GOOD_COLOR_CYCLE
    size = 30
    for i, (sq, rq) in enumerate(zip(xs, ys)):
        plt.scatter(sq, rq, alpha=0.5, marker=markers[i], s=size, c=colors[i], edgecolors=colors[i], label=labels[i])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def main(collated_stats_npz_file):
    loaded_dict = np.load(collated_stats_npz_file)
    stats_arrays = extract_variable(loaded_dict, 'collated_stats_per_image_per_cat')
    categories = extract_variable(loaded_dict, 'categories')
    problem_config = extract_variable(loaded_dict, 'problem_config')
    fig_output_dir = os.path.join(os.path.dirname(collated_stats_npz_file), 'figs')
    if not os.path.exists(fig_output_dir):
        os.makedirs(fig_output_dir)
    assert set(categories) == set(problem_config.semantic_vals)
    del loaded_dict

    category_idxs_to_display = range(1, problem_config.n_semantic_classes)
    plot_averages_with_error_bars(stats_arrays, problem_config, category_idxs_to_display=category_idxs_to_display,
                                  output_dir=fig_output_dir)
    plot_scatterplot_sq_rq(stats_arrays, problem_config, category_idxs_to_display=category_idxs_to_display,
                           output_dir=fig_output_dir)
    plot_hists_pq_rq_sq(stats_arrays, problem_config, category_idxs_to_display=category_idxs_to_display,
                        output_dir=fig_output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('collated_stats_npz', type=str, help='Output file from eval; typically '
                                                             'collated_stats_per_img_per_cat.npz in '
                                                             'the test output directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    main(collated_stats_npz_file=args.collated_stats_npz)
