import argparse

import matplotlib.pyplot as plt
import numpy as np
import os

from instanceseg.utils import display as display_pyutils

default_markers = display_pyutils.MARKERS
default_colors = display_pyutils.GOOD_COLOR_CYCLE


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


def swap_outer_and_inner_keys(d):
    outer_keys = list(d.keys())
    inner_keys = list(d[outer_keys[0]].keys())
    for k, v in d.items():
        assert set(v.keys()) == set(inner_keys), 'All inner keys in dict must be the same: {} vs {}'.format(set(
            v.keys()), set(inner_keys))
    new_dict = {}
    for inner_k in inner_keys:
        new_dict[inner_k] = {
            outer_k: d[outer_k][inner_k] for outer_k in outer_keys
        }
    return new_dict


def autolabel_above_bars(ax, fontsize=4, vertical_offset=3):
    """Attach a text label above each bar in *rects*, displaying its height."""
    rects = ax.patches
    for rect in rects:
        height = rect.get_height()
        label = '{:.2g}'.format(height)

        # ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')
        ax.annotate(label, xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, vertical_offset),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=fontsize)


def plot_bar_graph_over_full_dataset(stats_arrays_total, stats_arrays_per_class, category_names_by_id,
                                     categories_to_display=None, save_to_workspace=True, output_dir='/tmp/',
                                     category_colors_by_id=None):
    stat_types = [k for k in stats_arrays_total.keys() if k != 'n']
    assert set(stat_types) == set(stats_arrays_per_class.keys()), \
        'Keys must be the same: {} vs {}'.format(stats_arrays_total.keys(), stats_arrays_per_class.keys())
    example_stat_type = list(stats_arrays_per_class.keys())[0]
    categories_available = list(stats_arrays_per_class[example_stat_type].keys())
    assert all(set(categories_available) == set(stats_arrays_per_class[stat_type].keys())
               for stat_type in stat_types)
    total_types_available = list(stats_arrays_total[example_stat_type].keys())
    assert all(set(total_types_available) == set(stats_arrays_total[stat_type].keys())
               for stat_type in stat_types)

    categories_to_display = categories_to_display if categories_to_display is not None else categories_available

    category_colors_by_id = category_colors_by_id or \
                            {cid: default_colors[cid % len(categories_to_display)]
                             for cid in range(len(categories_to_display))}

    facecolors = [[0.0, 0.0, 0.0] for _ in total_types_available] + \
                 [category_colors_by_id[cat_id] for cat_id in categories_to_display]
    edgecolors = [[0.0, 0.0, 0.0] for _ in total_types_available] + \
                 [category_colors_by_id[cat_id] for cat_id in categories_to_display]
    for stat_type in stat_types:
        ylabel = stat_type
        y_cat = [stats_arrays_per_class[stat_type][cat_id] for cat_id in categories_to_display]
        y_tots = [stats_arrays_total[stat_type][total_type] for total_type in total_types_available]
        y = y_tots + y_cat
        x_offset = 1
        x = [i for i in range(len(y_tots))] + [i + x_offset + len(y_tots) for i in range(len(y_cat))]
        x_labels = total_types_available + [category_names_by_id[cat_id] for cat_id in categories_to_display]

        plt.figure(1)
        plt.clf()
        fig, ax = plt.subplots()
        rects = ax.bar(x=x, height=y, yerr=None, align='center', alpha=0.9, edgecolor=edgecolors,
                       color=facecolors, capsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation='vertical')
        ax.set_title(ylabel)
        ax.yaxis.grid(True)
        autolabel_above_bars(ax)
        figname = 'avg_' + ylabel + '.png'
        plt.tight_layout()
        if save_to_workspace:
            display_pyutils.save_fig_to_workspace(figname)
        plt.savefig(os.path.join(output_dir, figname))


def plot_averages_with_error_bars(stats_arrays, semantic_class_names, category_idxs_to_display=None,
                                  save_to_workspace=True, output_dir='/tmp/', category_colors=None):
    """
    Note: we filter n_inst == 0 from the averages.
    """

    category_colors = category_colors or [default_colors[i % len(semantic_class_names)] for i in range(len(
        semantic_class_names))]

    category_idxs_to_display = category_idxs_to_display if category_idxs_to_display is not None \
        else range(stats_arrays[stats_arrays.keys()[0]].shape[1])
    average_per_cat = {}
    std_per_cat = {}

    for stat_type, stat_array in stats_arrays.items():
        average_per_cat[stat_type] = np.nanmean(stat_array, axis=0)
        std_per_cat[stat_type] = np.nanstd(stat_array, axis=0)

    category_names_to_display = [semantic_class_names[idx] for idx in category_idxs_to_display]
    stat_types = average_per_cat.keys()
    xs = [range(len(category_idxs_to_display)) for _ in range(len(stat_types))]
    ys = [average_per_cat[stat_type][category_idxs_to_display] for stat_type in stat_types]
    stds = [std_per_cat[stat_type][category_idxs_to_display] for stat_type in stat_types]

    for ylabel, x, y, std in zip(stat_types, xs, ys, stds):
        plt.figure(1)
        plt.clf()
        fig, ax = plt.subplots()
        ax.bar(x=x, height=y, yerr=std, align='center', alpha=0.5, ecolor='black',
               color=[category_colors[i] for i in category_idxs_to_display], capsize=5)
        ax.set_ylabel(ylabel)
        ax.set_xticks(category_idxs_to_display)
        ax.set_xticklabels(list(category_names_to_display), rotation='vertical')
        ax.set_title(ylabel)
        ax.yaxis.grid(True)
        figname = 'avg_with_err_' + ylabel + '.png'
        plt.tight_layout()
        if save_to_workspace:
            display_pyutils.save_fig_to_workspace(figname)
        plt.savefig(os.path.join(output_dir, figname))


def plot_scatterplot_sq_rq(stats_arrays, semantic_class_names, category_idxs_to_display=None, save_to_workspace=True,
                           output_dir='/tmp/', category_colors=None, xlim=(0,1), ylim=(0,1)):
    category_colors = category_colors or [default_colors[i % len(semantic_class_names)]
                                          for i in range(len(semantic_class_names))]
    category_idxs_to_display = category_idxs_to_display if category_idxs_to_display is not None \
        else range(stats_arrays[stats_arrays.keys()[0]].shape[1])
    category_names_to_display = [semantic_class_names[idx] for idx in category_idxs_to_display]
    sqs = [stats_arrays['sq'][:, i] for i in category_idxs_to_display]
    rqs = [stats_arrays['rq'][:, i] for i in category_idxs_to_display]

    plt.figure(1)
    plt.clf()
    nanscatter_list_of_xs_and_ys(sqs, rqs, labels=category_names_to_display, xlabel='SQ', ylabel='RQ',
                                 colors=category_colors)
    figname = 'sq_vs_rq' + '.png'
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    if save_to_workspace:
        display_pyutils.save_fig_to_workspace(figname)
    plt.savefig(os.path.join(output_dir, figname))

    for cat_idx, sq_cat, rq_cat in zip(category_idxs_to_display, sqs, rqs):
        plt.figure(1)
        plt.clf()
        cat_color = category_colors[cat_idx]
        cat_name = semantic_class_names[cat_idx]
        nanscatter_list_of_xs_and_ys([sq_cat], [rq_cat], labels=[cat_name], xlabel='SQ', ylabel='RQ',
                                     colors=[cat_color])
        figname = 'sq_vs_rq_{}'.format(cat_name) + '.png'
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)
        if save_to_workspace:
            display_pyutils.save_fig_to_workspace(figname)
        plt.savefig(os.path.join(output_dir, figname))


def plot_scatterplot_sq_ninst(stats_arrays, semantic_class_names, category_idxs_to_display=None, save_to_workspace=True,
                              output_dir='/tmp/', category_colors=None, xlim=None, ylim=(0,1)):
    category_colors = category_colors or [default_colors[i % len(semantic_class_names)]
                                          for i in range(len(semantic_class_names))]
    category_idxs_to_display = category_idxs_to_display if category_idxs_to_display is not None \
        else range(stats_arrays[stats_arrays.keys()[0]].shape[1])
    category_names_to_display = [semantic_class_names[idx] for idx in category_idxs_to_display]
    sqs = [stats_arrays['sq'][:, i] for i in category_idxs_to_display]
    ninst = [stats_arrays['n_inst'][:, i] for i in category_idxs_to_display]

    plt.figure(1)
    plt.clf()
    ninst_nan = [np.copy(a) for a in ninst]
    for a, b in zip(ninst_nan, ninst):
        a[b == 0] = np.nan
    nanscatter_list_of_xs_and_ys(ninst_nan, sqs, labels=category_names_to_display, xlabel='# inst', ylabel='SQ',
                                 colors=category_colors)
    figname = 'ninst_vs_sq' + '.png'
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    if save_to_workspace:
        display_pyutils.save_fig_to_workspace(figname)
    plt.savefig(os.path.join(output_dir, figname))

    for cat_idx, sq_cat, ninst_cat in zip(category_idxs_to_display, sqs, ninst_nan):
        if not np.all(ninst_cat[~np.isnan(ninst_cat)].astype(int) == ninst_cat[~np.isnan(ninst_cat)]):
            raise Exception('Number of instances should be integer-valued')
        plt.figure(1)
        plt.clf()
        cat_color = category_colors[cat_idx]
        cat_name = semantic_class_names[cat_idx]
        nanscatter_list_of_xs_and_ys([ninst_cat], [sq_cat], labels=[cat_name], xlabel='# inst', ylabel='SQ',
                                     colors=[cat_color])
        figname = 'ninst_vs_sq_{}'.format(cat_name) + '.png'
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)
        if save_to_workspace:
            display_pyutils.save_fig_to_workspace(figname)
        plt.savefig(os.path.join(output_dir, figname))


def plot_scatterplot_rq_ninst(stats_arrays, semantic_class_names, category_idxs_to_display=None, save_to_workspace=True,
                              output_dir='/tmp/', category_colors=None, xlim=None, ylim=(0, 1)):
    category_colors = category_colors or [default_colors[i % len(semantic_class_names)]
                                          for i in range(len(semantic_class_names))]
    category_idxs_to_display = category_idxs_to_display if category_idxs_to_display is not None \
        else range(stats_arrays[stats_arrays.keys()[0]].shape[1])
    category_names_to_display = [semantic_class_names[idx] for idx in category_idxs_to_display]
    rqs = [stats_arrays['rq'][:, i] for i in category_idxs_to_display]
    ninst = [stats_arrays['n_inst'][:, i] for i in category_idxs_to_display]
    ninst_nan = [np.copy(a) for a in ninst]
    for a, b in zip(ninst_nan, ninst):
        a[b == 0] = np.nan

    plt.figure(1)
    plt.clf()
    nanscatter_list_of_xs_and_ys(ninst_nan, rqs, labels=category_names_to_display, xlabel='# inst', ylabel='RQ',
                                 colors=category_colors)
    figname = 'ninst_vs_rq' + '.png'
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    if save_to_workspace:
        display_pyutils.save_fig_to_workspace(figname)
    plt.savefig(os.path.join(output_dir, figname))

    for cat_idx, rq_cat, ninst_cat in zip(category_idxs_to_display, rqs, ninst_nan):
        if not np.all(ninst_cat[~np.isnan(ninst_cat)].astype(int) == ninst_cat[~np.isnan(ninst_cat)]):
            raise Exception('Number of instances should be integer-valued')
        plt.figure(1)
        plt.clf()
        cat_color = category_colors[cat_idx]
        cat_name = semantic_class_names[cat_idx]
        nanscatter_list_of_xs_and_ys([ninst_cat], [rq_cat], labels=[cat_name], xlabel='# inst', ylabel='RQ',
                                     colors=[cat_color])
        figname = 'ninst_vs_rq_{}'.format(cat_name) + '.png'
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)
        if save_to_workspace:
            display_pyutils.save_fig_to_workspace(figname)
        plt.savefig(os.path.join(output_dir, figname))


def plot_hists_pq_rq_sq(stats_arrays, semantic_class_names, category_idxs_to_display=None, save_to_workspace=True,
                        output_dir='/tmp/', category_colors=None, ylim=None):
    category_colors = category_colors or [default_colors[i % len(semantic_class_names)] for i in range(len(
        semantic_class_names))]
    category_idxs_to_display = category_idxs_to_display if category_idxs_to_display is not None \
        else range(stats_arrays[stats_arrays.keys()[0]].shape[1])
    category_names_to_display = [semantic_class_names[idx] for idx in category_idxs_to_display]

    for stat_type, stat_array in stats_arrays.items():
        for catidx, catname in zip(category_idxs_to_display, category_names_to_display):
            plt.figure(1)
            plt.clf()
            display_pyutils.nanhistogram(stat_array[:, catidx], label=catname, color=category_colors[catidx])
            figname = '{}_{}_hist'.format(stat_type, catname) + '.png'
            plt.title(figname.replace('.png', '').replace('_', ' '))
            if ylim is not None or stat_type == 'n_inst':
                plt.ylim(ylim)
            if save_to_workspace:
                display_pyutils.save_fig_to_workspace(figname)
            plt.savefig(os.path.join(output_dir, figname))

    subplot_idx = 1
    R, C = len(stats_arrays.keys()), len(category_idxs_to_display)
    figsize = (display_pyutils.BIG_FIGSIZE[0] * C / 10, display_pyutils.BIG_FIGSIZE[1] * R / 3)
    fig = plt.figure(1)
    plt.clf()
    plt.figure(1, figsize=figsize)
    r = 0
    for stat_type, stat_array in stats_arrays.items():
        ax1 = plt.subplot(R, C, subplot_idx)
        ax1_subplot_num = subplot_idx
        plt.ylabel(stat_type + ' (#images)', labelpad=20)
        for catidx, catname in zip(category_idxs_to_display, category_names_to_display):
            if subplot_idx == ax1_subplot_num:
                plt.subplot(R, C, subplot_idx)
            else:
                ax = plt.subplot(R, C, subplot_idx, sharex=ax1, sharey=ax1)
            if r == 0:
                plt.title(catname, rotation=45, y=1.4)
            display_pyutils.nanhistogram(stat_array[:, catidx], label=catname, color=category_colors[catidx])
            if subplot_idx != ax1_subplot_num:
                plt.setp(plt.gca().get_yticklabels(), visible=False)
            plottype = '{}_{}'.format(stat_type, catname)
            subplotname = '{}_hist'.format(plottype)
            subplot_idx += 1
        r += 1

    # plt.tight_layout()
    # if ylim is not None:
    #     plt.ylim(ylim)
    figname = 'all_hist.png'
    fig.set_size_inches(figsize)
    if save_to_workspace:
        display_pyutils.save_fig_to_workspace(figname)
    plt.savefig(os.path.join(output_dir, figname), dpi=500)


def nanscatter_list_of_xs_and_ys(xs, ys, labels=None, xlabel=None, ylabel=None, colors=None, remove_nan=True):
    if labels is None:
        labels = ['{}'.format(i) for i in range(len(xs))]
    markers = display_pyutils.MARKERS
    colors = colors or display_pyutils.GOOD_COLOR_CYCLE
    size = 30
    for i, (x, y) in enumerate(zip(xs, ys)):
        clr_idx = i % len(colors)
        marker_idx = i % len(markers)
        clr = colors[clr_idx]
        nanscatter_x_y(x, y, alpha=0.5, marker=markers[marker_idx], s=size, c=clr, label=labels[i],
                       remove_nan=remove_nan)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


def nanscatter_x_y(x, y, marker=None, alpha=1.0, c=None, label=None, s=None, remove_nan=True):
    if c is not None:
        if max(c) <= 1:
            denom = 1.0
        else:
            denom = 255.0
        c = np.array([c / denom for c in c]).reshape(1, 3)  # Turn into a tuple instead of ndarray
    if remove_nan:
        assert np.all(~np.isnan(x) == ~np.isnan(y))
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
    plt.scatter(x, y, alpha=alpha, marker=marker, s=s, c=c, edgecolors=c, label=label)
    plt.legend()


def main(collated_stats_npz_file, supercategories_to_ignore=('void', 'background'), values_to_ignore=(-1,)):
    loaded_dict = dict(np.load(collated_stats_npz_file, allow_pickle=True))
    stats_arrays_per_img = extract_variable(loaded_dict, 'collated_stats_per_image_per_cat')
    stats_arrays_total_dataset = extract_variable(loaded_dict, 'dataset_totals')
    vals_to_ignore_in_collation = extract_variable(loaded_dict, 'vals_to_ignore_in_collation')
    categories = extract_variable(loaded_dict, 'categories')
    problem_config = extract_variable(loaded_dict, 'problem_config')
    fig_output_dir = os.path.join(os.path.dirname(collated_stats_npz_file), 'figs')
    if not os.path.exists(fig_output_dir):
        os.makedirs(fig_output_dir)
    corresponding_labels_table = [l for l in problem_config.labels_table if l['id'] not in vals_to_ignore_in_collation]

    assert set(categories) == set([l['id'] for l in corresponding_labels_table])

    category_idxs_to_display = [i for i, l in enumerate(corresponding_labels_table)
                                if l['supercategory'] not in supercategories_to_ignore
                                and l['id'] not in values_to_ignore]
    semantic_class_names = [corresponding_labels_table[i]['name'] for i in category_idxs_to_display]
    del loaded_dict

    colors_norm1 = [np.array(l.color) / 255.0 for l in corresponding_labels_table]

    # Per image plots

    for stat_type, stat_array in stats_arrays_per_img.items():
        n_inst_arrs = stats_arrays_per_img['n_inst']
        if stat_type != 'n_inst':
            stat_array[n_inst_arrs == 0] = np.nan
    semantic_class_names_before_selection = [l['name'] for l in corresponding_labels_table]
    plot_averages_with_error_bars(
        stats_arrays_per_img, semantic_class_names=semantic_class_names_before_selection,
        category_idxs_to_display=category_idxs_to_display,
        output_dir=fig_output_dir, category_colors=colors_norm1)
    plot_scatterplot_sq_rq(stats_arrays_per_img, semantic_class_names=semantic_class_names_before_selection,
                           category_idxs_to_display=category_idxs_to_display, output_dir=fig_output_dir,
                           category_colors=colors_norm1)
    plot_scatterplot_sq_ninst(stats_arrays_per_img, semantic_class_names=semantic_class_names_before_selection,
                              category_idxs_to_display=category_idxs_to_display, output_dir=fig_output_dir,
                              category_colors=colors_norm1)
    plot_scatterplot_rq_ninst(stats_arrays_per_img, semantic_class_names=semantic_class_names_before_selection,
                              category_idxs_to_display=category_idxs_to_display, output_dir=fig_output_dir,
                              category_colors=colors_norm1)

    plot_hists_pq_rq_sq(stats_arrays_per_img, semantic_class_names=semantic_class_names_before_selection,
                        category_idxs_to_display=category_idxs_to_display, output_dir=fig_output_dir,
                        category_colors=colors_norm1)
    stat_arrays_total_dataset_per_class_by_stat = swap_outer_and_inner_keys(stats_arrays_total_dataset['per_class'])
    stat_arrays_total_dataset_per_subdiv = swap_outer_and_inner_keys({k: stats_arrays_total_dataset[k] for k in
                                                                      stats_arrays_total_dataset.keys() if k !=
                                                                      'per_class'})
    categories_to_display = [corresponding_labels_table[i].id for i in category_idxs_to_display]
    plot_bar_graph_over_full_dataset(stat_arrays_total_dataset_per_subdiv, stat_arrays_total_dataset_per_class_by_stat,
                                     category_names_by_id={corresponding_labels_table[i].id:
                                                               corresponding_labels_table[i].name
                                                           for i in category_idxs_to_display},
                                     categories_to_display=categories_to_display,
                                     output_dir=fig_output_dir,
                                     category_colors_by_id={l.id: np.array(l.color) / 255.0
                                                            for l in corresponding_labels_table})

    # Dataset total plots


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('collated_stats_npz', type=str, help='Output file from eval; typically '
                                                             'collated_stats_per_img_per_cat.npz in '
                                                             'the test output directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    main(collated_stats_npz_file=args.collated_stats_npz)
