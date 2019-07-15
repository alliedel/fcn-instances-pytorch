import os
import numpy as np
import matplotlib.pyplot as plt


def bar_graph(x, height, width=1.0):
    plt.bar(x, height, width=width)


collated_stats_npz_file = 'cache/synthetic/train_instances_filtered_2019-06-24-163353_VCS-8df0680/' \
                          'collated_stats_per_img_per_cat.npz'
# collated_stats_npz_file = 'cache/cityscapes/train_instances_filtered_2019-05-14-133452_VCS-1e74989_SAMPLER-' \
#                           'car_2_4_BACKBONE-resnet50_ITR-1000000_NPER-4_SSET-car_person/' \
#                           'collated_stats_per_img_per_cat.npz'

loaded_dict = np.load(collated_stats_npz_file)

stats_arrays = loaded_dict['collated_stats_per_image_per_cat'].item()
categories = loaded_dict['categories'].item()
problem_config = loaded_dict['problem_config'].item()
del loaded_dict

average_per_cat = {}
std_per_cat = {}

for stat_type, stat_array in stats_arrays.items():
    average_per_cat[stat_type] = np.average(stat_array, axis=0)
    std_per_cat[stat_type] = np.std(stat_array, axis=0)

category_idxs_to_display = range(1, problem_config.n_semantic_classes)

for stat_type in average_per_cat.keys():
    plt.figure(1); plt.clf()
    fig, ax = plt.subplots()
    ax.bar(category_idxs_to_display, average_per_cat[stat_type][category_idxs_to_display],
           yerr=std_per_cat[stat_type][category_idxs_to_display], align='center',
           alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(stat_type)
    ax.set_xticks(category_idxs_to_display)
    ax.set_xticklabels(list(categories[i]['name'] for i in category_idxs_to_display))
    ax.set_title(stat_type)
    ax.yaxis.grid(True)

    # bar_graph(x=category_idxs_to_display, height=average_per_cat[stat_type][category_idxs_to_display])
    plt.savefig(stat_type + '.png')
