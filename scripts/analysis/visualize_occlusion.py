import instanceseg.utils.display as display_pyutils
import matplotlib.pyplot as plt
import numpy as np
from instanceseg.datasets import dataset_statistics, dataset_registry, dataset_generator_registry

FIGSIZE = (8, 8)
DPI = 300


def pie_chart(values_as_list, column_labels, autopct='%1.1f%%'):
    explode = [0.1 * int(val != max(values_as_list)) for i, val in
               enumerate(values_as_list)]  # explode largest slice
    # Plot
    colors = [display_pyutils.GOOD_COLORS[np.mod(i, len(display_pyutils.GOOD_COLORS))] for i in
              range(len(values_as_list))]
    patches, text, _ = plt.pie(values_as_list, explode=explode, labels=column_labels, colors=colors,
                               autopct=autopct, shadow=True, startangle=140)
    plt.axis('equal')

    sort_legend = True
    if sort_legend:
        patches, column_labels, dummy = zip(*sorted(zip(patches, column_labels, values_as_list),
                                                    key=lambda x: x[2],
                                                    reverse=True))

    plt.legend(patches, column_labels, loc='center left', fontsize=8, bbox_to_anchor=(-0.02, 0.5))
    plt.tight_layout(pad=1.2)


def write_stats(stats, semantic_class_names, split, metric_name='unknown_stat',
                workspace_dir=display_pyutils.WORKSPACE_DIR):
    # Write statistics
    # Data to plot
    display_pyutils.set_my_rc_defaults()
    counts_np = stats.numpy()

    if counts_np.shape[1] != len(semantic_class_names):
        import ipdb;
        ipdb.set_trace()
        raise ValueError('semantic class names dont match size of counts')

    counts_per_class = counts_np.sum(axis=0)
    image_counts_per_class = (counts_np > 0).sum(axis=0)
    num_class_per_img = {
        semantic_class: counts_np[:, semantic_class_names.index(semantic_class)] for
        semantic_class in semantic_class_names
    }
    num_images_with_this_many_cls = {
        scn: [np.sum(num_class_per_img[scn] == x) for x in
              range(int(num_class_per_img[scn].max()) + 1)]
        for scn in num_class_per_img.keys()
    }

    plt.figure(1, figsize=FIGSIZE)
    plt.clf()
    ttl = split + '_' + 'total_' + metric_name + '_per_class' + '_across_dataset'
    metric_values = counts_per_class
    labels = [s + '={}'.format(v) for s, v in zip(semantic_class_names, metric_values)]
    pie_chart(metric_values, labels)
    plt.title(ttl)
    display_pyutils.save_fig_to_workspace(ttl + '.png', workspace_dir=workspace_dir)

    plt.figure(2, figsize=FIGSIZE)
    plt.clf()
    ttl = split + '_' + 'num_images_with_at_least_1' + metric_name + '_per_class'
    metric_values = image_counts_per_class
    labels = [s + '={}'.format(v) for s, v in zip(semantic_class_names, metric_values)]
    pie_chart(metric_values, labels)
    plt.title(ttl)
    display_pyutils.save_fig_to_workspace(ttl + '.png', workspace_dir=workspace_dir)

    for semantic_class in semantic_class_names:
        plt.figure(3, figsize=FIGSIZE)
        plt.clf()
        ttl = split + '_' + 'num_images_per_num_{}_{}'.format(semantic_class, metric_name)
        metric_values = num_images_with_this_many_cls[semantic_class]
        labels = ['{} {}={}'.format(num, semantic_class, v) for num, v in enumerate(metric_values)]
        pie_chart(metric_values, labels)
        plt.title(ttl)
        display_pyutils.save_fig_to_workspace(ttl + '.png', workspace_dir=workspace_dir)
    print('Images written to {}'.format(workspace_dir))


def main(dataset_type='cityscapes'):
    default_train_dataset, default_val_dataset, transformer_tag = \
        dataset_generator_registry.get_default_datasets_for_instance_counts(dataset_type)
    # occlusion_counts_file = dataset_registry.REGISTRY['cityscapes'].get_occlusion_count_filename(
    #     split='train', transformer_tag=transformer_tag)
    default_datasets = {
        'train': default_train_dataset,
        'val': default_val_dataset
    }
    occlusion_counts_files = {
        split: dataset_registry.REGISTRY[dataset_type].get_occlusion_counts_filename(
            split, transformer_tag)
        for split in ('train', 'val')
    }
    semantic_class_names = default_train_dataset.semantic_class_names
    occlusion_counts = {
        split: dataset_statistics.OcclusionsOfSameClass(
            range(len(semantic_class_names)), semantic_class_names=semantic_class_names,
            cache_file=occlusion_counts_files[split]) for split in ('train', 'val')
    }
    # occlusion_counts = {
    #     split: dataset_statistics.OcclusionsOfSameClass(
    #         range(semantic_class_names), semantic_class_names=semantic_class_names,
    #         cache_file=occlusion_counts_files[split])
    #     for split in ('train', 'val')
    # }
    for split in ('train', 'val'):
        occlusion_counts[split].compute_or_retrieve(default_datasets[split])
    for split in ('train', 'val'):
        write_stats(occlusion_counts[split].stat_tensor, split=split,
                    semantic_class_names=semantic_class_names, metric_name='occlusion_counts')


if __name__ == '__main__':
    main()
