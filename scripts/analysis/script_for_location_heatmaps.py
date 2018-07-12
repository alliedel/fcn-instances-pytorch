import argparse
import os.path as osp

import display_pyutils
import matplotlib.pyplot as plt
import torch

import torchfcn.utils.logs
import torchfcn.utils.scripts
from torchfcn.analysis import score_heatmaps


FIGSIZE = (10, 10)
DPI = 300


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='directory that contains the checkpoint and config', type=str, required=True)
    parser.add_argument('--gpu', help='gpu identifier (int)', type=int, default=0)
    parser.add_argument('--relative', help='should export relative positions?', type=bool, default=True)
    parser.add_argument('--absolute', help='should export absolute positions?', type=bool, default=True)
    parser.add_argument('--clim_mult', help='should export absolute positions?', type=bool, default=True)
    args = parser.parse_args()
    return args


COLORMAP = 'cubehelix'  # 'cubehelix', 'inferno', 'magma', 'viridis', 'gray'


def write_absolute_heatmaps(absolute_heatmap_average, split, instance_problem):
    sem_inst_class_list = instance_problem.semantic_instance_class_list
    inst_id_list = instance_problem.instance_count_id_list
    channel_names = instance_problem.get_channel_labels()
    plt.figure(0, figsize=FIGSIZE)
    plt.clf()
    for channel_idx, (sem_idx, inst_id) in enumerate(zip(sem_inst_class_list, inst_id_list)):
        channel_name = channel_names[channel_idx]
        display_pyutils.matshow_and_save_list_to_workspace(
            [absolute_heatmap_average[channel_idx, :, :]],
            filename_base_ext='{}_score_heatmaps_{}.png'.format(split, channel_name),
            show_filenames_as_titles=True, cmap=COLORMAP, dpi=DPI)


def write_relative_heatmaps_by_channel(list_of_relative_heatmap_averages, instance_problem, split):

    channel_names = instance_problem.get_channel_labels()
    n_channels = len(list_of_relative_heatmap_averages)
    sem_inst_class_list = instance_problem.semantic_instance_class_list
    inst_id_list = instance_problem.instance_count_id_list
    ch_subplot_rc_arrangement = [(sem_val, inst_id) for sem_val, inst_id in zip(sem_inst_class_list, inst_id_list)]
    print('Writing heatmaps relative to each channel')
    for channel_idx in range(n_channels):
        h = plt.figure(0, figsize=FIGSIZE)
        plt.clf()
        hm = list_of_relative_heatmap_averages[channel_idx]
        list_of_subtitles = ['rel-to {}'.format(channel_names[rel_ch_idx])
                             for rel_ch_idx in range(n_channels)]
        display_pyutils.display_list_of_images([hm[rel_ch_idx, :, :]
                                                for rel_ch_idx in range(n_channels)],
                                               list_of_titles=list_of_subtitles, cmap=COLORMAP, arrange='custom',
                                               arrangement_rc_list=ch_subplot_rc_arrangement,
                                               smart_clims=True)
        filename = '{}_score_heatmaps_rel_by_ch_idx_{}.png'.format(
            split, channel_names[channel_idx])
        plt.suptitle(filename)
        display_pyutils.save_fig_to_workspace(filename, dpi=DPI)
        h.clear()


def write_relative_heatmaps_by_sem_cls(list_of_relative_heatmap_averages_rel_by_semantic, instance_problem, split):
    print('Summing and writing heatmaps relative to each semantic class')
    channel_names = instance_problem.get_channel_labels()
    sem_class_names = instance_problem.class_names
    sem_class_vals = range(instance_problem.n_semantic_classes)
    n_channels = len(list_of_relative_heatmap_averages_rel_by_semantic)
    n_semantic_classes = len(sem_class_names)
    for channel_idx in range(n_channels):
        h = plt.figure(0, figsize=FIGSIZE)
        plt.clf()
        hm = list_of_relative_heatmap_averages_rel_by_semantic[channel_idx]
        list_of_subtitles = ['rel-to {}'.format(sem_class_names[rel_sem_idx]) for rel_sem_idx in range(
            n_semantic_classes)]
        display_pyutils.display_list_of_images([hm[rel_sem_cls, :, :]
                                                for rel_sem_cls in range(len(sem_class_vals))],
                                               list_of_titles=list_of_subtitles, cmap=COLORMAP, arrange='rows',
                                               smart_clims=True)
        filename = '{}_score_heatmaps_rel_by_sem_cls_{}.png'.format(
            split, channel_names[channel_idx])
        plt.suptitle(filename)
        print('Writing image {}/{}'.format(channel_idx + 1, n_channels))
        display_pyutils.save_fig_to_workspace(filename, dpi=DPI)
        h.clear()


def compute_relative_per_sem_class_heatmaps(list_of_relative_heatmap_averages, instance_problem):
    n_semantic_classes = instance_problem.n_semantic_classes
    sem_class_vals = range(n_semantic_classes)
    sem_inst_class_list = instance_problem.semantic_instance_class_list

    channels_by_sem_val = [[c for c, s in enumerate(sem_inst_class_list) if s == sem_val] for sem_val in
                           sem_class_vals]

    # heatmaps relative to channels, summed by semantic class
    list_of_relative_heatmap_averages_rel_by_semantic = [torch.stack(
        [hm[chs, :, :].sum(dim=0) for chs in channels_by_sem_val], dim=0)
        for i, hm in enumerate(list_of_relative_heatmap_averages)]
    return list_of_relative_heatmap_averages_rel_by_semantic


def main():
    args = parse_args()
    logdir = args.logdir
    torchfcn.utils.scripts.set_random_seeds()
    display_pyutils.set_my_rc_defaults()
    cuda = torch.cuda.is_available()
    if display_pyutils.check_for_emptied_workspace():
        print('Workspace clean.')
    else:
        print('Workspace not clean, but running anyway.')

    # Load directory
    cfg, model_pth, out_dir, problem_config, model, my_trainer, optim, dataloaders = \
        torchfcn.utils.logs.load_everything_from_logdir(logdir, gpu=args.gpu, packed_as_dict=False)
    model.eval()

    # Write log directory name to folder
    with open(osp.join(display_pyutils.WORKSPACE_DIR, osp.basename(osp.normpath(logdir))), 'w') as fid:
        fid.write(logdir)

    for split in ['train', 'val']:
        # NOTE(allie): At the moment, clims is not synced.  Need to get the min/max and pass them in.

        # print('Getting absolute heatmaps')
        # absolute_heatmap_average = score_heatmaps.get_per_image_per_channel_heatmaps(
        #     model, dataloaders[split], cfg, cuda)
        # print('Writing images')
        # write_absolute_heatmaps(absolute_heatmap_average, split, my_trainer.instance_problem)

        print('Computing heatmaps relative to each channel')
        list_of_relative_heatmap_averages = score_heatmaps.get_relative_per_image_per_channel_heatmaps(
            model, dataloaders[split], cfg, cuda, my_trainer)
        print('Writing images')
        write_relative_heatmaps_by_channel([x.cpu().numpy() for x in list_of_relative_heatmap_averages],
                                           my_trainer.instance_problem, split)

        print('Computing heatmaps relative to each channel')
        list_of_relative_heatmap_averages_rel_by_semantic = compute_relative_per_sem_class_heatmaps(
            list_of_relative_heatmap_averages, my_trainer.instance_problem)
        write_relative_heatmaps_by_sem_cls([x.cpu().numpy() for x in list_of_relative_heatmap_averages_rel_by_semantic],
                                           my_trainer.instance_problem, split)


if __name__ == '__main__':
    main()
