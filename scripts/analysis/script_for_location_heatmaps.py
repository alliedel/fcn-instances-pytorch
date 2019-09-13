import argparse
import os.path as osp
import numpy as np

import matplotlib.pyplot as plt
import torch

import instanceseg.utils.display as display_pyutils
import instanceseg.utils.logs
import instanceseg.utils.script_setup
from instanceseg.analysis import score_heatmaps

FIGSIZE = (10, 10)
DPI = 300


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='directory that contains the checkpoint and config', type=str, required=True)
    parser.add_argument('--gpu', '-g', help='gpu identifier (int)', type=int, default=0)
    parser.add_argument('--relative', help='should export relative positions?', type=bool, default=True)
    parser.add_argument('--absolute', help='should export absolute positions?', type=bool, default=True)
    args = parser.parse_args()
    return args


COLORMAP = 'cubehelix'  # 'cubehelix', 'inferno', 'magma', 'viridis', 'gray'


def write_absolute_heatmaps(absolute_heatmap_average, split, instance_problem):
    channel_names = instance_problem.get_channel_labels()
    is_background = ['background' in s for s in channel_names]
    non_bground_channel_idxs = [i for i, is_b in enumerate(is_background) if not is_b]

    for channel_idx in non_bground_channel_idxs:
        h = plt.figure(0, figsize=FIGSIZE)
        plt.clf()
        if is_background[channel_idx]:
            continue
        image = absolute_heatmap_average[channel_idx, :, :]
        filename = '{}_absolute_heatmap_{}.png'.format(split, channel_names[channel_idx])
        display_pyutils.display_list_of_images([image],
                                               cmap=COLORMAP, smart_clims=True, arrange='rows',
                                               list_of_titles=['{}, absolute score'.format(channel_names[channel_idx])])
        # filename_base_ext='{}_score_heatmaps_{}.png'
        plt.suptitle(filename)
        display_pyutils.save_fig_to_workspace(filename, dpi=DPI)
        h.clear()


def write_relative_heatmaps_by_channel(list_of_relative_heatmap_averages, instance_problem, split):
    channel_names = instance_problem.get_channel_labels()
    sem_inst_class_list = instance_problem.model_channel_semantic_ids
    is_background = ['background' in s for s in channel_names]
    non_bground_channel_idxs = [i for i, is_b in enumerate(is_background) if not is_b]
    inst_id_list = instance_problem.instance_count_id_list
    non_bg_sem_classes = list(np.unique([s for i, s in enumerate(sem_inst_class_list) if not is_background[i]]))
    inst_starts_at_1 = 1 - int(any([inst_id_list[i] == 0 for i, is_b in enumerate(is_background) if not is_b]))
    # ch_subplot_rc_arrangement = [(non_bg_sem_classes.index(sem_val), inst_id - inst_starts_at_1)
    ch_subplot_rc_arrangement = [(inst_id - inst_starts_at_1, non_bg_sem_classes.index(sem_val))
                                 for sem_val, inst_id, bg in
                                 zip(sem_inst_class_list, inst_id_list, is_background) if not bg]
    # R, C = max([arr[0] for arr in ch_subplot_rc_arrangement]), max([arr[1] for arr in ch_subplot_rc_arrangement])

    print('Writing heatmaps relative to each channel')
    for channel_idx in non_bground_channel_idxs:
        h = plt.figure(0, figsize=FIGSIZE)
        plt.clf()
        hm = list_of_relative_heatmap_averages[channel_idx]
        list_of_subtitles = ['{} score,\n centered on {} GT'.format(channel_names[channel_idx], channel_names[
            rel_ch_idx])
                             for rel_ch_idx in non_bground_channel_idxs]
        display_pyutils.display_list_of_images([hm[rel_ch_idx, :, :]
                                                for rel_ch_idx in non_bground_channel_idxs],
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
    sem_class_names = instance_problem.semantic_class_names
    n_channels = len(list_of_relative_heatmap_averages_rel_by_semantic)
    is_background = ['background' in s for s in channel_names]
    non_bg_sem_class_idxs = list(np.unique([i for i, s in enumerate(sem_class_names) if 'background' not in s]))
    non_bg_sem_classes = list(np.unique([s for i, s in enumerate(sem_class_names) if 'background' not in s]))
    non_bground_channel_idxs = [i for i, is_b in enumerate(is_background) if not is_b]
    for channel_idx in non_bground_channel_idxs:
        h = plt.figure(0, figsize=FIGSIZE)
        plt.clf()
        hm = list_of_relative_heatmap_averages_rel_by_semantic[channel_idx]
        list_of_subtitles = ['{} score,\n centered on {} GT'.format(channel_names[channel_idx], sem_class_name)
                             for rel_sem_idx, sem_class_name in enumerate(non_bg_sem_classes)]
        display_pyutils.display_list_of_images([hm[rel_sem_cls, :, :]
                                                for rel_sem_cls in non_bg_sem_class_idxs],
                                               list_of_titles=list_of_subtitles, cmap=COLORMAP, arrange='rows',
                                               smart_clims=True)
        filename = '{}_score_heatmaps_rel_by_sem_cls_{}.png'.format(split, channel_names[channel_idx])
        plt.suptitle(filename)
        print('Writing image {}/{}'.format(channel_idx + 1, n_channels))
        display_pyutils.save_fig_to_workspace(filename, dpi=DPI)
        h.clear()


def compute_relative_per_sem_class_heatmaps(list_of_relative_heatmap_averages, instance_problem):
    n_semantic_classes = instance_problem.n_semantic_classes
    sem_class_vals = range(n_semantic_classes)
    sem_inst_class_list = instance_problem.model_channel_semantic_ids

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
    instanceseg.utils.script_setup.set_random_seeds()
    display_pyutils.set_my_rc_defaults()
    cuda = torch.cuda.is_available()
    if display_pyutils.check_for_emptied_workspace():
        print('Workspace clean.')
    else:
        print('Workspace not clean, but running anyway.')

    # Load directory
    cfg, model_pth, out_dir, problem_config, model, my_trainer, optim, dataloaders = \
        instanceseg.utils.logs.load_everything_from_logdir(logdir, gpu=args.gpu, packed_as_dict=False)
    model.eval()

    # Write log directory name to folder
    with open(osp.join(display_pyutils.WORKSPACE_DIR, osp.basename(osp.normpath(logdir))), 'w') as fid:
        fid.write(logdir)

    for split in ['train', 'val']:
        # NOTE(allie): At the moment, clims is not synced.  Need to get the min/max and pass them in.

        print('Getting absolute heatmaps')
        absolute_heatmap_average = score_heatmaps.get_per_image_per_channel_heatmaps(
            model, dataloaders[split], cfg, cuda)
        print('Writing images')
        write_absolute_heatmaps(absolute_heatmap_average.cpu().numpy(), split, my_trainer.instance_problem)

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
