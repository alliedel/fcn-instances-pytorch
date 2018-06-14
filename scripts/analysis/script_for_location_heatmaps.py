import argparse

import display_pyutils
import matplotlib.pyplot as plt
import torch

from torchfcn import script_utils
from torchfcn.analysis import score_heatmaps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='directory that contains the checkpoint and config', type=str, required=True)
    parser.add_argument('--gpu', help='gpu identifier (int)', type=int, default=0)
    parser.add_argument('--relative', help='should export relative positions?', type=bool, default=True)
    parser.add_argument('--absolute', help='should export absolute positions?', type=bool, default=True)
    args = parser.parse_args()
    return args


COLORMAP = 'cubehelix'  # 'cubehelix', 'inferno', 'magma', 'viridis', 'gray'


def main():
    args = parse_args()
    logdir = args.logdir

    cuda = torch.cuda.is_available()

    cfg, model_pth, out_dir, problem_config, model, my_trainer, optim, dataloaders = \
        script_utils.load_everything_from_logdir(logdir, gpu=args.gpu, packed_as_dict=False)

    sem_inst_class_list = my_trainer.instance_problem.semantic_instance_class_list
    inst_id_list = my_trainer.instance_problem.instance_count_id_list
    channel_names = my_trainer.instance_problem.get_channel_labels()

    for split in ['train', 'val']:
        # NOTE(allie): At the moment, clims is not synced.  Need to get the min/max and pass them in.

        # Write absolute positioning heatmaps
        print('Getting absolute heatmaps')
        absolute_heatmap_average = score_heatmaps.get_per_image_per_channel_heatmaps(
            model, dataloaders[split], cfg, cuda)
        print('Writing images')
        for channel_idx, (sem_idx, inst_id) in enumerate(zip(sem_inst_class_list, inst_id_list)):
            channel_name = channel_names[channel_idx]
            display_pyutils.matshow_and_save_list_to_workspace(
                [absolute_heatmap_average[channel_idx, :, :]],
                filename_base_ext='{}_score_heatmaps_ch{}_{}.png'.format(split, channel_idx, channel_name),
                show_filenames_as_titles=True, cmap=COLORMAP)

        # Write relative positioning heatmaps
        print('Getting heatmaps relative to semantic class')
        list_of_relative_heatmap_averages = score_heatmaps.get_relative_per_image_per_channel_heatmaps(
            model, dataloaders[split], cfg, cuda, my_trainer)
        print('Writing images')
        n_semantic_classes = my_trainer.instance_problem.n_semantic_classes
        sem_class_vals = range(n_semantic_classes)
        sem_class_names = my_trainer.instance_problem.class_names

        channels_by_sem_val = [[c for c, s in enumerate(sem_inst_class_list) if s == sem_val] for sem_val in
                               sem_class_vals]

        list_of_relative_heatmap_averages_rel_by_semantic = [torch.stack(
            [hm[chs, :, :].sum(dim=0) for chs in channels_by_sem_val], dim=0)
            for i, hm in enumerate(list_of_relative_heatmap_averages)]

        n_channels = len(list_of_relative_heatmap_averages)

        # heatmaps relative to channels, summed by semantic class
        print('Getting heatmaps relative to each channel')
        ch_subplot_rc_arrangement = [(sem_val, inst_id) for sem_val, inst_id in zip(sem_inst_class_list, inst_id_list)]
        for channel_idx in range(n_channels):
            h = plt.figure(0, figsize=(20,20))
            hm = list_of_relative_heatmap_averages_rel_by_semantic[channel_idx]
            list_of_subtitles = ['rel-to {}'.format(sem_class_names[rel_sem_idx]) for rel_sem_idx in range(
                n_semantic_classes)]
            display_pyutils.display_list_of_images([hm[rel_sem_cls, :, :]
                                                    for rel_sem_cls in range(len(sem_class_vals))],
                                                   list_of_titles=list_of_subtitles, cmap=COLORMAP, arrange='rows')
            filename = '{}_score_heatmaps_rel_by_sem_cls_{}.png'.format(
                split, channel_names[channel_idx])
            plt.suptitle(filename)
            print('Writing image {}/{}'.format(channel_idx + 1, n_channels))
            display_pyutils.save_fig_to_workspace(filename)
            h.clear()

        # heatmaps relative to channels, individual.
        for channel_idx in range(n_channels):
            h = plt.figure(0)
            hm = list_of_relative_heatmap_averages[channel_idx]
            list_of_subtitles = ['rel-to {}: {}'.format(rel_ch_idx, channel_names[rel_ch_idx])
                                 for rel_ch_idx in range(n_channels)]
            display_pyutils.display_list_of_images([hm[rel_ch_idx, :, :]
                                                    for rel_ch_idx in range(n_channels)],
                                                   list_of_titles=list_of_subtitles, cmap=COLORMAP, arrange='custom',
                                                   arrangement_rc_list=ch_subplot_rc_arrangement)
            filename = '{}_score_heatmaps_rel_by_ch_idx_{}.png'.format(
                split, channel_names[channel_idx])
            plt.suptitle(filename)
            display_pyutils.save_fig_to_workspace(filename)
            h.clear()


if __name__ == '__main__':
    main()
