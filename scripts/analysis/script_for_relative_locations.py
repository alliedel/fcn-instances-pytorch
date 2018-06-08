import argparse

import display_pyutils
import numpy as np
import torch
import torch.nn.functional as F

from torchfcn import script_utils
from torchfcn.datasets import dataset_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='directory that contains the checkpoint and config', type=str, required=True)
    parser.add_argument('--gpu', help='gpu identifier (int)', type=int, default=0)
    args = parser.parse_args()
    return args


def get_center_min_max(h, dest_h):
    pad_vertical = (dest_h - h) // 2 if h < dest_h else 0
    return pad_vertical, (pad_vertical + h)


def get_relative_per_image_per_channel_heatmaps(model, dataloader, cfg, cuda, my_trainer):
    if cfg['augment_semantic']:
        raise NotImplementedError('Gotta augment semantic first before running through model.')

    largest_image_shape = (0, 0)
    n_channels = None
    for idx, (x, _) in enumerate(dataloader):
        largest_image_shape = [max(largest_image_shape[0], x.size(2)), max(largest_image_shape[1], x.size(3))]
        if n_channels is None:
            x = dataset_utils.prep_input_for_scoring(x, cuda=cuda)
            score = model(x)
            n_channels = score.size(1)
    heatmap_img_shape_rc = [2 * dim_sz for dim_sz in largest_image_shape]
    list_of_heatmap_scores = [torch.zeros(n_channels, *heatmap_img_shape_rc)
                              for _ in range(n_channels)]  # indexed by relative channel
    list_of_heatmap_counts = [torch.zeros(n_channels, *heatmap_img_shape_rc)
                              for _ in range(n_channels)]  # indexed by relative channel
    sem_inst_class_list = model.problem_config.semantic_instance_class_list
    inst_id_list = model.problem_config.instance_count_id_list
    for idx, (x, (sem_lbl, inst_lbl)) in enumerate(dataloader):
        x, sem_lbl, inst_lbl = dataset_utils.prep_inputs_for_scoring(x, sem_lbl, inst_lbl, cuda=cuda)
        score = model(x)
        r1, r2 = get_center_min_max(x.size(2), heatmap_img_shape_rc[0])
        c1, c2 = get_center_min_max(x.size(3), heatmap_img_shape_rc[1])
        softmax_scores = F.softmax(score, dim=1).data.cpu()
        inst_lbl_pred = score.data.max(dim=1)[1].cpu()[:, :, :]
        pred_permutations, loss = my_trainer.my_cross_entropy(score, sem_lbl, inst_lbl)
        # scores_permuted = instance_utils.permute_scores(score, pred_permutations)

        for channel_idx_to_be_relative_to in range(n_channels):
            """
            We grab the ground truth location of the instance assigned to this channel
            """
            gt_idx = pred_permutations[channel_idx_to_be_relative_to]
            sem_val, inst_val = sem_inst_class_list[gt_idx], inst_id_list[gt_idx]
            com_rc = dataset_utils.compute_centroid_binary_mask(
                np.logical_and(sem_lbl == sem_val, inst_lbl == inst_val))
            loc_rc_in_heatmap_image = com_rc[0] + r1, com_rc[1] + c1

            list_of_heatmap_scores[channel_idx_to_be_relative_to][:, r1:r2, c1:c2] += softmax_scores
            list_of_heatmap_counts[channel_idx_to_be_relative_to][:, r1:r2, c1:c2] += 1
            raise NotImplementedError
            heatmaps_shifted.append(heatmap_shifted)


    heatmap_average = heatmap_scores / heatmap_counts
    heatmap_average[heatmap_counts == 0] = 0
    return list_of_relative_heatmaps


def main():
    args = parse_args()
    logdir = args.logdir
    cfg, model_pth, out_dir, problem_config, model, my_trainer, optim, dataloaders = \
        script_utils.load_everything_from_logdir(logdir, gpu=args.gpu, packed_as_dict=False)
    cuda = torch.cuda.is_available()

    for split in ['train', 'val']:
        heatmap_average = get_relative_per_image_per_channel_heatmaps(model, dataloaders[split], cfg, cuda, my_trainer)
        try:
            heatmap_image_list_3d = heatmap_average.numpy().transpose(1, 2, 0)
            display_pyutils.imwrite_list_as_3d_array_to_workspace(
                heatmap_image_list_3d, filename_base_ext='{}_score_relative_heatmaps_imsave.png'.format(split))
            display_pyutils.matshow_and_save_3d_array_to_workspace(
                heatmap_image_list_3d,
                filename_base_ext='{}_score_relative_heatmaps_matshow_clim_synced.png'.format(split),
                show_filenames_as_titles=True)
            display_pyutils.matshow_and_save_3d_array_to_workspace(
                heatmap_image_list_3d, sync_clims=False,
                filename_base_ext='{}_score_relative_heatmaps_clim_unsynced_per_image.png'.format(split),
                show_filenames_as_titles=True)
        except Exception as exc:
            import ipdb; ipdb.set_trace()
            raise


if __name__ == '__main__':
    main()
