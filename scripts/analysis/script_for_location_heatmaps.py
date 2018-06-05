import argparse

import display_pyutils
import torch
import torch.nn.functional as F

from torchfcn import script_utils
from torchfcn.datasets import dataset_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='directory that contains the checkpoint and config', type=str, required=True)
    parser.add_argument('-gpu', help='directory that contains the checkpoint and config', type=int, default=0)
    args = parser.parse_args()
    return args


def get_center_min_max(h, dest_h):
    pad_vertical = (dest_h - h) // 2 if h < dest_h else 0
    return pad_vertical, (pad_vertical + h)


def get_per_image_per_channel_heatmaps(model, dataloader, cfg, cuda):
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

    heatmap_scores = torch.zeros(n_channels, *largest_image_shape)
    heatmap_counts = torch.zeros(n_channels, *largest_image_shape)

    for idx, (x, (sem_lbl, inst_lbl)) in enumerate(dataloader):
        x, sem_lbl, inst_lbl = dataset_utils.prep_inputs_for_scoring(x, sem_lbl, inst_lbl, cuda=cuda)
        score = model(x)
        r1, r2 = get_center_min_max(x.size(2), heatmap_scores.size(1))
        c1, c2 = get_center_min_max(x.size(3), heatmap_scores.size(2))
        softmax_scores = F.softmax(score, dim=1).data.cpu()
        inst_lbl_pred = score.data.max(dim=1)[1].cpu()[:, :, :]
        # pred_permutations, loss = my_trainer.my_cross_entropy(score, sem_lbl, inst_lbl)
        # scores_permuted = instance_utils.permute_scores(score, pred_permutations)

        heatmap_scores[:, r1:r2, c1:c2] += softmax_scores
        heatmap_counts[:, r1:r2, c1:c2] += 1

    heatmap_average = heatmap_scores / heatmap_counts
    heatmap_average[heatmap_counts == 0] = 0
    return heatmap_average


def main():
    args = parse_args()
    logdir = args.logdir
    cfg, model_pth, out_dir, problem_config, model, my_trainer, optim, dataloaders = \
        script_utils.load_everything_from_logdir(logdir, gpu=args.gpu, packed_as_dict=False)
    cuda = torch.cuda.is_available()
    initial_model, start_epoch, start_iteration = script_utils.get_model(cfg, problem_config,
                                                                         checkpoint_file=None, semantic_init=None,
                                                                         cuda=cuda)
    for split in ['train', 'val']:
        heatmap_average = get_per_image_per_channel_heatmaps(model, dataloaders[split], cfg, cuda)
        try:
            heatmap_image_list_3d = heatmap_average.numpy().transpose(1, 2, 0)
            display_pyutils.imwrite_list_as_3d_array_to_workspace(
                heatmap_image_list_3d, filename_base_ext='{}_score_heatmaps_imsave.png'.format(split))
            display_pyutils.matshow_and_save_3d_array_to_workspace(
                heatmap_image_list_3d, filename_base_ext='{}_score_heatmaps_matshow_clim_synced.png'.format(split))
            display_pyutils.matshow_and_save_3d_array_to_workspace(
                heatmap_image_list_3d, sync_clims=False,
                filename_base_ext='{}_score_heatmaps_clim_unsynced_per_image.png'.format(split))
        except Exception as exc:
            import ipdb;
            ipdb.set_trace()
            raise


if __name__ == '__main__':
    main()
