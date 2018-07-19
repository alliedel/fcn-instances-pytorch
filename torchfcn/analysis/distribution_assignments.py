import torch
from torch.nn import functional as F
import numpy as np
import tqdm

from torchfcn.datasets import dataset_utils


def get_center_min_max(h, dest_h, floor=False):
    if floor:
        pad_vertical = (dest_h - h) // 2 if h < dest_h else 0
    else:
        pad_vertical = (dest_h - h) / 2 if h < dest_h else 0
    return pad_vertical, (pad_vertical + h)


def get_per_channel_per_image_sizes(model, dataloader, cuda, my_trainer):

    sem_inst_class_list = my_trainer.instance_problem.semantic_instance_class_list
    inst_id_list = my_trainer.instance_problem.instance_count_id_list
    n_channels = len(sem_inst_class_list)

    assigned_instance_sizes = np.zeros((len(dataloader), n_channels), dtype=int)

    for idx, (x, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader),
                                                   desc='Matching channels to instances and computing sizes'):
        x, sem_lbl, inst_lbl = dataset_utils.prep_inputs_for_scoring(x, sem_lbl, inst_lbl, cuda=cuda)
        assert x.size(0) == 1, NotImplementedError('Assuming batch size 1 at the moment')
        score = model(x)
        # softmax_scores = F.softmax(score, dim=1).data.cpu()
        # inst_lbl_pred = score.data.max(dim=1)[1].cpu()[:, :, :]
        pred_permutations, loss = my_trainer.my_cross_entropy(score, sem_lbl, inst_lbl)
        # scores_permuted = instance_utils.permute_scores(score, pred_permutations)
        sem_lbl_np = sem_lbl.data.cpu().numpy()
        inst_lbl_np = inst_lbl.data.cpu().numpy()

        data_idx = 0
        for channel_idx in range(n_channels):
            """
            We grab the ground truth location of the instance assigned to this channel
            """
            assigned_gt_idx = (pred_permutations[:, channel_idx]).item()
            sem_val, inst_val = sem_inst_class_list[assigned_gt_idx], inst_id_list[assigned_gt_idx]
            assigned_instance_sizes[idx, channel_idx] = dataset_utils.get_instance_size(
                sem_lbl_np[data_idx, ...], sem_val, inst_lbl_np[data_idx, ...], inst_val)
    return assigned_instance_sizes


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
