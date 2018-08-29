import numpy as np
import torch
import tqdm
from torch.nn import functional as F

import instanceseg.utils.eval
from instanceseg.losses import iou
from instanceseg.utils import datasets, instance_utils


def get_center_min_max(h, dest_h, floor=False):
    if floor:
        pad_vertical = (dest_h - h) // 2 if h < dest_h else 0
    else:
        pad_vertical = (dest_h - h) / 2 if h < dest_h else 0
    return pad_vertical, (pad_vertical + h)


def get_per_channel_per_image_sizes_and_losses(model, dataloader, cuda, my_trainer):
    sem_inst_class_list = my_trainer.instance_problem.semantic_instance_class_list
    inst_id_list = my_trainer.instance_problem.instance_count_id_list
    n_channels = len(sem_inst_class_list)

    assigned_instance_sizes = np.zeros((len(dataloader), n_channels), dtype=int)
    losses_by_channel = np.zeros((len(dataloader), n_channels), dtype=float)
    ious_by_channel = np.zeros((len(dataloader), n_channels), dtype=float)
    soft_ious_by_channel = np.zeros((len(dataloader), n_channels), dtype=float)

    for idx, (x, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader),
                                                   desc='Matching channels to instances and computing sizes'):
        x, sem_lbl, inst_lbl = datasets.prep_inputs_for_scoring(x, sem_lbl, inst_lbl, cuda=cuda)
        assert x.size(0) == 1, NotImplementedError('Assuming batch size 1 at the moment')
        score = model(x)
        # softmax_scores = F.softmax(score, dim=1).data.cpu()
        # inst_lbl_pred = score.data.max(dim=1)[1].cpu()[:, :, :]
        pred_permutations, loss, component_loss = my_trainer.compute_loss(score, sem_lbl, inst_lbl)
        sem_lbl_np = sem_lbl.data.cpu().numpy()
        inst_lbl_np = inst_lbl.data.cpu().numpy()
        lt_combined = my_trainer.gt_tuple_to_combined(sem_lbl_np, inst_lbl_np)
        label_trues = lt_combined
        label_pred = score.data.max(dim=1)[1].cpu().numpy()[:, :, :]
        label_preds_permuted = instance_utils.permute_labels(label_pred, pred_permutations)
        scores_permuted = instance_utils.permute_scores(score, pred_permutations)
        confusion_matrix = instanceseg.utils.eval.calculate_confusion_matrix_from_arrays(
            label_preds_permuted, label_trues, n_channels)
        ious = instanceseg.utils.eval.calculate_iou(confusion_matrix)
        soft_iou, soft_iou_components = iou.lovasz_softmax_2d(F.softmax(scores_permuted, dim=1), sem_lbl, inst_lbl,
                                                              my_trainer.instance_problem.semantic_instance_class_list,
                                                              my_trainer.instance_problem.instance_count_id_list,
                                                              return_loss_components=True)

        data_idx = 0
        for channel_idx in range(n_channels):
            """
            We grab the ground truth location of the instance assigned to this channel
            """
            assigned_gt_idx = (pred_permutations[:, channel_idx]).item()
            sem_val, inst_val = sem_inst_class_list[assigned_gt_idx], inst_id_list[assigned_gt_idx]
            assigned_instance_sizes[idx, channel_idx] = datasets.get_instance_size(
                sem_lbl_np[data_idx, ...], sem_val, inst_lbl_np[data_idx, ...], inst_val)
            losses_by_channel[idx, channel_idx] = component_loss[:, assigned_gt_idx].view(-1,)
            ious_by_channel[idx, channel_idx] = ious[assigned_gt_idx]
            soft_ious_by_channel[idx, channel_idx] = soft_iou_components[assigned_gt_idx]

    return assigned_instance_sizes, losses_by_channel, ious_by_channel, soft_ious_by_channel


def get_per_image_per_channel_heatmaps(model, dataloader, cfg, cuda):
    if cfg['augment_semantic']:
        raise NotImplementedError('Gotta augment semantic first before running through model.')

    largest_image_shape = (0, 0)
    n_channels = None
    for idx, (x, _) in enumerate(dataloader):
        largest_image_shape = [max(largest_image_shape[0], x.size(2)), max(largest_image_shape[1], x.size(3))]
        if n_channels is None:
            x = datasets.prep_input_for_scoring(x, cuda=cuda)
            score = model(x)
            n_channels = score.size(1)

    heatmap_scores = torch.zeros(n_channels, *largest_image_shape)
    heatmap_counts = torch.zeros(n_channels, *largest_image_shape)

    for idx, (x, (sem_lbl, inst_lbl)) in enumerate(dataloader):
        x, sem_lbl, inst_lbl = datasets.prep_inputs_for_scoring(x, sem_lbl, inst_lbl, cuda=cuda)
        score = model(x)
        r1, r2 = get_center_min_max(x.size(2), heatmap_scores.size(1))
        c1, c2 = get_center_min_max(x.size(3), heatmap_scores.size(2))
        softmax_scores = F.softmax(score, dim=1).data.cpu()
        inst_lbl_pred = score.data.max(dim=1)[1].cpu()[:, :, :]
        # pred_permutations, losses = my_trainer.my_cross_entropy(score, sem_lbl, inst_lbl)
        # scores_permuted = instance_utils.permute_scores(score, pred_permutations)

        heatmap_scores[:, r1:r2, c1:c2] += softmax_scores
        heatmap_counts[:, r1:r2, c1:c2] += 1

    heatmap_average = heatmap_scores / heatmap_counts
    heatmap_average[heatmap_counts == 0] = 0
    return heatmap_average
