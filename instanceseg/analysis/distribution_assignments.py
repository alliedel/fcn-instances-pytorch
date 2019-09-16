import numpy as np
import torch
import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

import instanceseg.utils.eval
from instanceseg.losses import iou
from instanceseg.utils import datasets, instance_utils


def get_center_min_max(h, dest_h, floor=False):
    if floor:
        pad_vertical = (dest_h - h) // 2 if h < dest_h else 0
    else:
        pad_vertical = (dest_h - h) / 2 if h < dest_h else 0
    return pad_vertical, (pad_vertical + h)


def get_per_channel_per_image_sizes_and_losses(model, dataloader: DataLoader, cuda, my_trainer):
    sem_inst_class_list = my_trainer.instance_problem.model_channel_semantic_ids
    n_channels = len(sem_inst_class_list)
    n_images = len(dataloader.dataset)
    assigned_instance_sizes = np.zeros((n_images, n_channels), dtype=int)
    losses_by_channel = np.zeros((n_images, n_channels), dtype=float)
    ious_by_channel = np.zeros((n_images, n_channels), dtype=float)
    soft_ious_by_channel = np.zeros((n_images, n_channels), dtype=float)
    image_idx = 0
    for batch_idx, (x, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader),
                                                         desc='Matching channels to instances and computing sizes'):
        x, sem_lbl, inst_lbl = datasets.prep_inputs_for_scoring(x, sem_lbl, inst_lbl, cuda=cuda)
        assert x.size(0) == 1, NotImplementedError('Assuming batch size 1 at the moment')
        scores = model(x)
        # softmax_scores = F.softmax(score, dim=1).data.cpu()
        # inst_lbl_pred = score.data.max(dim=1)[1].cpu()[:, :, :]
        assignments, loss, component_loss = my_trainer.compute_loss(scores, sem_lbl, inst_lbl)
        sem_lbl_np = sem_lbl.data.cpu().numpy()
        inst_lbl_np = inst_lbl.data.cpu().numpy()
        label_pred = scores.data.max(dim=1)[1].cpu().numpy()[:, :, :]

        for data_idx in range(x.size(0)):
            losses_by_channel[image_idx, :] = component_loss[data_idx, :]
            channel_sem_vals = assignments.sem_vals[data_idx, :]
            channel_inst_vals = assignments.assigned_gt_inst_vals[data_idx, :]
            pred_sem, pred_inst = instance_utils.decompose_semantic_and_instance_labels(label_pred[data_idx, ...],
                                                                                        channel_inst_vals,
                                                                                        channel_sem_vals)
            gt_combined_channelwise = \
                instance_utils.label_tuple_to_channel_ids(sem_lbl_np[data_idx, ...], inst_lbl_np[data_idx, ...],
                                                          channel_semantic_values=channel_sem_vals,
                                                          channel_instance_values=channel_inst_vals)
            pred_big_channelwise = \
                instance_utils.label_tuple_to_channel_ids(sem_lbl_np, inst_lbl_np,
                                                          channel_semantic_values=channel_sem_vals,
                                                          channel_instance_values=channel_inst_vals)

            confusion_matrix = instanceseg.utils.eval.calculate_confusion_matrix_from_arrays(
                pred_big_channelwise, gt_combined_channelwise, n_channels)
            ious = instanceseg.utils.eval.calculate_iou(confusion_matrix)
            soft_iou, soft_iou_components = iou.lovasz_softmax_2d(
                F.softmax(scores, dim=1), sem_lbl, inst_lbl, sem_vals_by_channel=channel_sem_vals,
                gt_instance_vals_by_channel=channel_inst_vals, return_loss_components=True)
            soft_ious_by_channel[image_idx, :] = soft_iou_components[data_idx, :]
            ious_by_channel[image_idx, :] = ious
            for i, channel_idx in enumerate(assignments.model_channels.shape[0]):
                assert assignments.model_channels[data_idx, i] == channel_idx
                # channel_idx == i at earliest implementation
                assigned_instance_sizes[image_idx, channel_idx] = datasets.get_instance_size(
                    sem_lbl_np[data_idx, ...], assignments.sem_vals[i], inst_lbl_np[data_idx, ...],
                    assignments.assigned_gt_inst_vals[data_idx, i])

            image_idx += 1

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

        heatmap_scores[:, r1:r2, c1:c2] += softmax_scores
        heatmap_counts[:, r1:r2, c1:c2] += 1

    heatmap_average = heatmap_scores / heatmap_counts
    heatmap_average[heatmap_counts == 0] = 0
    return heatmap_average
