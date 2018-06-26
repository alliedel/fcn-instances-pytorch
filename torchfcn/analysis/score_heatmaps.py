import torch
from torch.nn import functional as F

from torchfcn.datasets import dataset_utils


def get_center_min_max(h, dest_h, floor=False):
    if floor:
        pad_vertical = (dest_h - h) // 2 if h < dest_h else 0
    else:
        pad_vertical = (dest_h - h) / 2 if h < dest_h else 0
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
    heatmap_img_shape_rc = [2 * dim_sz + 2 for dim_sz in largest_image_shape]
    list_of_heatmap_scores = [torch.zeros(n_channels, *heatmap_img_shape_rc)
                              for _ in range(n_channels)]  # indexed by relative channel
    list_of_heatmap_counts = [torch.zeros(n_channels, *heatmap_img_shape_rc)
                              for _ in range(n_channels)]  # indexed by relative channel
    sem_inst_class_list = my_trainer.instance_problem.semantic_instance_class_list
    inst_id_list = my_trainer.instance_problem.instance_count_id_list

    for idx, (x, (sem_lbl, inst_lbl)) in enumerate(dataloader):
        x, sem_lbl, inst_lbl = dataset_utils.prep_inputs_for_scoring(x, sem_lbl, inst_lbl, cuda=cuda)
        score = model(x)
        cropped_r1, cropped_r2 = get_center_min_max(x.size(2), heatmap_img_shape_rc[0], floor=True)
        cropped_c1, cropped_c2 = get_center_min_max(x.size(3), heatmap_img_shape_rc[1], floor=True)
        softmax_scores = F.softmax(score, dim=1).data.cpu()
        inst_lbl_pred = score.data.max(dim=1)[1].cpu()[:, :, :]
        pred_permutations, loss = my_trainer.my_cross_entropy(score, sem_lbl, inst_lbl)
        # scores_permuted = instance_utils.permute_scores(score, pred_permutations)

        assert x.size(0) == 1, NotImplementedError('Assuming batch size 1 at the moment')
        data_idx = 0
        for channel_idx_to_be_relative_to in range(n_channels):
            """
            We grab the ground truth location of the instance assigned to this channel
            """
            gt_idx = (pred_permutations[:, channel_idx_to_be_relative_to]).item()
            sem_val, inst_val = sem_inst_class_list[gt_idx], inst_id_list[gt_idx]
            gt_location_mask = (sem_lbl[data_idx, ...] == sem_val).int() * (inst_lbl[data_idx, ...] == inst_val).int()
            if gt_location_mask.sum().data.cpu().numpy() == 0:
                # ground truth instance non-existent
                continue
            else:
                com_rc = dataset_utils.compute_centroid_binary_mask(gt_location_mask.data.cpu().numpy())
                image_center = dataset_utils.get_image_center(gt_location_mask.size(), floor=False)
                com_offset_rc = [int(c - img_c) for c, img_c in zip(com_rc, image_center)]
            try:
                loc_rc_in_heatmap_image = cropped_r1 - com_offset_rc[0], cropped_c1 - com_offset_rc[1]
                r1, r2 = loc_rc_in_heatmap_image[0], loc_rc_in_heatmap_image[0] + x.size(2)
                c1, c2 = loc_rc_in_heatmap_image[1], loc_rc_in_heatmap_image[1] + x.size(3)
                for c in range(n_channels):
                    list_of_heatmap_scores[c][channel_idx_to_be_relative_to, r1:r2, c1:c2] += \
                        softmax_scores[data_idx, c, ...]
                    list_of_heatmap_counts[c][channel_idx_to_be_relative_to, r1:r2, c1:c2] += 1
            except Exception as exc:
                import ipdb; ipdb.set_trace()
                raise

    list_of_heatmap_averages = [heatmap_scores / heatmap_counts for heatmap_scores, heatmap_counts in zip(
        list_of_heatmap_scores, list_of_heatmap_counts)]
    for heatmap, heatmap_counts in zip(list_of_heatmap_averages, list_of_heatmap_counts):
        heatmap[heatmap_counts == 0] = 0

    return list_of_heatmap_averages


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
