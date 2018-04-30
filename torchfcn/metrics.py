
def compute_analytics(self, sem_label, inst_label, label_preds, pred_scores, pred_permutations):
    if type(pred_scores) is list:
        try:
            pred_scores_stacked = torch.cat(pred_scores, dim=0)
            abs_scores_stacked = torch.abs(pred_scores_stacked)
        except:
            pred_scores_stacked = np.concatenate(pred_scores, axis=0)
            abs_scores_stacked = np.abs(pred_scores_stacked)
    else:
        pred_scores_stacked = pred_scores
        try:  # if tensor
            abs_scores_stacked = torch.abs(pred_scores_stacked)
        except:
            abs_scores_stacked = np.abs(pred_scores_stacked)
    try:
        softmax_scores = F.softmax(pred_scores_stacked, dim=1)
    except:
        softmax_scores = F.softmax(torch.from_numpy(pred_scores_stacked), dim=1)
    analytics = {
        'scores': {
            'max': pred_scores_stacked.max(),
            # 'mean': pred_scores_stacked.mean(),
            # 'median': pred_scores_stacked.median(),
            'min': pred_scores_stacked.min(),
            # 'abs_mean': abs_scores_stacked.mean(),
            'abs_median': abs_scores_stacked.median(),
        },
    }
    for channel in range(pred_scores_stacked.size(1)):
        channel_scores = pred_scores_stacked[:, channel, :, :]
        abs_channel_scores = abs_scores_stacked[:, channel, :, :]
        analytics['per_channel_scores/{}'.format(channel)] = {
            'max': channel_scores.max(),
            # 'mean': channel_scores.mean(),
            # 'median': channel_scores.median(),
            'min': channel_scores.min(),
            # 'abs_mean': abs_channel_scores.mean(),
            'abs_median': abs_channel_scores.median(),
        }
    channel_labels = self.instance_problem.get_channel_labels('{}_{}')
    for inst_idx, sem_cls in enumerate(self.instance_problem.semantic_instance_class_list):
        same_sem_cls_channels = [channel for channel, cls in enumerate(
            self.instance_problem.semantic_instance_class_list) if cls == sem_cls]
        all_maxes = softmax_scores[:, same_sem_cls_channels, :, :].max(dim=1)[0]
        all_sums = softmax_scores[:, same_sem_cls_channels, :, :].sum(dim=1)
        # get 'smearing' level across all channels
        sem_channel_keyname = 'instance_commitment/{}'.format(self.instance_problem.class_names[sem_cls])
        if sem_channel_keyname not in analytics.keys():
            # first of its sem class -- run semantic analytics here
            relevant_pixels = sem_label == sem_cls
            import ipdb; ipdb.set_trace()
            analytics[sem_channel_keyname] = \
                (all_maxes[relevant_pixels] / all_sums[relevant_pixels]).mean(),

        # get 'commitment' level within each channel
        pixels_assigned_to_me = (label_preds == inst_idx)
        pixels_assigned_to_my_sem_cls = torch.sum([(label_preds == channel) for channel in
                                                   same_sem_cls_channels])
        # channel_usage_fraction: fraction of channels of same semantic class that are used
        inst_channel_keyname = 'channel_usage_fraction/{}'.format(channel_labels[inst_idx])
        analytics[inst_channel_keyname] = pixels_assigned_to_me.int().sum() / \
                                          pixels_assigned_to_my_sem_cls.int().sum()

    return analytics

