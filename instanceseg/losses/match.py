import numpy as np
from scipy import optimize

from instanceseg.losses.xentropy import DEBUG_ASSERTS
from instanceseg.models.model_utils import is_nan
from instanceseg.utils.misc import get_logger

try:
    import torch
except:
    torch = None

logger = get_logger(name='match', file='/tmp/debug_match.txt')


GT_VALUE_FOR_FALSE_POSITIVE = -2

# TODO(allie): Test different normalization schemes
# TODO(allie): Allow for more target (gt) channels than prediction channels


def solve_matching_problem(cost_tensor: torch.Tensor):
    """
    Returns matching assignment, sorted by row index.
    """
    if torch is not None:
        assert type(cost_tensor) is np.ndarray or torch.is_tensor(cost_tensor)
    else:
        assert type(cost_tensor) is np.ndarray
    cost_tensor_for_assignment = cost_tensor.detach() if cost_tensor.requires_grad else cost_tensor
    row_ind, col_ind = optimize.linear_sum_assignment(cost_tensor_for_assignment)
    ind_idxs_sorted_by_row = np.argsort(row_ind)
    col_ind = [col_ind[idx] for idx in ind_idxs_sorted_by_row]
    return col_ind


def create_pytorch_cost_matrix(single_class_component_loss_fcn, predictions, sem_lbl, inst_lbl,
                               model_channel_semantic_ids, sem_val, size_average=True):
    """

    :param single_class_component_loss_fcn: f(yhat, binary_y) where yhat, binary_2d_gt are (N, H, W)
    :param predictions: C,H,W
    :param sem_lbl: (H,W)
    :param inst_lbl: (H,W)
    :param model_channel_semantic_ids:
    :param sem_val:
    :param size_average:
    :return:
        cost_arr[prediction, ground_truth]
        model_channels_for_this_cls: channel indices for predictions corresponding to each row in the cost matrix
        gt_inst_vals_present: ground truth instance values corresponding to each col in the cost matrix


        ** Note: We have to fill out the false positive loss as well -- this is important for matching, because it
        may not be the same for every prediction (if it's IoU, it will be (1), but it could be proportional to the
        number of predicted pixels, for instance. -- this means that we must have at least as many columns (# gt) as
        rows (# pred) -- the last columns will be reserved for false positives, and we'll be able to identify them as
        the columns where the groundtruth value is GT_VALUE_FOR_FALSE_POSITIVE
    """

    if DEBUG_ASSERTS:
        assert inst_lbl.size() == sem_lbl.size()
        assert predictions.size()[1:] == inst_lbl.size()
    if size_average:
        # TODO(allie): Verify this is correct (and not sem_lbl >=0, or some combo)
        normalizer = (inst_lbl >= 0).data.sum()
    else:
        normalizer = 1
    model_channels_for_this_cls = [i for i, sem_inst_val in enumerate(model_channel_semantic_ids)
                                   if sem_inst_val == sem_val]
    # inst_id_lbls_for_this_class = [instance_id_labels[i] for i in sem_inst_idxs_for_this_class]
    gt_inst_vals_present = sorted([x.detach().item() for x in torch.unique(inst_lbl[sem_lbl == sem_val])])
        # can speed up by using range(torch.max())
    n_pred = len(model_channels_for_this_cls)
    n_gt = len(gt_inst_vals_present)  # unique takes a long time..
    if n_gt < n_pred:
        n_false_extra_predictions = n_pred - n_gt
        inst_value_not_in_gt = GT_VALUE_FOR_FALSE_POSITIVE  # -2?
        assert inst_value_not_in_gt not in gt_inst_vals_present
        gt_inst_vals_present.extend([GT_VALUE_FOR_FALSE_POSITIVE for _ in range(n_false_extra_predictions)])
        n_gt = n_pred
    cost_tensor = torch.empty((n_pred, n_gt))

    if normalizer == 0:
        cost_tensor[:, :] = normalizer.detach()
        print(Warning('WARNING: image contained all void class. Setting error to 0 for all channels.'))
    else:
        for r, model_channel in enumerate(model_channels_for_this_cls):
            for c, gt_inst_val in enumerate(gt_inst_vals_present):
                cost_tensor[r, c] = single_class_component_loss_fcn(predictions[model_channel, :, :],
                                            (sem_lbl == sem_val).float() * (inst_lbl == gt_inst_val).float()) \
                                    / normalizer

    if DEBUG_ASSERTS:
        try:
            assert not torch.any(torch.isnan(cost_tensor))
        except AssertionError as ae:
            import ipdb;
            ipdb.set_trace()
            raise Exception('costs reached nan in cost_list_2d')
    return cost_tensor, model_channels_for_this_cls, gt_inst_vals_present


def convert_pytorch_costs_to_ints(cost_list_2d_variables, multiplier=None, infinity_cap=1e15):
    log_infinity_cap = np.log10(infinity_cap)
    if multiplier is None:
        # Choose multiplier that keeps as many digits of precision as possible without creating
        # overflow errors
        absolute_max = float(0.0)
        for cl in cost_list_2d_variables:
            for c in cl:
                absolute_max = max(absolute_max, abs(c.item()))
        if absolute_max == 0:
            multiplier = 1
            # multiplier = 10 ** 10
        else:
            multiplier = 10 ** (log_infinity_cap - int(np.log10(absolute_max)))

    cost_matrix_int = [[int(multiplier * pred_gt.item()) for pred_gt in pred] for pred in cost_list_2d_variables]
    # cost_matrix_int = [[int(multiplier * cost_list_2d_variables[r_pred][c_gt].item())
    #                     for c_gt in range(num_classes)]
    #                    for r_pred in range(num_classes)]
    if DEBUG_ASSERTS:
        try:
            assert all([not is_nan(cost_list_1d[j])
                        for cost_list_1d in cost_matrix_int for j in range(len(cost_list_1d))])
        except:
            import ipdb;
            ipdb.set_trace()
            raise Exception('costs in cost_matrix_int reached nan')
    return cost_matrix_int, multiplier

