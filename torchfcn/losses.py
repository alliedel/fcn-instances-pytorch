import torch
import torch.nn.functional as F
import numpy as np
from ortools.graph import pywrapgraph

import local_pyutils
from torchfcn.datasets import dataset_utils

logger = local_pyutils.get_logger()

DEBUG_ASSERTS = True


# TODO(allie): dynamically choose multiplier and/or check to see if it gets us into a
# reasonable range (for converting cost matrix to ints)


# TODO(allie): Consider doing cost computation in batches to speed match costs up if that's a
# bottleneck (you can pass batches into the nll_loss function, and you can use reduce=False
# if you want the individual pixel components -- that'd be especially great for debugging)
# Get matching subsets (1 per semantic class)

# For each subset:
# Form a one-hot weight vector for the groundtruth-prediction index pair
# Pass the full score, target in with those two one-hot weight vectors. <-- On second
# thought, my weight vector may always be [0, 1] and the predictions will be


# TODO(allie): loss should be size averaged here somehow.  Gotta decide if it's overall or per
# class. -- Significantly affects the cost matrix and the contribution of the classes to the
# overall matching loss.


def invert_permutation(perm):
    return []


def cross_entropy2d(scores, target, semantic_instance_labels=None, matching=True, **kwargs):
    # Convert scores to predictions
    # log_p: (n, c, h, w)
    log_predictions = F.log_softmax(scores)

    if matching:
        assert semantic_instance_labels is not None, ValueError('semantic_instance_labels is '
                                                                'needed for matching.')
        pred_permutations, loss = cross_entropy2d_with_matching(log_predictions, target,
                                                                semantic_instance_labels, **kwargs)
        assert pred_permutations.shape[0] == 1, NotImplementedError
        # Somehow the gradient is no longer getting backpropped through loss, so I just recompute
        #  it here with the permutation I computed.
        loss = cross_entropy2d_without_matching(log_predictions[:, pred_permutations[0, :], :, :],
                                                target, **kwargs)
    else:
        pred_permutations = None
        loss = cross_entropy2d_without_matching(log_predictions, target, **kwargs)

    return pred_permutations, loss


def cross_entropy2d_with_matching(log_predictions, target, semantic_instance_labels,
                                  **kwargs):
    target_onehot = dataset_utils.labels_to_one_hot(target, len(semantic_instance_labels))
    # Allocate memory
    n, c = log_predictions.size()[0:2]
    all_prediction_indices = np.empty((n, c), dtype=int)
    all_pred_permutations = np.empty((n, c), dtype=int)
    all_costs = []  # dataset_utils.zeros_like(log_predictions, (n, c))

    # Compute optimal match & costs for each image in the batch
    for i in range(n):
        prediction_indices, pred_permutation, costs = \
            compute_optimal_match_loss(log_predictions[i, ...],
                                       target_onehot[i, ...],
                                       semantic_instance_labels)
        all_prediction_indices[i, ...] = prediction_indices
        all_pred_permutations[i, ...] = pred_permutation
        all_costs.append(torch.cat(costs))
    all_costs = torch.cat([c[torch.np.newaxis, :] for c in all_costs], dim=0).squeeze().float()
    loss_train = all_costs.sum()
    if DEBUG_ASSERTS:
        assert len(all_costs) == len(semantic_instance_labels)
    for inst_idx in range(log_predictions.size(1)):
        val = log_predictions[:, inst_idx, :, :].data.sum()
        # logger.info('sum(y_pred[:, {}, :, :]): {}'.format(inst_idx, val))
    return all_pred_permutations, loss_train


def cross_entropy2d_with_individual_terms_test(log_predictions, target,
                                               weight=None, size_average=True):
    target_onehot = dataset_utils.labels_to_one_hot(target, log_predictions.size(1))
    # input: (n, c, h, w), target: (n, h, w)
    loss = -torch.sum(target_onehot * log_predictions)
    if size_average:
        loss = loss / torch.sum(target_onehot[:, 1:, :, :])
    return loss


def cross_entropy2d_without_matching(log_predictions, target, weight=None, size_average=True):
    """
    Target should *not* be onehot.
    """
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = log_predictions.size()
    # log_p: (n*h*w, c)
    log_predictions = log_predictions.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_predictions = log_predictions[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_predictions = log_predictions.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_predictions, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def compute_optimal_match_loss(predictions, target_onehot, semantic_instance_labels):
    """
    target: C,H,W.  C is the number of instances for ALL semantic classes.
    predictions: C,H,W
    semantic_instance_labels: the mapping from ground truth index to semantic labels.  This is
    needed so we only allow instances in the same semantic class to compete.

    gt_indices, perm_permutation -- indices into (0, ..., C-1) for gt and predictions of the
     matches.
    costs -- cost of each of the matches (also length C)
    """
    assert len(semantic_instance_labels) == predictions.size(0) == target_onehot.size(0)
    gt_indices, pred_permutations, costs = [], [], []
    num_inst_classes = len(semantic_instance_labels)
    unique_labels = local_pyutils.unique(semantic_instance_labels)
    for label in unique_labels:
        idxs = [i for i in range(num_inst_classes) if (semantic_instance_labels[i] == label)]
        cost_list_2d = create_pytorch_cross_entropy_cost_matrix(predictions, target_onehot, idxs)
        cost_matrix, multiplier = convert_pytorch_costs_to_ints(cost_list_2d)

        assignment = pywrapgraph.LinearSumAssignment()

        for ground_truth in range(len(cost_matrix)):
            for prediction in range(len(cost_matrix[0])):
                assignment.AddArcWithCost(ground_truth, prediction,
                                          cost_matrix[ground_truth][prediction])
        check_status(assignment.Solve(), assignment)
        debug_print_assignments(assignment, multiplier)
        gt_indices += idxs
        pred_permutations += [idxs[assignment.RightMate(i)] for i in range(len(idxs))]
        costs += [cost_list_2d[i][assignment.RightMate(i)] for i in range(len(idxs))]
    sorted_indices = np.argsort(gt_indices)
    gt_indices = np.array(gt_indices)[sorted_indices]
    pred_permutations = np.array(pred_permutations)[sorted_indices]
    costs = [costs[i] for i in sorted_indices]
    return gt_indices, pred_permutations, costs


def nll2d_single_class_term(log_predictions_single_instance_cls, binary_target_single_instance_cls):
    lp = log_predictions_single_instance_cls
    bt = binary_target_single_instance_cls
    if DEBUG_ASSERTS:
        assert lp.size() == bt.size()
    try:
        res = -torch.sum(lp.view(-1,) * bt.view(-1,))
    except:
        import ipdb; ipdb.set_trace()
        raise
    return res


def create_pytorch_cross_entropy_cost_matrix(log_predictions, target_onehot, foreground_idxs):
    # predictions: C,H,W
    # target: C,H,W
    if DEBUG_ASSERTS:
        assert log_predictions.size() == target_onehot.size()
    normalizer = float(target_onehot.size(1) * target_onehot.size(2))
    cost_list_2d = [[nll2d_single_class_term(log_predictions[lp_cls, :, :],
                                             target_onehot[t_cls, :, :]) / normalizer
                     for t_idx, t_cls in enumerate(foreground_idxs)]
                    for lp_idx, lp_cls in enumerate(foreground_idxs)]
    # normalize by number of pixels
    # TODO(allie): Consider normalizing by number of pixels that actually have that class(?)

    return cost_list_2d


def convert_pytorch_costs_to_ints(cost_list_2d_variables, multiplier=None):
    infinity_cap = 1e12
    if multiplier is None:
        # Choose multiplier that keeps as many digits of precision as possible without creating
        # overflow errors
        absolute_max = float(0.0)
        for cl in cost_list_2d_variables:
            for c in cl:
                absolute_max = max(absolute_max, abs(c.data[0]))
        if absolute_max == 0:
            multiplier = 10 ** (10)
        else:
            multiplier = 1 if absolute_max > infinity_cap \
                else 10 ** (10 - int(np.log10(absolute_max)))

    num_classes = len(cost_list_2d_variables)
    cost_matrix_int = [[long(multiplier * cost_list_2d_variables[r][c].data[0]) for r in range(
        num_classes)]
                       for c in range(num_classes)]
    return cost_matrix_int, multiplier


def check_status(solve_status, assignment):
    if solve_status != assignment.OPTIMAL:
        if solve_status == assignment.INFEASIBLE:
            raise Exception('No assignment is possible.')
        elif solve_status == assignment.POSSIBLE_OVERFLOW:
            raise Exception('Some input costs are too large and may cause an integer overflow.')
        else:
            raise Exception('Unknown exception')


def debug_print_assignments(assignment, multiplier):
    logger.debug('Total cost = {}'.format(assignment.OptimalCost()))
    for i in range(0, assignment.NumNodes()):
        logger.debug('prediction %d assigned to ground truth %d.  Cost = %f' % (
            i, assignment.RightMate(i), float(assignment.AssignmentCost(i)) / multiplier))
