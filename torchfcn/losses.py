import local_pyutils
import numpy as np
import torch
import torch.nn.functional as F
from ortools.graph import pywrapgraph
from torchfcn.models.model_utils import is_nan, any_nan

logger = local_pyutils.get_logger()

DEBUG_ASSERTS = True

# TODO(allie): Enable exporting of cost matrices through tensorboard as images
# TODO(allie): Compute other losses ('mixing', 'wrong identity', 'poor shape') along with some
# image stats like between-instance distance
# TODO(allie): Figure out why nll_loss doesn't match my implementation with sum of individual terms

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


# TODO(allie): Investigate loss 'size averaging'.  Gotta decide if should be overall or per
# class. -- Significantly affects the cost matrix and the contribution of the classes to the
# overall matching loss.
# TODO(allie): Do something with inst_lbl == 0 for instance classes.

def cross_entropy2d(scores, sem_lbl, inst_lbl, semantic_instance_labels, instance_id_labels, matching=True,
                    break_here=False, recompute_optimal_loss=False, return_loss_components=False, **kwargs):
    # Convert scores to predictions
    # log_p: (n, c, h, w)
    if break_here:
        import ipdb;
        ipdb.set_trace()
    log_predictions = F.log_softmax(scores, dim=1)

    if DEBUG_ASSERTS:
        try:
            assert not is_nan(scores.data).sum()
        except:
            import ipdb; ipdb.set_trace()
            raise Exception('scores reached nan')

        try:
            assert not is_nan(log_predictions.data).sum()
        except:
            import ipdb; ipdb.set_trace()
            raise Exception('log_predictions reached nan')

    if matching:
        ret = cross_entropy2d_with_matching(log_predictions, sem_lbl, inst_lbl, semantic_instance_labels,
                                            instance_id_labels, return_loss_components=return_loss_components, **kwargs)
        if return_loss_components:
            pred_permutations, loss, loss_components = ret
        else:
            pred_permutations, loss = ret
        if DEBUG_ASSERTS or recompute_optimal_loss:
            ret = cross_entropy2d_without_matching(
                log_predictions[:, pred_permutations[0, :], :, :], sem_lbl, inst_lbl,
                semantic_instance_labels, instance_id_labels, return_loss_components=return_loss_components, **kwargs)
            if return_loss_components:
                loss_recomputed, loss_components = ret
            else:
                loss_recomputed = ret
            if recompute_optimal_loss:
                loss = loss_recomputed
    else:
        n, c = log_predictions.size()[0:2]
        pred_permutations = np.empty((n, c), dtype=int)
        for i in range(n):
            pred_permutations[i, :] = range(c)
        ret = cross_entropy2d_without_matching(log_predictions, sem_lbl, inst_lbl,
                                               semantic_instance_labels,
                                               instance_id_labels,
                                               return_loss_components=return_loss_components,
                                               **kwargs)
        if return_loss_components:
            loss, loss_components = ret
        else:
            loss = ret
    if return_loss_components:
        return pred_permutations, loss, loss_components
    else:
        return pred_permutations, loss


def tensors_are_close(tensor_a, tensor_b):
    return torch.np.allclose(tensor_a.cpu().numpy(), tensor_b.cpu().numpy())


def cross_entropy2d_with_matching(log_predictions, sem_lbl, inst_lbl, semantic_instance_labels,
                                  instance_id_labels, size_average=True, return_loss_components=False):
    # Allocate memory
    n, c = log_predictions.size()[0:2]
    all_prediction_indices = np.empty((n, c), dtype=int)
    all_pred_permutations = np.empty((n, c), dtype=int)
    all_costs = []  # dataset_utils.zeros_like(log_predictions, (n, c))

    # Compute optimal match & costs for each image in the batch
    for i in range(n):
        prediction_indices, pred_permutation, costs = \
            compute_optimal_match_loss(log_predictions[i, ...],
                                       sem_lbl[i, ...], inst_lbl[i, ...],
                                       semantic_instance_labels, instance_id_labels,
                                       size_average=size_average)
        all_prediction_indices[i, ...] = prediction_indices
        all_pred_permutations[i, ...] = pred_permutation
        all_costs.append(torch.cat(costs))
    all_costs = torch.cat([c[torch.np.newaxis, :] for c in all_costs], dim=0).float()
    loss_train = all_costs.sum()
    if DEBUG_ASSERTS:
        if all_costs.size(1) != len(semantic_instance_labels):
            import ipdb;
            ipdb.set_trace()
            raise Exception
    if return_loss_components:
        return all_pred_permutations, loss_train, all_costs
    else:
        return all_pred_permutations, loss_train


def cross_entropy2d_without_matching(log_predictions, sem_lbl, inst_lbl, semantic_instance_labels,
                                     instance_id_labels, weight=None, size_average=True, return_loss_components=False):
    """
    Target should *not* be onehot.
    log_predictions: NxCxHxW
    sem_lbl, inst_lbl: NxHxW
    """
    assert sem_lbl.size() == inst_lbl.size()
    assert (log_predictions.size(0), log_predictions.size(2), log_predictions.size(3)) == \
           sem_lbl.size()
    assert weight is None, NotImplementedError
    losses = []
    for sem_inst_idx, (sem_val, inst_val) in enumerate(zip(semantic_instance_labels, instance_id_labels)):
        try:
            binary_target_single_instance_cls = ((sem_lbl == sem_val) * (inst_lbl == inst_val)).float()
        except:
            import ipdb;
            ipdb.set_trace()
            raise Exception
        log_predictions_single_instance_cls = log_predictions[:, sem_inst_idx, :, :]
        losses.append(nll2d_single_class_term(log_predictions_single_instance_cls,
                                              binary_target_single_instance_cls))
    losses = torch.cat([c[torch.np.newaxis, :] for c in losses], dim=0).float()
    loss = sum(losses)
    if size_average:
        normalizer = (inst_lbl >= 0).data.float().sum()
        loss /= normalizer
        losses /= normalizer

    if return_loss_components:
        return loss, losses
    else:
        return loss


def compute_optimal_match_loss(log_predictions, sem_lbl, inst_lbl, semantic_instance_labels, instance_id_labels,
                               size_average=True):
    """
    target: C,H,W.  C is the number of instances for ALL semantic classes.
    predictions: C,H,W
    semantic_instance_labels: the mapping from ground truth index to semantic labels.  This is
    needed so we only allow instances in the same semantic class to compete.
    gt_indices, perm_permutation -- indices into (0, ..., C-1) for gt and predictions of the
     matches.
    costs -- cost of each of the matches (also length C)
    """
    assert len(semantic_instance_labels) == log_predictions.size(0)
    gt_indices, pred_permutations, costs = [], [], []
    num_inst_classes = len(semantic_instance_labels)
    unique_semantic_values = local_pyutils.unique(semantic_instance_labels)
    for sem_val in unique_semantic_values:
        idxs = [i for i in range(num_inst_classes) if (semantic_instance_labels[i] == sem_val)]
        cost_list_2d = create_pytorch_cross_entropy_cost_matrix(log_predictions, sem_lbl, inst_lbl,
                                                                semantic_instance_labels,
                                                                instance_id_labels,
                                                                sem_val, size_average=size_average)
        cost_matrix, multiplier = convert_pytorch_costs_to_ints(cost_list_2d)
        assignment = pywrapgraph.LinearSumAssignment()

        for ground_truth in range(len(cost_matrix)):
            for prediction in range(len(cost_matrix[0])):
                try:
                    assignment.AddArcWithCost(ground_truth, prediction,
                                              cost_matrix[prediction][ground_truth])
                except:
                    print(cost_matrix[prediction][ground_truth])
                    import ipdb; ipdb.set_trace()
                    raise
        check_status(assignment.Solve(), assignment)
        debug_print_assignments(assignment, multiplier)
        gt_indices += idxs
        pred_permutations += [idxs[assignment.RightMate(i)] for i in range(len(idxs))]
        costs += [cost_list_2d[assignment.RightMate(i)][i] for i in range(len(idxs))]
    sorted_indices = np.argsort(gt_indices)
    gt_indices = np.array(gt_indices)[sorted_indices]
    pred_permutations = np.array(pred_permutations)[sorted_indices]
    costs = [costs[i] for i in sorted_indices]
    return gt_indices, pred_permutations, costs


def nll2d_single_class_term(log_predictions_single_instance_cls, binary_target_single_instance_cls):
    lp = log_predictions_single_instance_cls
    bt = binary_target_single_instance_cls
    if DEBUG_ASSERTS:
        try:
            assert lp.size() == bt.size()
        except:
            import ipdb;
            ipdb.set_trace()
            raise
    try:
        res = -torch.sum(lp.view(-1, ) * bt.view(-1, ))
    except:
        import ipdb;
        ipdb.set_trace()
        raise
    return res


def create_pytorch_cross_entropy_cost_matrix(log_predictions, sem_lbl, inst_lbl,
                                             semantic_instance_labels, instance_id_labels,
                                             sem_val, size_average=True):
    # cost_list_2d[prediction][ground_truth]
    # predictions: C,H,W
    # target: (sem_lbl, inst_lbl): (H,W, H,W)
    if DEBUG_ASSERTS:
        try:
            assert inst_lbl.size() == sem_lbl.size()
            assert log_predictions.size()[1:] == inst_lbl.size()
        except:
            import ipdb;
            ipdb.set_trace()
            raise
    if size_average:
        # TODO(allie): Verify this is correct (and not sem_lbl >=0, or some combo)
        normalizer = (inst_lbl >= 0).data.sum()
    else:
        normalizer = 1
    sem_inst_idxs_for_this_class = [i for i, sem_inst_val in enumerate(semantic_instance_labels) if
                                    sem_inst_val == sem_val]
    inst_id_lbls_for_this_class = [instance_id_labels[i] for i in sem_inst_idxs_for_this_class]

    # TODO(allie): allow for more target (gt) idxs than the number of lp idxs.
    cost_list_2d = [[nll2d_single_class_term(log_predictions[sem_inst_idx, :, :],
                                             (sem_lbl == sem_val).float() *
                                             (inst_lbl == inst_val).float())
                     / normalizer for inst_val in inst_id_lbls_for_this_class]
                    for sem_inst_idx in sem_inst_idxs_for_this_class]
    # TODO(allie): Consider normalizing by number of pixels that actually have that class(?)
    if DEBUG_ASSERTS:
        try:
            assert all([not any_nan(cost_list_1d[j].data)
                        for cost_list_1d in cost_list_2d for j in range(len(cost_list_1d))])
        except:
            import ipdb; ipdb.set_trace()
            raise Exception('costs reached nan')
    return cost_list_2d


def convert_pytorch_costs_to_ints(cost_list_2d_variables, multiplier=None):
    infinity_cap = 1e15
    log_infinity_cap = np.log10(infinity_cap)
    if multiplier is None:
        # Choose multiplier that keeps as many digits of precision as possible without creating
        # overflow errors
        absolute_max = float(0.0)
        for cl in cost_list_2d_variables:
            for c in cl:
                absolute_max = max(absolute_max, abs(c.data[0]))
        if absolute_max == 0:
            multiplier = 1
            # multiplier = 10 ** 10
        else:
            multiplier = 10 ** (log_infinity_cap - int(np.log10(absolute_max)))

    num_classes = len(cost_list_2d_variables)
    cost_matrix_int = [[int(multiplier * cost_list_2d_variables[r_pred][c_gt].data[0])
                        for c_gt in range(num_classes)]
                       for r_pred in range(num_classes)]
    if DEBUG_ASSERTS:
        try:
            assert all([not is_nan(cost_list_1d[j])
                        for cost_list_1d in cost_matrix_int for j in range(len(cost_list_1d))])
        except:
            import ipdb; ipdb.set_trace()
            raise Exception('costs in cost_matrix_int reached nan')
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
        logger.debug('ground truth %d assigned to prediction %d.  Cost = %f' % (
            i, assignment.RightMate(i), float(assignment.AssignmentCost(i)) / multiplier))


def compute_instance_mixing_metric(scores):
    """
    Computes 'mixing' metric as the ratio of the assigned
    """
