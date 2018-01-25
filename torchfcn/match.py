from ortools.graph import pywrapgraph
import numpy as np

import local_pyutils

# TODO(allie): dynamically choose multiplier and/or check to see if it gets us into a
# reasonable range (for converting cost matrix to ints)


DEBUG_ASSERTS = True

logger = local_pyutils.get_logger()


def create_pytorch_cross_entropy_cost_matrix(predictions, target, cost_fcn):
    # predictions: 1,C,H,W
    # target: 1,C,H,W  # TODO(allie): extend to N instead of 1
    sz1 = predictions.size()
    sz2 = target.size()
    assert (sz1[0], sz1[2], sz1[3]) == (sz2[0], sz2[1], sz2[2]), \
        'predictions.size() = {}; target.size() = {}'.format(
            predictions.size(), target.size())

    predictions = predictions.squeeze(dim=0)
    target = target.squeeze(dim=0)
    num_classes = predictions.size(0)
    cost_list_2d = [[cost_fcn(predictions[r, :, :], target[c, :, :]) for r in range(num_classes)]
                    for c in range(num_classes)]
    return cost_list_2d


def convert_pytorch_costs_to_ints(cost_list_2d_variables, multiplier=1e6):
    num_classes = len(cost_list_2d_variables)
    cost_matrix_int = [[long(multiplier * cost_list_2d_variables[r][c].data) for r in range(
        num_classes)]
                       for c in range(num_classes)]
    return cost_matrix_int, multiplier


def match_all_semantic_classes(predictions, target, cost_fcn, semantic_instance_labels,
                               target_in_int_form):
    """
    predictions, target: 1,C,H,W.  C is the number of instances for ALL semantic classes.
    semantic_instance_labels: the mapping from ground truth index to semantic labels.  This is needed so
    we only allow instances in the same semantic class to compete.

    predictions_indices, gt_permutation -- indices into (0, ..., C-1) for predictions and gt of the matches.
    costs -- cost of each of the matches (also length C)
    """
    if (not target_in_int_form and predictions.size(0) != target.size(0)) \
            or (not target_in_int_form and predictions.size()[1:] != target.size()) \
            or (target_in_int_form and (predictions.size() != target.size())):
        import ipdb;
        ipdb.set_trace()
        raise AssertionError(
            'predictions.size() = {}; target.size() = {}'.format(predictions.size(), target.size()))
    assert predictions.size(1) == len(semantic_instance_labels), \
        'predictions.size(1) = {}; ' \
        'len(semantic_instance_labels) = {}'.format(predictions.size(1),
                                                    len(semantic_instance_labels))
    unique_labels = np.unique(semantic_instance_labels)
    predictions_indices = []
    gt_permutation = []
    costs = []
    num_inst_classes = len(semantic_instance_labels)
    for label in unique_labels:
        idxs = [i for i in range(num_inst_classes) if semantic_instance_labels[i] == label]
        try:
            assignment, multiplier, cost_list_2d = match_single_semantic_class(
                predictions[idxs, :, :], target[target == idxs], cost_fcn)
        except Exception as ex:
            print(ex)
            import ipdb;
            ipdb.set_trace()
            raise
        predictions_indices += idxs
        gt_permutation += [idxs[assignment.RightMate(i)] for i in range(len(idxs))]
        costs += [cost_list_2d[i][assignment.RightMate(i)] for i in range(len(idxs))]
    sorted_indices = np.argsort(predictions_indices)
    predictions_indices = np.array(predictions_indices)[sorted_indices]
    gt_permutation = np.array(gt_permutation)[sorted_indices]
    costs = [costs[i] for i in sorted_indices]
    return predictions_indices, gt_permutation, costs


def match_single_semantic_class(predictions, target, cost_fcn):
    """
    predictions, target: N,C,H,W.  C is the number of classes for a SINGLE semantic class
    (e.g. - car 1, car 2).  This is important because we don't want to match car channels
    to person channels (at test time, we won't be able to identify the semantic class of a
    particular channel unless we do this division)
    """
    cost_list_2d = create_pytorch_cross_entropy_cost_matrix(predictions, target, cost_fcn)
    cost_matrix, multiplier = convert_pytorch_costs_to_ints(cost_list_2d)
    rows = len(cost_matrix)
    cols = len(cost_matrix[0])

    assignment = pywrapgraph.LinearSumAssignment()

    for prediction in range(rows):
        for ground_truth in range(cols):
            assignment.AddArcWithCost(prediction, ground_truth,
                                      cost_matrix[prediction][ground_truth])

    solve_status = assignment.Solve()
    if solve_status != assignment.OPTIMAL:
        if solve_status == assignment.INFEASIBLE:
            print('Algorithm failed with cost_matrix {}'.format(cost_matrix))
            raise Exception('No assignment is possible.')
        elif solve_status == assignment.POSSIBLE_OVERFLOW:
            raise Exception('Some input costs are too large and may cause an integer overflow.')
        else:
            raise Exception('Unknown exception')

    logger.debug('Total cost = {}'.format(assignment.OptimalCost()))
    for i in range(0, assignment.NumNodes()):
        logger.debug('prediction %d assigned to ground truth %d.  Cost = %f' % (
            i, assignment.RightMate(i), float(assignment.AssignmentCost(i)) / multiplier))

    return assignment, multiplier, cost_list_2d
