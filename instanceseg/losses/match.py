import numpy as np
from ortools.graph import pywrapgraph

from instanceseg.losses.xentropy import DEBUG_ASSERTS
from instanceseg.models.model_utils import any_nan, is_nan
from instanceseg.utils.misc import get_logger, unique

logger = get_logger()


# TODO(allie): Test different normalization schemes
# TODO(allie): Allow for more target (gt) channels than prediction channels


def compute_optimal_match_loss(single_class_component_loss_fcn, log_predictions, sem_lbl, inst_lbl,
                               semantic_instance_labels, instance_id_labels, size_average=True):
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
    unique_semantic_values = unique(semantic_instance_labels)
    for sem_val in unique_semantic_values:
        idxs = [i for i in range(num_inst_classes) if (semantic_instance_labels[i] == sem_val)]
        cost_list_2d = create_pytorch_cost_matrix(single_class_component_loss_fcn, log_predictions,
                                                  sem_lbl, inst_lbl, semantic_instance_labels,
                                                  instance_id_labels, sem_val, size_average=size_average)
        cost_matrix, multiplier = convert_pytorch_costs_to_ints(cost_list_2d)
        assignment = pywrapgraph.LinearSumAssignment()

        for ground_truth in range(len(cost_matrix)):
            for prediction in range(len(cost_matrix[0])):
                try:
                    assignment.AddArcWithCost(ground_truth, prediction,
                                              cost_matrix[prediction][ground_truth])
                except:
                    print(cost_matrix[prediction][ground_truth])
                    import ipdb;
                    ipdb.set_trace()
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


def create_pytorch_cost_matrix(single_class_component_loss_fcn, log_predictions, sem_lbl, inst_lbl,
                               semantic_instance_labels, instance_id_labels, sem_val, size_average=True):
    """

    :param single_class_component_loss_fcn: f(yhat, binary_y) where yhat, binary_2d_gt are (N, H, W)
    :param log_predictions: C,H,W
    :param sem_lbl: (H,W)
    :param inst_lbl: (H,W)
    :param semantic_instance_labels:
    :param instance_id_labels:
    :param sem_val:
    :param size_average:
    :return:
        cost_list_2d[prediction][ground_truth]
    """

    if DEBUG_ASSERTS:
        assert inst_lbl.size() == sem_lbl.size()
        assert log_predictions.size()[1:] == inst_lbl.size()
    if size_average:
        # TODO(allie): Verify this is correct (and not sem_lbl >=0, or some combo)
        normalizer = (inst_lbl >= 0).data.sum()
    else:
        normalizer = 1
    sem_inst_idxs_for_this_class = [i for i, sem_inst_val in enumerate(semantic_instance_labels) if
                                    sem_inst_val == sem_val]
    inst_id_lbls_for_this_class = [instance_id_labels[i] for i in sem_inst_idxs_for_this_class]

    if normalizer == 0:
        print(Warning('WARNING: image contained all void class.  Setting error to 0 for all channels.'))
        cost_list_2d = [[0 for inst_val in inst_id_lbls_for_this_class
                         for sem_inst_idx in sem_inst_idxs_for_this_class]]
    else:
        cost_list_2d = [[
            single_class_component_loss_fcn(
                log_predictions[sem_inst_idx, :, :],
                (sem_lbl == sem_val).float() * (inst_lbl == inst_val).float()) / normalizer
            for inst_val in inst_id_lbls_for_this_class]
            for sem_inst_idx in sem_inst_idxs_for_this_class]
    if DEBUG_ASSERTS:
        try:
            assert all([not any_nan(cost_list_1d[j].data)
                        for cost_list_1d in cost_list_2d for j in range(len(cost_list_1d))])
        except:
            import ipdb;
            ipdb.set_trace()
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
            import ipdb;
            ipdb.set_trace()
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
