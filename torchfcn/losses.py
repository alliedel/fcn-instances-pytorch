import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import local_pyutils
from torchfcn import match

logger = local_pyutils.get_logger()

DEBUG_ASSERTS = True


raise NotImplementedError('Continue here')
# You've gotta test whether you can pass a 1x1xHxW tensor in -- does it accept single class
# things?  Otherwise, we'll just make it binary and 'unweight' the 'background' class we form
# from "all other classes".  We may be able to avoid this when the number of instances per that
# semantic class is larger than 1.


def cross_entropy2d(score, target, semantic_instance_labels, matching=True, size_average=False):
    # TODO(allie): Consider doing cost computation in batches to speed this up if that's a
    # bottleneck (you can pass batches into the nll_loss function, and you can use reduce=False
    # if you want the individual pixel components -- that'd be especially great for debugging)
    # Get matching subsets (1 per semantic class)

    # For each subset:
    # Form a one-hot weight vector for the groundtruth-prediction index pair
    # Pass the full score, target in with those two one-hot weight vectors. <-- On second
    # thought, my weight vector may always be [0, 1] and the predictions will be

    return









def nll2d_single_class_term(log_predictions, target, which_class):
    """
    size_average is False -- we always return the sum.  The reason: if it's the background class,
    we may want to weight it differently.
    Quick explanation: if you did sum([nll2d_single_class_term(lp, y, cls) for cls in
    range(n_classes)]), you'd get the same as nll(lp, y, cls)
    """
    n_classes = log_predictions.size(1)
    # noinspection PyArgumentList
    weight = torch.FloatTensor([1 if i == which_class else 0 for i in range(n_classes)])
    loss = F.nll_loss(log_predictions, target, weight=weight, size_average=False)
    return loss


def cross_entropy2d(scores, target, semantic_instance_labels=None, matching=True, **kwargs):
    # Convert scores to predictions
    # log_p: (n, c, h, w)
    log_predictions = F.log_softmax(scores)

    if matching:
        loss, gt_permutations = cross_entropy2d_with_matching(log_predictions, target,
                                                              semantic_instance_labels, **kwargs)
    else:
        gt_permutations = None
        loss = cross_entropy2d_without_matching(log_predictions, target, **kwargs)
    return gt_permutations, loss


def cross_entropy2d_with_matching(log_predictions, target, semantic_instance_labels,
                                  **kwargs):
    loss_fcn = lambda yhat, y: cross_entropy2d_without_matching(yhat, y, **kwargs)
    n = log_predictions.size(0)
    c = log_predictions.size(1)
    all_prediction_indices = np.empty((n, c), dtype=int)
    all_gt_permutation = np.empty((n, c), dtype=int)
    all_costs = Variable(torch.DoubleTensor(n, c))
    for i in range(n):
        prediction_i = log_predictions[i, ...]
        target_i = target[i, ...]
        prediction_indices, gt_permutation, costs = \
            match.match_all_semantic_classes(prediction_i, target_i, loss_fcn,
                                             semantic_instance_labels)
        all_prediction_indices[i, ...] = prediction_indices
        all_gt_permutation[i, ...] = gt_permutation
        all_costs[i, ...] = torch.cat(costs)

    all_costs = all_costs.squeeze()
    loss_train = all_costs.sum()
    if DEBUG_ASSERTS:
        assert len(all_costs) == len(semantic_instance_labels)
    for inst_idx in range(log_predictions.size(1)):
        val = log_predictions[:, inst_idx, :, :].data.sum()
        logger.info('sum(y_pred[:, {}, :, :]): {}'.format(inst_idx, val))
    return loss_train, all_gt_permutation


def cross_entropy2d_without_matching(log_predictions, target, weight=None, size_average=True):
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
