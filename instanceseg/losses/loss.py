import numpy as np
import torch
from torch.nn import functional as F

from instanceseg.losses import match
from instanceseg.losses import xentropy, iou
from instanceseg.losses.xentropy import DEBUG_ASSERTS
from instanceseg.models.model_utils import is_nan

loss_transformers_2d = {  # scores->predictions used directly by the loss function
    'cross_entropy': lambda x: F.log_softmax(x, dim=1),
    'soft_iou': lambda x: F.softmax(x, dim=1)
}

# lp=log_predictions_single_instance_cls,
# bt=binary_target_single_instance_cls
# pp=prediction_probs_single_instance_cls

single_class_component_loss_functions = {
    'cross_entropy': lambda lp, bt: xentropy.nll2d_single_class_term(lp, bt),
    'soft_iou': lambda pp, bt: iou.lovasz_softmax_2d_single_class_term(pp, bt)
}


def loss_2d_factory(loss_type, semantic_instance_labels, instance_id_labels, return_loss_components=False,
                    matching=True, **kwargs_to_pass_along):
    """
    loss_type: 'cross_entropy', 'soft_iou'
    """
    assert loss_type in loss_transformers_2d.keys() and loss_type in single_class_component_loss_functions.keys(), \
        NotImplementedError('We haven\'t implemented the loss type {}'.format(loss_type))

    def loss_fcn(scores, sem_lbl, inst_lbl):
        if DEBUG_ASSERTS:
            assert not is_nan(scores.data).sum(), 'Scores reached NaN'

        predictions = loss_transformers_2d[loss_type](scores)

        if matching:
            size_average = kwargs_to_pass_along.pop('size_average', True)
            assert not kwargs_to_pass_along, NotImplementedError('Unsure what to do with {}'.format(
                kwargs_to_pass_along.keys()))
            pred_permutations, total_loss, loss_components = matching_loss(
                single_class_component_loss_functions[loss_type], predictions, sem_lbl, inst_lbl,
                semantic_instance_labels, instance_id_labels, size_average=size_average)
        else:
            n, c = scores.size()[0:2]
            pred_permutations = np.empty((n, c), dtype=int)
            for i in range(n):
                pred_permutations[i, :] = range(c)
            if loss_type == 'cross_entropy':
                size_average = kwargs_to_pass_along.pop('size_average', True)
                ret = xentropy.cross_entropy2d_without_matching(predictions, sem_lbl, inst_lbl,
                                                                semantic_instance_labels, instance_id_labels,
                                                                return_loss_components=return_loss_components,
                                                                size_average=size_average,
                                                                **kwargs_to_pass_along)
            elif loss_type == 'soft_iou':
                size_average = kwargs_to_pass_along.pop('size_average', True)
                assert size_average is True, NotImplementedError('We didnt implement any normalization scheme here, '
                                                                 'but iou has size averaging essentially built in')
                ret = iou.lovasz_softmax_2d(predictions, sem_lbl, inst_lbl,
                                            semantic_instance_labels, instance_id_labels,
                                            return_loss_components=return_loss_components,
                                            **kwargs_to_pass_along)
            else:
                raise ValueError('I don\'t recognize the loss type {}'.format(loss_type))
            if return_loss_components:
                total_loss, loss_components = ret
            else:
                total_loss = ret
                loss_components = None
        if return_loss_components:
            assert loss_components is not None
            return pred_permutations, total_loss, loss_components

    return loss_fcn


def matching_loss(single_class_component_loss_fcn, predictions, sem_lbl, inst_lbl, semantic_instance_labels,
                  instance_id_labels, size_average=True):
    """
    :param single_class_component_loss_fcn: f(predictions, sem_lbl, inst_lbl)
    :param predictions: should be 'preprocessed' -- take softmax / log as needed for whatever form
    single_class_component_loss_fcn expects.
    :param instance_id_labels:
    :param size_average:
    :param semantic_instance_labels:
    :param inst_lbl:
    :param sem_lbl:

    note: returned loss components indexed by ground truth order
    """
    # Allocate memory
    n, c = predictions.size()[0:2]
    all_prediction_indices = np.empty((n, c), dtype=int)
    all_pred_permutations = np.empty((n, c), dtype=int)
    all_costs = []  # dataset_utils.zeros_like(log_predictions, (n, c))

    # Compute optimal match & costs for each image in the batch
    for i in range(n):
        prediction_indices, pred_permutation, costs = \
            match.compute_optimal_match_loss(single_class_component_loss_fcn, predictions[i, ...], sem_lbl[i, ...],
                                             inst_lbl[i, ...], semantic_instance_labels, instance_id_labels,
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
    return all_pred_permutations, loss_train, all_costs
