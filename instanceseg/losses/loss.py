import numpy as np
import torch
from torch.nn import functional as F

from instanceseg.losses import match
from instanceseg.losses import xentropy, iou
from instanceseg.losses.xentropy import DEBUG_ASSERTS


# TODO(allie): Implement test: Compare component loss function with full loss function when matching is off


def get_subclasses(cls):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in get_subclasses(c)])

# matching_component_loss_registry = {loss_class.loss_type: loss_class
#                                     for loss_class in get_subclasses(ComponentMatchingLossBase)}


def loss_object_factory(loss_type, semantic_instance_class_list, instance_id_count_list, matching, size_average):

    if loss_type == 'cross_entropy':
        loss_object = CrossEntropyComponentMatchingLoss(semantic_instance_class_list, instance_id_count_list, matching,
                                                        size_average)
    elif loss_type == 'soft_iou':
        loss_object = SoftIOUComponentMatchingLoss(semantic_instance_class_list, instance_id_count_list, matching, size_average)
    else:
        raise NotImplementedError

    return loss_object


# MATCHING_LOSS_CLASS_REGISTRY = get_matching_component_loss_registry()

class ComponentLossAbstractInterface(object):
    """
    An agreed upon interface -- the minimum requirements for creating a loss function that works with our trainer.
    """
    def loss_fcn(self, scores, sem_lbl, inst_lbl):
        """
        inputs:
         scores: NxCxHxW
         sem_lbl: NxHxW
         inst_lbl: NxHxW

        return:
         pred_permutations: NxC  (pred_permutations[i, :] = range(C) without matching)
         total_loss: scalar Variable: avg of loss_components
         loss_components: NxC
        """
        raise NotImplementedError


class ComponentMatchingLossBase(ComponentLossAbstractInterface):
    """
    Base class for matching loss functions -- allows us to take any 'normal' component loss and make a specialized
    matching loss object out of it.
    """
    loss_type = None

    def __init__(self, semantic_instance_labels=None, instance_id_labels=None, matching=True, size_average=True):
        if matching:
            assert semantic_instance_labels is not None and instance_id_labels is not None, ValueError(
                'We need semantic and instance ids to perform matching')
        self.matching = matching
        self.semantic_instance_labels = semantic_instance_labels
        self.instance_id_labels = instance_id_labels
        self.size_average = size_average
        self.only_present = True
        if self.loss_type is None:
            raise NotImplementedError('Loss type should be defined in subclass of {}'.format(__class__))

    def transform_scores_to_predictions(self, scores):
        """
       'Preprocessing' step: converts scores to, e.g. - log probabilities.

        :param scores: any arbitrarily scaled output of the CNN
        :return predictions: a transformation of scores that matches the properties needed for the loss
        (e.g. - probabilities that add to 1 along channels, etc.)
        """
        raise NotImplementedError

    def component_loss(self, single_channel_prediction, binary_target):
        raise NotImplementedError

    def compute_matching_loss(self, predictions, sem_lbl, inst_lbl):
        """
        Note: predictions should be 'preprocessed' -- take softmax / log as needed for whatever form
            single_class_component_loss_fcn expects.
        Note: returned loss components indexed by ground truth order
        """

        # Allocate memory
        batch_sz, n_channels = predictions.size(0), predictions.size(1)
        all_gt_indices = np.empty((batch_sz, n_channels), dtype=int)
        all_pred_permutations = np.empty((batch_sz, n_channels), dtype=int)
        all_costs = []  # dataset_utils.zeros_like(log_predictions, (n, c))

        # Compute optimal match & costs for each image in the batch
        for i in range(batch_sz):
            gt_indices, pred_permutation, costs = \
                self._compute_optimal_match_loss_single_img(predictions[i, ...], sem_lbl[i, ...],
                                                            inst_lbl[i, ...])
            all_gt_indices[i, ...] = gt_indices
            all_pred_permutations[i, ...] = pred_permutation
            try:
                all_costs.append(torch.stack(costs))
            except:
                import ipdb; ipdb.set_trace()
                raise
        all_costs = torch.cat([c[None, :] for c in all_costs], dim=0).float()
        loss_train = all_costs.sum()
        if DEBUG_ASSERTS:
            if all_costs.size(1) != len(self.semantic_instance_labels):
                import ipdb;
                ipdb.set_trace()
                raise Exception
        pred_permutations, total_loss, loss_components = all_pred_permutations, loss_train, all_costs
        return pred_permutations, total_loss, loss_components

    def get_binary_gt_for_channel(self, sem_lbl, inst_lbl, channel_idx):
        sem_val, inst_val = self.semantic_instance_labels[channel_idx], self.instance_id_labels[channel_idx]
        return ((sem_lbl == sem_val) * (inst_lbl == inst_val)).float()

    def compute_nonmatching_loss(self, predictions, sem_lbl, inst_lbl):
        batch_sz, n_channels = predictions.size(0), predictions.size(1)
        pred_permutations = np.empty((batch_sz, n_channels), dtype=int)
        for i in range(batch_sz):
            pred_permutations[i, :] = range(n_channels)

        losses = []
        component_losses = []
        n_channels = len(self.semantic_instance_labels)
        for channel_idx in range(n_channels):
            binary_target_single_instance_cls = self.get_binary_gt_for_channel(sem_lbl, inst_lbl, channel_idx).view(-1)
            predictions_single_instance_cls = predictions[:, channel_idx, :, :].view(-1)
            new_loss = self.component_loss(binary_target_single_instance_cls, predictions_single_instance_cls)
            losses.append(new_loss)
            if self.only_present and binary_target_single_instance_cls.sum() == 0:
                component_losses.append(0)
                continue
            component_losses.append(new_loss)  # TODO(allie): Fix if we move away from batch size 1

        losses = torch.cat([c[None, :] for c in losses], dim=0).float()
        loss = sum(losses)
        if self.size_average:
            normalizer = (inst_lbl >= 0).data.float().sum()
            loss /= normalizer
            losses /= normalizer

        return pred_permutations, iou.mean(losses), component_losses

    def loss_fcn(self, scores, sem_lbl, inst_lbl):
        predictions = self.transform_scores_to_predictions(scores)
        if self.matching:
            pred_permutations, total_loss, loss_components = self.compute_matching_loss(predictions, sem_lbl, inst_lbl)
        else:
            pred_permutations, total_loss, loss_components = self.compute_nonmatching_loss(predictions, sem_lbl,
                                                                                           inst_lbl)
        return pred_permutations, total_loss, loss_components

    def _compute_optimal_match_loss_single_img(self, predictions, sem_lbl, inst_lbl):
        """
        Note: this function returns optimal match loss for a single image (not a batch)
        target: C,H,W.  C is the number of instances for ALL semantic classes.
        predictions: C,H,W
        semantic_instance_labels: the mapping from ground truth index to semantic labels.  This is
        needed so we only allow instances in the same semantic class to compete.
        gt_indices, perm_permutation -- indices into (0, ..., C-1) for gt and predictions of the
         matches.
        costs -- cost of each of the matches (also length C)
        """
        # print('APD: inside compute_optimal_match_loss_single_img')
        semantic_instance_labels = self.semantic_instance_labels
        assert len(semantic_instance_labels) == predictions.size(0), \
            'first dimension of predictions should be the number of channels.  It is {} instead. ' \
            'Are you trying to pass an entire batch into the loss function?'.format(predictions.size(0))
        gt_indices, pred_permutations, costs = [], [], []
        num_inst_classes = len(semantic_instance_labels)
        unique_semantic_values = range(max(semantic_instance_labels) + 1)
        for sem_val in unique_semantic_values:

            # print('APD: Running on sem_val {}'.format(sem_val))
            idxs = [i for i in range(num_inst_classes) if (semantic_instance_labels[i] == sem_val)]

            assignment, cost_list_2d = self._compute_optimal_match_loss_for_one_sem_cls(
                predictions, sem_lbl, inst_lbl, sem_val)
            # print('APD: Finished compute_optimal_match_loss_for_one_sem_cls')
            gt_indices += idxs
            try:
                # print('APD: Inside try statement')
                # print(len(idxs), assignment.NumNodes())
                pred_permutations += [idxs[assignment.RightMate(i)] for i in range(len(idxs))]
                # print('APD: past pred_permutations')
                costs += [cost_list_2d[assignment.RightMate(i)][i] for i in range(len(idxs))]
                # print('APD: past costs')
            except Exception as exc:
                print(exc)
                import ipdb; ipdb.set_trace()
                raise

        # print('APD: finished looping over classes')
        sorted_indices = np.argsort(gt_indices)
        gt_indices = np.array(gt_indices)[sorted_indices]
        pred_permutations = np.array(pred_permutations)[sorted_indices]
        costs = [costs[i] for i in sorted_indices]
        return gt_indices, pred_permutations, costs

    def _compute_optimal_match_loss_for_one_sem_cls(self, predictions, sem_lbl, inst_lbl, sem_val):
        cost_matrix, multiplier, cost_list_2d = self.build_cost_matrix_for_one_sem_cls(
            predictions, sem_lbl, inst_lbl, sem_val)
        assignment = match.solve_matching_problem(cost_matrix, multiplier)
        return assignment, cost_list_2d

    def build_all_sem_cls_cost_matrices_as_tensor_data(self, predictions, sem_lbl, inst_lbl, cost_list_only=True):
        if len(predictions.size()) == 4:
            if predictions.size(0) == 1:
                raise Exception('predictions, sem_lbl, inst_lbl should be formatted as coming from a single image.  '
                                'Yours is formatted as a minibatch, with size {}'.format(predictions.size()))
            else:
                raise Exception('predictions, sem_lbl, and inst_lbl should be a 3-D tensor (not 4-D)')
        unique_semantic_values = range(max(self.semantic_instance_labels) + 1)
        cost_matrix_tuples = [self.build_cost_matrix_for_one_sem_cls(predictions, sem_lbl, inst_lbl, sem_val=sem_val)
                              for sem_val in unique_semantic_values]
        if not cost_list_only:
            return cost_matrix_tuples
        else:
            cost_matrices_as_lists_of_variables = [c[2] for c in cost_matrix_tuples]
            cost_matrices_as_tensors = [torch.stack([torch.stack([cij.data for cij in ci])
                                                     for ci in c]) for c in cost_matrices_as_lists_of_variables
                                        ]
            return cost_matrices_as_tensors

    def build_cost_matrix_for_one_sem_cls(self, predictions, sem_lbl, inst_lbl, sem_val):
        # print('APD: building cost list 2d')
        cost_list_2d = match.create_pytorch_cost_matrix(self.component_loss, predictions,
                                                        sem_lbl, inst_lbl,
                                                        self.semantic_instance_labels,
                                                        self.instance_id_labels, sem_val,
                                                        size_average=self.size_average)
        # print('APD: Cost list 2d size: {}'.format((len(cost_list_2d), len(cost_list_2d[0]))))
        cost_matrix, multiplier = match.convert_pytorch_costs_to_ints(cost_list_2d)
        # print('APD: Cost matrix size: {}'.format((len(cost_matrix), len(cost_matrix[0]))))
        # print('APD: leaving build_cost_matrix_for_one_sem_cls')
        return cost_matrix, multiplier, cost_list_2d


class CrossEntropyComponentMatchingLoss(ComponentMatchingLossBase):
    loss_type = 'cross_entropy'

    def __init__(self, semantic_instance_labels=None, instance_id_labels=None, matching=True, size_average=True):
        super().__init__(semantic_instance_labels, instance_id_labels, matching, size_average)

    def transform_scores_to_predictions(self, scores):
        assert len(scores.size()) == 4
        return F.log_softmax(scores, dim=1)

    def component_loss(self, single_channel_prediction, binary_target):
        return xentropy.nll2d_single_class_term(single_channel_prediction, binary_target)


class SoftIOUComponentMatchingLoss(ComponentMatchingLossBase):
    loss_type = 'soft_iou'

    def __init__(self, semantic_instance_labels=None, instance_id_labels=None, matching=True, size_average=False):
        if size_average:
            raise Exception('Pretty sure you didn\'t want size_average to be True since it\'s already embedded in iou.')
        super().__init__(semantic_instance_labels, instance_id_labels, matching, size_average)

    def transform_scores_to_predictions(self, scores):
        assert len(scores.size()) == 4
        return F.softmax(scores, dim=1)

    def component_loss(self, single_channel_prediction, binary_target):
        return iou.my_soft_iou_loss(single_channel_prediction, binary_target)
