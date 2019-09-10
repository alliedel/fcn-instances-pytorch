import numpy as np
import torch
from torch.nn import functional as F

from instanceseg.losses import match
from instanceseg.losses import xentropy, iou
from instanceseg.losses.xentropy import DEBUG_ASSERTS

# TODO(allie): Implement test: Compare component loss function with full loss function when matching is off
from instanceseg.utils.misc import AttrDict

LOSS_TYPES = ['cross_entropy', 'soft_iou', 'xent']


def get_subclasses(cls):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in get_subclasses(c)])


# matching_component_loss_registry = {loss_class.loss_type: loss_class
#                                     for loss_class in get_subclasses(ComponentMatchingLossBase)}


def loss_object_factory(loss_type, semantic_instance_class_list, instance_id_count_list, matching, size_average):
    assert loss_type in LOSS_TYPES, 'Loss type must be one of {}; not {}'.format(LOSS_TYPES, loss_type)
    if loss_type == 'cross_entropy' or loss_type == 'xent':
        loss_object = CrossEntropyComponentMatchingLoss(semantic_instance_class_list, instance_id_count_list, matching,
                                                        size_average)
    elif loss_type == 'soft_iou':
        loss_object = SoftIOUComponentMatchingLoss(semantic_instance_class_list, instance_id_count_list, matching,
                                                   size_average)
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
         LossMatchAssignments
         {component_channels: NxC  we expect channels[i, ...] = range(C), but we put this here to be sure.
         component_sem_vals: NxC  corresponding semantic values for each pred/gt channel
                                  component_sem_vals[i, c] = sem_inst_idxs[component_channels[c]]
         gt_inst_vals: NxC        corresponding ground truth inst values for each matched pred channel
         }
         total_loss: scalar, requires_grad=True: avg (for instance) of loss_components
         loss_components_by_channel: NxC
        """
        raise NotImplementedError


class LossMatchAssignments(AttrDict):
    def __init__(self, model_channels, assigned_gt_inst_vals, sem_values, unassigned_gt_sem_inst_tuples):
        # super(LossMatchAssignments, self).__init__(self)
        n_channels = model_channels.shape[1]
        assert assigned_gt_inst_vals.shape[1] == n_channels
        assert sem_values.shape[1] == n_channels
        self.model_channels = model_channels  # torch.empty((0, n_channels))
        self.assigned_gt_inst_vals = assigned_gt_inst_vals  # torch.empty((0, n_channels))
        self.sem_values = sem_values  # torch.empty((0, n_channels))
        self.unassigned_gt_sem_inst_tuples = unassigned_gt_sem_inst_tuples

    @classmethod
    def allocate(cls, n_images, n_channels):
        model_channels = torch.empty((n_images, n_channels))
        assigned_gt_inst_vals = torch.empty((n_images, n_channels))  # torch.empty((0, n_channels))
        sem_values = torch.empty((n_images, n_channels))
        unassigned_gt_sem_inst_tuples = [[] for _ in range(n_images)]
        return cls(model_channels=model_channels,
                   assigned_gt_inst_vals=assigned_gt_inst_vals,
                   sem_values=sem_values,
                   unassigned_gt_sem_inst_tuples=unassigned_gt_sem_inst_tuples)

    @classmethod
    def assemble(cls, list_of_loss_match_assignments):
        model_channels = torch.cat([lma.model_channels for lma in list_of_loss_match_assignments], dim=0)
        assigned_gt_inst_vals = torch.cat([lma.assigned_gt_inst_vals for lma in list_of_loss_match_assignments], dim=0)
        sem_values = torch.cat([lma.sem_values for lma in list_of_loss_match_assignments], dim=0)
        unassigned_gt_sem_inst_tuples = []
        for lma in list_of_loss_match_assignments:
            unassigned_gt_sem_inst_tuples.extend(lma.unassigned_gt_sem_inst_tuples)
        assembled_assignments = cls(model_channels=model_channels,
                                    assigned_gt_inst_vals=assigned_gt_inst_vals,
                                    sem_values=sem_values,
                                    unassigned_gt_sem_inst_tuples=unassigned_gt_sem_inst_tuples)
        return assembled_assignments

    def insert_assignment_for_image(self, image_index, model_channels, assigned_gt_inst_vals, sem_values,
                                    unassigned_gt_sem_inst_tuples):
        self.model_channels[image_index, :] = model_channels  # torch.empty((0, n_channels))
        self.assigned_gt_inst_vals[image_index, :] = assigned_gt_inst_vals# torch.empty((0, n_channels))
        self.sem_values[image_index, :] = sem_values  # torch.empty((0, n_channels))
        self.unassigned_gt_sem_inst_tuples[image_index] = unassigned_gt_sem_inst_tuples


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
        self.unique_semantic_values = sorted([s for s in np.unique(semantic_instance_labels)])
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

        loss_components_per_channel = torch.empty((batch_sz, n_channels))
        # []  # dataset_utils.zeros_like(log_predictions, (n, c))
        assignments = LossMatchAssignments.allocate(n_images=batch_sz, n_channels=n_channels)

        # Compute optimal match & costs for each image in the batch
        for i in range(batch_sz):
            assignment_values, costs = \
                self._compute_optimal_match_loss_single_img(predictions[i, ...], sem_lbl[i, ...], inst_lbl[i, ...])
            assignments.insert_assignment_for_image(i, **assignment_values)
            loss_components_per_channel[i, :] = costs
            # all_costs.append(costs)
        # all_costs = torch.cat([c[None, :] for c in all_costs], dim=0).float()
        total_train_loss = loss_components_per_channel.sum()
        if DEBUG_ASSERTS:
            if loss_components_per_channel.size(1) != len(self.semantic_instance_labels):
                import ipdb;
                ipdb.set_trace()
                raise Exception
        return assignments, total_train_loss, loss_components_per_channel

    def get_binary_gt_for_channel(self, sem_lbl, inst_lbl, channel_idx):
        sem_val, inst_val = self.semantic_instance_labels[channel_idx], self.instance_id_labels[channel_idx]
        return ((sem_lbl == sem_val) * (inst_lbl == inst_val)).float()

    def compute_nonmatching_loss(self, predictions, sem_lbl, inst_lbl):
        batch_sz, n_channels = predictions.size(0), predictions.size(1)

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

        return iou.mean(losses), component_losses

    def loss_fcn(self, scores, sem_lbl, inst_lbl):
        """
        # component_channels: NxC  we expect channels[i, ...] = range(C), but we put this here to be sure.
        # component_sem_vals: NxC  corresponding semantic values for each pred/gt channel
        #                          component_sem_vals[i, c] = sem_inst_idxs[component_channels[c]]
        # gt_inst_vals: NxC        corresponding ground truth inst values for each matched pred channel
        # total_loss: scalar, requires_grad=True: avg (for instance) of loss_components
        # loss_components_by_channel: NxC
        # additional_out: {}
        """

        predictions = self.transform_scores_to_predictions(scores)
        if self.matching:
            assignments, total_loss, loss_components_by_channel = \
                self.compute_matching_loss(predictions, sem_lbl, inst_lbl)
        else:
            total_loss, loss_components_by_channel = self.compute_nonmatching_loss(predictions, sem_lbl, inst_lbl)
            assignments = None
        return assignments, total_loss, loss_components_by_channel

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
        C = predictions.size(0)
        assert len(semantic_instance_labels) == C, \
            'first dimension of predictions should be the number of channels.  It is {} instead. ' \
            'Are you trying to pass an entire batch into the loss function?'.format(predictions.size(0))
        costs = -1 * torch.ones((C,))
        model_channels = torch.empty((C,), dtype=torch.long)
        sem_values = torch.empty((C,))
        assigned_gt_inst_values = torch.empty((C,))
        unassigned_gt_sem_inst_tuples = []
        for sem_val in self.unique_semantic_values:
            assert int(sem_val) == sem_val
            sem_val = int(sem_val)
            # print('APD: Running on sem_val {}'.format(sem_val))
            costs_this_cls, model_channels_for_this_cls, assigned_gt_inst_vals_this_cls, \
                unassigned_gt_inst_vals_this_cls = \
                self._compute_optimal_match_loss_for_one_sem_cls(predictions, sem_lbl, inst_lbl, sem_val)
            channel_idxs = torch.LongTensor(model_channels_for_this_cls)
            model_channels[channel_idxs] = channel_idxs
            costs[channel_idxs] = costs_this_cls
            assigned_gt_inst_values[channel_idxs] = torch.FloatTensor(assigned_gt_inst_vals_this_cls)
            sem_values[channel_idxs] = sem_val
            unassigned_gt_sem_inst_tuples.append([(sem_val, i) for i in unassigned_gt_inst_vals_this_cls])
        assert torch.all(costs != -1)  # costs for all channels were filled
        assignment_values = {
            'model_channels': model_channels,
            'assigned_gt_inst_vals': assigned_gt_inst_values,
            'sem_values': sem_values,
            'unassigned_gt_sem_inst_tuples': unassigned_gt_sem_inst_tuples
        }

        return assignment_values, costs

    def _compute_optimal_match_loss_for_one_sem_cls(self, predictions, sem_lbl, inst_lbl, sem_val):
        cost_tensor, model_channels_for_this_cls, gt_inst_vals_present = self.build_cost_tensor_for_one_sem_cls(
            predictions, sem_lbl, inst_lbl, sem_val)
        assigned_col_inds = match.solve_matching_problem(cost_tensor)
        assigned_gt_inst_vals = [gt_inst_vals_present[col_ind] for col_ind in assigned_col_inds]
        if len(assigned_col_inds) == len(gt_inst_vals_present):
            unassigned_gt_inst_vals = []
        else:
            unassigned_gt_inst_vals = [v for v in gt_inst_vals_present if v not in assigned_gt_inst_vals]
            assert (len(unassigned_gt_inst_vals) + len(assigned_gt_inst_vals)) == len(gt_inst_vals_present), \
                'Debug error'

        costs = cost_tensor[range(cost_tensor.shape[0]), assigned_col_inds]

        return costs, model_channels_for_this_cls, assigned_gt_inst_vals, unassigned_gt_inst_vals

    def build_cost_tensor_for_one_sem_cls(self, predictions, sem_lbl, inst_lbl, sem_val):
        """
        Creates cost_tensor[prediction, ground_truth]
        """

        cost_tensor, model_channels_for_this_cls, gt_inst_vals_present = match.create_pytorch_cost_matrix(
            self.component_loss, predictions, sem_lbl, inst_lbl,
            self.semantic_instance_labels, sem_val, size_average=self.size_average)
        return cost_tensor, model_channels_for_this_cls, gt_inst_vals_present


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
