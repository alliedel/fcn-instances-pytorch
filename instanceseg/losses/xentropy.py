import torch

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


# TODO(allie): Investigate losses 'size averaging'.  Gotta decide if should be overall or per
# class. -- Significantly affects the cost matrix and the contribution of the classes to the
# overall matching losses.
# TODO(allie): Do something with inst_lbl == 0 for instance classes.


DEBUG_ASSERTS = True


def tensors_are_close(tensor_a, tensor_b):
    return torch.np.allclose(tensor_a.cpu().numpy(), tensor_b.cpu().numpy())


def cross_entropy2d_without_matching(log_predictions, sem_lbl, inst_lbl, semantic_instance_labels,
                                     instance_id_labels, weight=None, size_average=True, return_loss_components=False):
    """
    Target should *not* be onehot.
    log_predictions: NxCxHxW
    sem_lbl, inst_lbl: NxHxW
    """
    assert sem_lbl.size() == inst_lbl.size()
    assert (log_predictions.size(0), log_predictions.size(2), log_predictions.size(3)) == sem_lbl.size()
    assert weight is None, NotImplementedError
    losses = []
    for sem_inst_idx, (sem_val, inst_val) in enumerate(zip(semantic_instance_labels, instance_id_labels)):
        binary_target_single_instance_cls = ((sem_lbl == sem_val) * (inst_lbl == inst_val)).float()
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


def nll2d_single_class_term(log_predictions_single_instance_cls, binary_target_single_instance_cls):
    lp = log_predictions_single_instance_cls
    bt = binary_target_single_instance_cls
    res = -torch.sum(lp.view(-1, ) * bt.view(-1, ))
    return res
