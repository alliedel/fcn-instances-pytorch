from scripts.configurations import synthetic_cfg
from torchfcn import script_utils
import numpy as np
from torchfcn.datasets import dataset_utils


def is_lr_ordered(sem_lbl, inst_lbl, unique_sem_lbls=None):
    if unique_sem_lbls is None:
        unique_sem_lbls = np.unique(sem_lbl)
    ordered = True
    for sem_val in unique_sem_lbls:
        unique_instance_idxs = sorted(np.unique(inst_lbl[sem_lbl == sem_val]))
        ordered_coms = []
        for inst_val in unique_instance_idxs:
            com = dataset_utils.compute_centroid_binary_mask(np.logical_and(sem_lbl == sem_val,
                                                                            inst_lbl == inst_val))
            ordered_coms.append(com)
        ordered_left_right_ordering = [x for x in np.argsort([com[1] for com in ordered_coms])]

        # Assert that they're actually in order
        if not all([x == y for x, y in zip(ordered_left_right_ordering, list(range(len(
                ordered_left_right_ordering))))]):
            ordered = False
            break
    return ordered


def test_lr_synthetic():
    cfg = synthetic_cfg.default_config
    print('Getting datasets')

    # Make sure something was unordered first
    cfg['ordering'] = None
    script_utils.set_random_seeds()
    train_dataset_unordered, _ = script_utils.get_voc_datasets(cfg, '/home/adelgior/data/datasets/', transform=False)
    unique_sem_lbls = range(train_dataset_unordered.n_semantic_classes)
    something_is_unordered = False
    for i, (sem_lbl, inst_lbl) in train_dataset_unordered:
        something_is_unordered = is_lr_ordered(sem_lbl, inst_lbl, unique_sem_lbls=unique_sem_lbls)
        if something_is_unordered:
            break
    if not something_is_unordered:
        raise Exception('All images were ordered before asserting LR ordering.  Can\'t verify ordering worked.')

    # Ensure left-right ordering works
    cfg['ordering'] = 'LR'
    script_utils.set_random_seeds()
    train_dataset_ordered, _ = script_utils.get_voc_datasets(cfg, '/home/adelgior/data/datasets/', transform=False)
    for i, (sem_lbl, inst_lbl) in train_dataset_ordered:
        assert is_lr_ordered(sem_lbl, inst_lbl, unique_sem_lbls=unique_sem_lbls), \
            Exception('Left-right ordering failed')


if __name__ == '__main__':
    test_lr_synthetic()
