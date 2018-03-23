import torch
import numpy as np

import local_pyutils


def combine_semantic_and_instance_labels(sem_lbl, inst_lbl, semantic_instance_class_list,
                                         set_extras_to_void=True, void_value=-1):
    """
    sem_lbl is size(img); inst_lbl is size(img).  inst_lbl is just the original instance
    image (inst_lbls at coordinates of person 0 are 0)
    """
    # TODO(allie): handle class overflow (from ground truth)
    assert set_extras_to_void == True, NotImplementedError
    assert sem_lbl.shape == inst_lbl.shape
    if torch.is_tensor(inst_lbl):
        y = inst_lbl.clone()
    else:
        y = inst_lbl.copy()
    y[...] = void_value
    unique_semantic_vals, inst_counts = np.unique(semantic_instance_class_list, return_counts=True)
    for sem_val, n_instances_for_this_sem_cls in zip(unique_semantic_vals, inst_counts):
        for inst_val in range(n_instances_for_this_sem_cls):
            sem_inst_idx = local_pyutils.nth_item(n=inst_val, item=sem_val,
                                                  iterable=semantic_instance_class_list)
            try:
                y[(sem_lbl == sem_val) * (inst_lbl == inst_val)] = sem_inst_idx
            except:
                import ipdb; ipdb.set_trace()
                raise
#    if np.sum(y == void_value) == 0:
#        raise Exception('void class got removed here')
    return y
