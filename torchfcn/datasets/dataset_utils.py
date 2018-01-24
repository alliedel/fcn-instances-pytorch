import torch
import numpy as np

DEBUG_ASSERT = True


def combine_semantic_and_instance_labels(sem_lbl, inst_lbl, n_max_per_class, n_semantic_classes,
                                         n_classes):
    """
    sem_lbl is size(img); inst_lbl is size(img).  inst_lbl is just the original instance
    image (inst_lbls at coordinates of person 0 are 0)
    """
    assert sem_lbl.shape == inst_lbl.shape
    if torch.np.any(inst_lbl >= n_max_per_class):
        raise Exception('There are more instances than the number you allocated for.')
        # if you don't want to raise an exception here, add a corresponding flag and use the
        # following line:
        # y = torch.min(inst_lbl, self.n_max_per_class)
    y = inst_lbl
    y += (sem_lbl - 1) * n_max_per_class
    y[y < 0] = 0  # background got 1+range(-self.n_max_per_class,0); all go to 0.

    instance_to_semantic_mapping = get_instance_to_semantic_mapping(
        n_max_per_class, n_semantic_classes)
    if DEBUG_ASSERT:
        mapping_as_list_of_semantic_classes = torch.np.nonzero(
            torch.from_numpy(np.arange(n_semantic_classes)[
                                 instance_to_semantic_mapping[:,:] == 1
                                 ])).squeeze()
        assert mapping_as_list_of_semantic_classes.size() == (n_classes, n_semantic_classes)
        # import ipdb; ipdb.set_trace()
        # assert instance_to_semantic_mapping[y.view(-1)].double()[sem_lbl.view(-1)] == 1

    return y


def get_instance_to_semantic_mapping(n_max_per_class, n_semantic_classes_with_background):
    """
    returns a binary matrix, where semantic_instance_mapping is N x S
    (N = # instances, S = # semantic classes)
    semantic_instance_mapping[inst_idx, :] is a one-hot vector,
    and semantic_instance_mapping[inst_idx, sem_idx] = 1 iff that instance idx is an instance
    of that semantic class.
    """

    if n_max_per_class == 1:
        instance_to_semantic_mapping_matrix = torch.eye(n_semantic_classes_with_background,
                                                        n_semantic_classes_with_background)
    else:
        n_instance_classes = \
            1 + (n_semantic_classes_with_background - 1) * n_max_per_class
        instance_to_semantic_mapping_matrix = torch.zeros(
            (n_instance_classes, n_semantic_classes_with_background)).float()

        semantic_instance_class_list = [0]
        for semantic_class in range(n_semantic_classes_with_background - 1):
            semantic_instance_class_list += [semantic_class for _ in range(
                n_max_per_class)]
        for instance_idx, semantic_idx in enumerate(semantic_instance_class_list):
            instance_to_semantic_mapping_matrix[instance_idx,
                                                semantic_idx] = 1
    return instance_to_semantic_mapping_matrix


def remap_to_reduced_semantic_classes(lbl, reduced_class_idxs, map_other_classes_to_bground=True):
    """
    reduced_class_idxs = idxs_into_all_voc
    """
    # Make sure all lbl classes can be mapped appropriately.
    if not map_other_classes_to_bground:
        original_classes_in_this_img = [i for i in range(lbl.min(), lbl.max() + 1)
                                        if torch.sum(lbl == i) > 0]
        bool_unique_class_in_reduced_classes = [lbl_cls in reduced_class_idxs
                                                for lbl_cls in original_classes_in_this_img
                                                if lbl_cls != -1]
        if not all(bool_unique_class_in_reduced_classes):
            print(bool_unique_class_in_reduced_classes)
            import ipdb;
            ipdb.set_trace()
            raise Exception('Image has class labels outside the subset.\n Subset: {}\n'
                            'Classes in the image:{}'.format(reduced_class_idxs,
                                                             original_classes_in_this_img))
    old_lbl = lbl.clone()
    lbl[...] = 0
    lbl[old_lbl == -1] = -1
    for new_idx, old_class_idx in enumerate(reduced_class_idxs):
        lbl[old_lbl == old_class_idx] = new_idx
    return lbl


def get_semantic_names_and_idxs(semantic_subset, full_set):
    """
    For VOC, full_set = voc.ALL_VOC_CLASS_NAMES
    """
    if semantic_subset is None:
        names = full_set
        idxs_into_all_voc = range(len(full_set))
    else:
        idx_name_tuples = [(idx, cls) for idx, cls in enumerate(full_set)
                           if cls in semantic_subset]
        idxs_into_all_voc = [tup[0] for tup in idx_name_tuples]
        names = [tup[1] for tup in idx_name_tuples]
        assert 'background' in names or 0 in names, ValueError('You must include background in the '
                                                           'list of classes.')
        if len(idxs_into_all_voc) != len(semantic_subset):
            unrecognized_class_names = [cls for cls in semantic_subset if cls not in names]
            raise Exception('unrecognized class name(s): {}'.format(unrecognized_class_names))
    return names, idxs_into_all_voc
