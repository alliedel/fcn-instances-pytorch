import torch
import numpy as np
from torch.autograd import Variable


DEBUG_ASSERT = True


def assert_validation_images_arent_in_training_set(train_loader, val_loader):
    for val_idx, (val_img, _) in enumerate(val_loader):
        for train_idx, (train_img, _) in enumerate(train_loader):
            if np.allclose(train_img.numpy(), val_img.numpy()):
                import ipdb; ipdb.set_trace()
                raise Exception('validation img {} appears as training img {}'.format(val_idx,
                                                                                      train_idx))


def transform_lbl(lbl):
    lbl = torch.from_numpy(lbl).long()
    return lbl


def transform_img(img, mean_bgr):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img -= mean_bgr
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def untransform_lbl(lbl):
    lbl = lbl.numpy()
    return lbl


def untransform_img(img, mean_bgr):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img += mean_bgr
    img = img.astype(np.uint8)
    img = img[:, :, ::-1]
    return img


def zeros_like(x, out_size=None):
    assert x.__class__.__name__.find('Variable') != -1 \
           or x.__class__.__name__.find('Tensor') != -1, "Object is neither a Tensor nor a Variable"
    if out_size is None:
        out_size = x.size()
    y = torch.zeros(out_size)
    if x.is_cuda:
        y = y.cuda()

    if x.__class__.__name__ == 'Variable':
        return torch.autograd.Variable(y, requires_grad=x.requires_grad)
    elif x.__class__.__name__.find('Tensor') != -1:
        return torch.zeros(y)


def labels_to_one_hot(input_labels, n_classes, output_onehot=None):
    """
    input_labels: either HxW or NxHxW
    output_onehot: either CxHxW or NxCxHxW depending on input_labels
    """
    void_class = 1  # If void class could exist (-1), we'll leave room for it and then remove it.
    ndims = len(input_labels.size())

    if output_onehot is None:
        if ndims == 2:
            out_size = (n_classes + void_class, input_labels.size(0), input_labels.size(1))
        elif ndims == 3:
            out_size = (input_labels.size(0), n_classes + void_class, input_labels.size(1),
                        input_labels.size(2))
        else:
            raise ValueError('input_labels should be HxW or NxHxW')
        output_onehot = zeros_like(input_labels, out_size=out_size)
    else:
        output_onehot = output_onehot.zero_()
    if ndims == 2:
        channel_dim = 0
        input_labels_expanded = input_labels[torch.np.newaxis, :, :] + void_class
    else:  # ndims == 3:
        channel_dim = 1
        input_labels_expanded = input_labels[:, torch.np.newaxis, :, :] + void_class
    try:
        output_onehot.scatter_(channel_dim, input_labels_expanded, 1)
    except:
        import ipdb; ipdb.set_trace()
        raise
    if ndims == 2:
        output_onehot = output_onehot[void_class:, :, :]
    else:
        output_onehot = output_onehot[:, void_class:, :, :]

    return output_onehot


def permute_instance_order(inst_lbl, n_max_per_class):
    for old_val, new_val in enumerate(np.random.permutation(range(n_max_per_class))):
        inst_lbl[inst_lbl == old_val] = new_val
    return inst_lbl


def combine_semantic_and_instance_labels(sem_lbl, inst_lbl, n_max_per_class,
                                         set_extras_to_void=False):
    """
    sem_lbl is size(img); inst_lbl is size(img).  inst_lbl is just the original instance
    image (inst_lbls at coordinates of person 0 are 0)
    """
    assert sem_lbl.shape == inst_lbl.shape
    if torch.np.any(inst_lbl.numpy() >= n_max_per_class) and not set_extras_to_void:
        raise Exception('more instances than the number you allocated for ({} vs {}).'.format(
            inst_lbl.max(), n_max_per_class))
        # if you don't want to raise an exception here, add a corresponding flag and use the
        # following line:
        # y = torch.min(inst_lbl, self.n_max_per_class)
    y = inst_lbl
    overflow_instances = inst_lbl >= n_max_per_class
    y += (sem_lbl - 1) * n_max_per_class + 1
    y[sem_lbl == -1] = -1
    y[y < 0] = 0  # background got 1+range(-self.n_max_per_class,0); all go to 0.
    y[overflow_instances] = -1

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
    if torch.is_tensor(lbl):
        old_lbl = lbl.clone()
    else:
        old_lbl = lbl.copy()
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


def pytorch_unique(pytorch_1d_tensor):
    if torch.is_tensor(pytorch_1d_tensor):
        unique_labels = []
        for sem_l in pytorch_1d_tensor:
            if sem_l not in unique_labels:
                unique_labels.append(sem_l)
        return pytorch_1d_tensor.type(unique_labels)
    else:
        raise Exception('pytorch_1d_tensor isn\'t actually a tensor!  Maybe you want to use '
                        'local_pyutils.unique() for a list or np.unique() for a np array.')
