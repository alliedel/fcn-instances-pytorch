import shutil

import PIL.Image
import numpy as np
import scipy.misc
import scipy.ndimage
import torch
from torch.autograd import Variable

# TODO(allie): Allow for augmentations

DEBUG_ASSERT = True


def prep_input_for_scoring(input_tensor, cuda):
    """
    n_semantic_classes only needed for augmenting.
    """
    if cuda:
        input_tensor = input_tensor.cuda()
    input_variable = Variable(input_tensor)
    return input_variable


def prep_inputs_for_scoring(img_tensor, sem_lbl_tensor, inst_lbl_tensor, cuda):
    """
    n_semantic_classes only needed for augmenting.
    """
    img_var, sem_lbl_var, inst_lbl_var = tuple(prep_input_for_scoring(x, cuda) for x in (img_tensor, sem_lbl_tensor,
                                                                                         inst_lbl_tensor))
    return img_var, sem_lbl_var, inst_lbl_var


def assert_validation_images_arent_in_training_set(train_loader, val_loader):
    for val_idx, (val_img, _) in enumerate(val_loader):
        for train_idx, (train_img, _) in enumerate(train_loader):
            if np.allclose(train_img.numpy(), val_img.numpy()):
                import ipdb; ipdb.set_trace()
                raise Exception('validation img {} appears as training img {}'.format(val_idx,
                                                                                      train_idx))


def convert_img_to_torch_tensor(img, mean_bgr=None):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    try:
        if mean_bgr is not None:
            img -= mean_bgr
    except:
        import ipdb; ipdb.set_trace()

    # NHWC -> NCWH
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def convert_lbl_to_torch_tensor(lbl):
    lbl = torch.from_numpy(lbl).long()  # NOTE(allie): lbl.float() (?)
    return lbl


def convert_torch_lbl_to_numpy(lbl):
    lbl = lbl.numpy()
    return lbl


def convert_torch_img_to_numpy(img, mean_bgr=None):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    if mean_bgr is not None:
        img += mean_bgr
    img = img.astype(np.uint8)
    img = img[:, :, ::-1]
    return img


def resize_lbl(lbl, resized_sz):
    if resized_sz is not None:
        lbl = lbl.astype(float)
        lbl = scipy.misc.imresize(lbl, (resized_sz[0], resized_sz[1]), 'nearest', mode='F')
    return lbl


def resize_img(img, resized_sz):
    if resized_sz is not None:
        img = scipy.misc.imresize(img, (resized_sz[0], resized_sz[1]))
    return img


def zeros_like(x, out_size=None):
    assert isinstance(x, torch.autograd.Variable) or torch.is_tensor(x)
    if out_size is None:
        out_size = x.size()
    y = torch.zeros(out_size)
    if x.is_cuda:
        y = y.cuda()

    if isinstance(x, torch.autograd.Variable):
        return torch.autograd.Variable(y, requires_grad=x.requires_grad)
    else:
        return y


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


def remap(lbl, new_idxs):
    if torch.is_tensor(lbl):
        old_lbl = lbl.clone()
    else:
        old_lbl = lbl.copy()
    lbl[...] = -2
    lbl[old_lbl == -1] = -1
    for old_class_idx, new_idx in enumerate(new_idxs):
        lbl[old_lbl == old_class_idx] = new_idx

    if DEBUG_ASSERT:
        if any(lbl == -2):
            untouched_values = old_lbl[lbl == -2]
            import ipdb; ipdb.set_trace()
            raise Exception('mapping was not thorough.  No value specified for {}'.format(untouched_values[0]))


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
        if 'background' not in names or 0 in names:
            print(Warning('Background is not included in the list of classes...'))
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


def augment_channels(tensor, augmentation_tensor, dim=0):
    return torch.cat([tensor, augmentation_tensor], dim=dim)


def load_img_as_dtype(img_file, dtype):
    img = PIL.Image.open(img_file)
    img = np.array(img, dtype=dtype)
    return img


def write_np_array_as_img(arr, filename):
    im = PIL.Image.fromarray(arr.astype(np.uint8))
    im.save(filename)


def write_np_array_as_img_with_borrowed_colormap_palette(arr, filename, filename_for_colormap):
    colormap_src = PIL.Image.open(filename_for_colormap)
    if colormap_src.mode == 'P':
        write_np_array_as_img_with_colormap_palette(arr, filename, palette=colormap_src)
    elif colormap_src.mode in ['I', 'L']:
        new_lbl_img = PIL.Image.fromarray(arr)
        new_lbl_img.convert(mode=colormap_src.mode)
        new_lbl_img.save(filename)
    else:
        raise NotImplementedError


def write_np_array_as_img_with_colormap_palette(arr, filename, palette):
    im = PIL.Image.fromarray(arr.astype(np.uint8))
    converted = im.quantize(palette=palette)
    converted.save(filename)


def generate_per_sem_instance_file(inst_absolute_lbl_file, sem_lbl_file, inst_lbl_file):
    """
    Nominally VOC-specific, though may be useful for other datasets.
    Converts instance labels so they start from 1 for every semantic class (instead of person 1, person 2, car 3,
    etc. -- remaps to person 1, person 2, car 1)
    """
    print('Generating per-semantic instance file: {}'.format(inst_lbl_file))
    sem_lbl = load_img_as_dtype(sem_lbl_file, np.int32)
    sem_lbl[sem_lbl == 255] = -1
    unique_sem_lbls = np.unique(sem_lbl)
    if sum(unique_sem_lbls > 0) <= 1:  # only one semantic object type
        shutil.copyfile(inst_absolute_lbl_file, inst_lbl_file)
    else:
        inst_lbl = load_img_as_dtype(inst_absolute_lbl_file, np.int32)
        inst_lbl[inst_lbl == 255] = -1
        for sem_val in unique_sem_lbls[unique_sem_lbls > 0]:
            first_instance_idx = inst_lbl[sem_lbl == sem_val].min()
            inst_lbl[sem_lbl == sem_val] -= (first_instance_idx - 1)
        inst_lbl[inst_lbl == -1] = 255
        write_np_array_as_img_with_borrowed_colormap_palette(inst_lbl, inst_lbl_file, inst_absolute_lbl_file)


def generate_ordered_instance_file(inst_lbl_file_unordered, sem_lbl_file, out_inst_lbl_file_ordered, ordering,
                                   increasing):
    print('Generating {}-ordered instance file: {}'.format(ordering, out_inst_lbl_file_ordered))
    sem_lbl = load_img_as_dtype(sem_lbl_file, np.int32)
    inst_lbl = load_img_as_dtype(inst_lbl_file_unordered, np.int32)
    inst_lbl[inst_lbl == 255] = -1

    inst_lbl = make_ordered_copy_of_inst_lbl(inst_lbl, sem_lbl, ordering, increasing)

    inst_lbl[inst_lbl == -1] = 255
    write_np_array_as_img_with_borrowed_colormap_palette(inst_lbl, out_inst_lbl_file_ordered, inst_lbl_file_unordered)


def make_ordered_copy_of_inst_lbl(inst_lbl, sem_lbl, ordering, increasing):
    old_inst_lbl = inst_lbl.copy()
    unique_sem_lbls = np.unique(sem_lbl)
    for sem_val in unique_sem_lbls[unique_sem_lbls > 0]:
        unique_instance_idxs = np.unique(old_inst_lbl[sem_lbl == sem_val])
        if DEBUG_ASSERT:
            assert not np.any(unique_instance_idxs == 0)
        unique_instance_idxs = unique_instance_idxs[unique_instance_idxs > 0]  # don't remap void
        attribute_values = []
        for old_inst_val in unique_instance_idxs:
            if ordering == 'size':
                ordering_attribute = get_instance_size(sem_lbl, sem_val, old_inst_lbl, old_inst_val)
            elif ordering == 'lr':
                ordering_attribute = get_instance_centroid(sem_lbl, sem_val, old_inst_lbl, old_inst_val)
            else:
                raise NotImplementedError
            attribute_values.append(ordering_attribute)
        increasing_ordering = [x for x in np.argsort([com[1] for com in attribute_values])]
        size_ordering = increasing_ordering if increasing else increasing_ordering[::-1]
        if not all([x == y for x, y in zip(size_ordering, list(range(len(size_ordering))))]):
            print('debug: confirmed we reordered at least one instance')
        old_inst_vals = [unique_instance_idxs[x] for x in size_ordering]

        for new_inst_val_minus_1, old_inst_val in enumerate(old_inst_vals):
            new_inst_val = new_inst_val_minus_1 + 1
            inst_lbl[np.logical_and(sem_lbl == sem_val, old_inst_lbl == old_inst_val)] = new_inst_val
    return inst_lbl


def get_instance_centroid(sem_lbl, sem_val, inst_lbl, inst_val):
    return compute_centroid_binary_mask(np.logical_and(sem_lbl == sem_val, inst_lbl == inst_val))


def get_instance_size(sem_lbl, sem_val, inst_lbl, inst_val):
    return (np.logical_and(sem_lbl == sem_val, inst_lbl == inst_val)).sum()


def get_image_center(img_size, floor=False):
    if floor:
        return [sz // 2 for sz in img_size]
    else:
        return [sz / 2 for sz in img_size]


def compute_centroid_binary_mask(binary_mask):
    return np.argwhere(binary_mask).sum(axis=0)/binary_mask.sum()
