import gc
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from . import misc


def generate_mem_report_dict():
    mem_report_as_dict = {}
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if (type(obj), obj.size()) in mem_report_as_dict:
                mem_report_as_dict[(type(obj), obj.size())] += 1
            else:
                mem_report_as_dict[(type(obj), obj.size())] = 1
    return mem_report_as_dict


def diff_mem_reports(mem_report_1, mem_report_2):
    # TODO(allie, someday): add a size tol to prune variables that don't change much
    new_vars_as_dict = {}
    same_vars_as_dict = {}
    diff_counts_as_dict = {}
    for var_type_and_size_tuple2, count2 in mem_report_2.items():
        try:
            count1 = mem_report_1[var_type_and_size_tuple2]
            if count2 != count1:
                diff_counts_as_dict[var_type_and_size_tuple2] = (count1, count2)
            else:
                same_vars_as_dict[var_type_and_size_tuple2] = count2
        except KeyError:
            new_vars_as_dict[var_type_and_size_tuple2] = count2

    return new_vars_as_dict, diff_counts_as_dict, same_vars_as_dict


def garbage_collect(verbose=False, threshold_for_printing=0, color='WARNING'):
    if color is True:
        color = 'WARNING'
    memory_before_garbage_collection = torch.cuda.memory_allocated(device=None)
    num_cleaned = gc.collect()
    memory_after_garbage_collection = torch.cuda.memory_allocated(device=None)
    memory_cleaned = memory_before_garbage_collection - memory_after_garbage_collection
    if verbose:
        if memory_cleaned > threshold_for_printing:
            print_str = 'Cleaned %d objects, %g MB' % (num_cleaned, memory_cleaned / 1e6)
            print(misc.color_text(print_str, color=color))
    return num_cleaned, memory_cleaned


def softmax_scores(compiled_scores, dim=1):
    return F.softmax(Variable(compiled_scores), dim=dim).data


def argmax_scores(compiled_scores, dim=1):
    return compiled_scores.max(dim=dim)[1]


def center_crop_to_reduced_size(tensor, cropped_size_rc, rc_axes=(1, 2)):
    if rc_axes == (1, 2):
        start_coords = (int((tensor.size(1) - cropped_size_rc[0]) / 2),
                        int((tensor.size(2) - cropped_size_rc[1]) / 2))
        cropped_tensor = tensor[:,
                         start_coords[0]:(start_coords[0] + cropped_size_rc[0]),
                         start_coords[1]:(start_coords[1] + cropped_size_rc[1])]
    elif rc_axes == (2, 3):
        assert len(tensor.size()) == 4, NotImplementedError
        start_coords = (int((tensor.size(2) - cropped_size_rc[0]) / 2),
                        int((tensor.size(3) - cropped_size_rc[1]) / 2))
        cropped_tensor = tensor[:, :,
                         start_coords[0]:(start_coords[0] + cropped_size_rc[0]),
                         start_coords[1]:(start_coords[1] + cropped_size_rc[1])]
        assert cropped_tensor.size()[2:] == cropped_size_rc
    else:
        raise NotImplementedError
    return cropped_tensor


def fast_remap(arr, old_vals, new_vals):
    arr_old = arr.copy()
    for old_val, new_val in zip(old_vals, new_vals):
        arr[arr_old == old_val] = new_val
    return arr