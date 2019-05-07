import torch
import gc
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
