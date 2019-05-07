import torch
import gc


def generate_mem_report_dict():
    mem_report_as_dict = {}
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            mem_report_as_dict[obj.__name__] = (type(obj), obj.size())
    return mem_report_as_dict


def diff_mem_reports(mem_report_1, mem_report_2):
    # TODO(allie, someday): add a size tol to prune variables that don't change much
    new_vars_as_dict = {}
    same_vars_as_dict = {}
    diff_size_as_dict = {}
    for var_name, mem_dets2 in mem_report_2.items():
        try:
            mem_dets1 = mem_report_1[var_name]
            if mem_dets1 != mem_dets2:
                diff_size_as_dict[var_name] = mem_dets2
            else:
                same_vars_as_dict[var_name] = mem_dets2
        except KeyError:
            new_vars_as_dict[var_name] = mem_dets2

    return new_vars_as_dict, diff_size_as_dict, same_vars_as_dict
