if DEBUG_MEMORY_ISSUES:
    memory_allocated = torch.cuda.memory_allocated(device=None)
    description = 'Valid iteration (split=%s)=%d, %g GB' % \
                  (split, self.state.iteration, memory_allocated / 1e9)
    t.set_description_str(description)
    mem_report_dict_old = mem_report_dict
    mem_report_dict = torch_utils.generate_mem_report_dict()
    new_vars_as_dict, diff_counts_as_dict, same_vars_as_dict = \
        torch_utils.diff_mem_reports(mem_report_dict_old, mem_report_dict)
    if batch_idx > num_images_to_visualize:
        print('\nNew vars:')
        pprint.pprint(new_vars_as_dict)
        print('\nDiff vars:')
        pprint.pprint(diff_counts_as_dict)
        vars_to_check = ['assignments_sb', 'val_loss_sb',
                         'segmentation_visualizations_sb',
                         'score_visualizations_sb']
        for var_name in vars_to_check:
            value = eval(var_name)
            if type(value) is list and len(value) > 0:
                element_sizes = [get_array_size(v) for v in value]
                if all([s == element_sizes[0] for s in element_sizes]):
                    var_size = \
                        '{} of {}'.format(len(element_sizes), element_sizes[0])
                else:
                    var_size = element_sizes
                var_type = 'list of {}'.format(type(value[0]))
            else:
                var_size = get_array_size(value)
                var_type = type(value)
            print('{}: {}, {}'.format(var_name, var_type, var_size))
