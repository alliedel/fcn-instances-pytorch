import os

if os.path.basename(os.path.abspath('.')) == 'debugging' or os.path.basename(os.path.abspath('.')) == 'scripts':
    os.chdir('../')


from scripts import test, evaluate, convert_test_results_to_coco


if 'panopticapi' not in os.environ['PYTHONPATH']:
    os.environ['PYTHONPATH'] += ':' + os.path.abspath(os.path.expanduser('./instanceseg/ext'))
    os.environ['PYTHONPATH'] += ':' + os.path.abspath(os.path.expanduser('./instanceseg/ext/panopticapi'))


if __name__ == '__main__':

    logdir = '../old_instanceseg/scripts/logs/cityscapes/train_instances_filtered_2019-05-14' \
             '-133452_VCS-1e74989_SAMPLER-car_2_4_BACKBONE-resnet50_ITR-1000000_NPER-4_SSET-car_person'
    test_split = 'val'
    replacement_dict_for_sys_args = ['cityscapes', '--logdir', logdir, '--{}_batch_size'.format(test_split), '2', '-g',
                                     '1', '--sampler', 'car_2_4_1image', '--test_split', test_split]
    # Test
    predictions_outdir, groundtruth_outdir, tester, logdir = test.main(replacement_dict_for_sys_args)

    # Convert
    out_dirs_root = convert_test_results_to_coco.get_outdirs_cache_root(logdir, predictions_outdir)
    problem_config = tester.exporter.instance_problem.load(tester.exporter.instance_problem_path)
    out_jsons, out_dirs = convert_test_results_to_coco.main(predictions_outdir, groundtruth_outdir, problem_config,
                                                            out_dirs_root)

    # Evaluate
    collated_stats_per_image_per_cat, categories = evaluate.main(out_jsons['gt'], out_jsons['pred'], out_dirs['gt'],
                                                                 out_dirs['pred'], problem_config)

    pass
