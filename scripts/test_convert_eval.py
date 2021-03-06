import os
import argparse
import numpy as np
import yaml

import instanceseg.utils.script_setup
from instanceseg.utils import script_setup as script_utils

if os.path.basename(os.path.abspath('.')) == 'debugging' or os.path.basename(os.path.abspath('.')) == 'scripts':
    os.chdir('../')

from scripts import test, evaluate, convert_test_results_to_coco
from scripts.visualizations import visualize_pq_stats, export_prediction_vs_gt_vis_sorted

if 'panopticapi' not in os.environ['PYTHONPATH']:
    os.environ['PYTHONPATH'] += ':' + os.path.abspath(os.path.expanduser('./instanceseg/ext'))
    os.environ['PYTHONPATH'] += ':' + os.path.abspath(os.path.expanduser('./instanceseg/ext/panopticapi'))


def get_test_parser_without_logdir():
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing_test_cache_dir', default=None,
                        help='if test.py and convert...py were already successful and you want to '
                             'run the rest of the pipeline (eval, visualize)')
    parser.add_argument('--test_split', default='val', help='train, val, test, or any other split the dataloader can '
                                                            'load for this dataset')
    parser.add_argument('--dataset_name', default=None, help='dataset; default=dataset you trained on')
    parser.add_argument('--gpu', '-g', type=int, nargs='+',
                        help='dataset; default=dataset you trained on', required=True)
    parser.add_argument('--sampler', default=None)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for the dataloader of the designated '
                                                                  'test split')
    parser.add_argument('--iou_threshold', default=0.5, type=float, help='Threshold to count TP in evaluation')
    parser.add_argument('--export_sorted_perf_images', default=True, help='Export tiled images ordered by performance')
    return parser


def get_test_parser():
    parser = get_test_parser_without_logdir()
    parser.add_argument('logdir', help='directory with model path')
    return parser


def parse_args():
    parser = get_test_parser()
    return parser.parse_args()


def main(logdir, test_split, gpu, sampler=None, batch_size=1, dataset_name=None, iou_threshold=0.5,
         export_sorted_perf_images=True, check_clean_tree=True, existing_test_cache_dir=None):
    if check_clean_tree:
        script_utils.check_clean_work_tree()

    # Test
    np.random.seed(100)
    if existing_test_cache_dir is not None:
        assert os.path.isdir(existing_test_cache_dir), ValueError('{} does not exist'.format(existing_test_cache_dir))
        cache_out_root = existing_test_cache_dir
    else:
        train_logdir = logdir.rstrip('/')
        config = yaml.safe_load(open(os.path.join(train_logdir, 'config.yaml'), 'rb'))
        dataset_name = dataset_name or config['dataset']
        replacement_dict_for_sys_args = [dataset_name, '--logdir', train_logdir, '--{}_batch_size'.format(test_split),
                                         str(batch_size), '-g', ' '.join(str(g) for g in gpu), '--test_split',
                                         test_split,
                                         '--sampler', sampler]
        predictions_outdir, groundtruth_outdir, tester, test_logdir = test.main(replacement_dict_for_sys_args,
                                                                                check_clean_tree=False)
        # Convert
        cache_out_root = instanceseg.utils.script_setup.get_cache_dir_from_test_logdir(test_logdir)
        problem_config = tester.exporter.instance_problem.load(tester.exporter.instance_problem_path)
        out_jsons, out_dirs = convert_test_results_to_coco.main(predictions_outdir, groundtruth_outdir, problem_config,
                                                                cache_out_root)

    # Evaluate
    collated_stats_per_image_file = evaluate.main(cache_out_root, iou_threshold=iou_threshold)

    visualize_pq_stats.main(collated_stats_per_image_file)

    if export_sorted_perf_images:
        export_prediction_vs_gt_vis_sorted.main(collated_stats_npz=collated_stats_per_image_file)

    return collated_stats_per_image_file


if __name__ == '__main__':
    args = parse_args()

    collated_stats_per_image_per_cat_file = main(**args.__dict__)

    pass
