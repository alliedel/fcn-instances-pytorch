import os
import argparse
import numpy as np

if os.path.basename(os.path.abspath('.')) == 'debugging' or os.path.basename(os.path.abspath('.')) == 'scripts':
    os.chdir('../')

from scripts import test, evaluate, convert_test_results_to_coco
from scripts.visualizations import visualize_pq_stats

if 'panopticapi' not in os.environ['PYTHONPATH']:
    os.environ['PYTHONPATH'] += ':' + os.path.abspath(os.path.expanduser('./instanceseg/ext'))
    os.environ['PYTHONPATH'] += ':' + os.path.abspath(os.path.expanduser('./instanceseg/ext/panopticapi'))

# logdir = '../old_instanceseg/scripts/logs/cityscapes/train_instances_filtered_2019-05-14' \
#          '-133452_VCS-1e74989_SAMPLER-car_2_4_BACKBONE-resnet50_ITR-1000000_NPER-4_SSET-car_person'
# logdir = 'scripts/logs/synthetic/train_instances_filtered_2019-06-24-163353_VCS-8df0680'

DEFAULT_GPU = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', help='directory with model path')
    parser.add_argument('--test_split', default='val', help='train, val, test, or any other split the dataloader can '
                                                            'load for this dataset')
    parser.add_argument('--dataset_name', default=None, help='dataset; default=dataset you trained on')
    parser.add_argument('--gpu', '-g', default=(DEFAULT_GPU,), type=int, nargs='+',
                        help='dataset; default=dataset you trained on')
    parser.add_argument('--sampler', default=None)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for the dataloader of the designated '
                                                                  'test split')
    return parser.parse_args()


def main(logdir, test_split, sampler=None, gpu=(DEFAULT_GPU,), batch_size=1, dataset_name=None):
    dataset_name = dataset_name or os.path.basename(os.path.dirname(logdir))
    replacement_dict_for_sys_args = [dataset_name, '--logdir', logdir, '--{}_batch_size'.format(test_split),
                                     str(batch_size), '-g', ' '.join(str(g) for g in gpu), '--test_split', test_split,
                                     '--sampler', sampler]
    # Test
    np.random.seed(100)
    predictions_outdir, groundtruth_outdir, tester, logdir = test.main(replacement_dict_for_sys_args)

    # Convert
    out_dirs_root = convert_test_results_to_coco.get_outdirs_cache_root(logdir, predictions_outdir)
    problem_config = tester.exporter.instance_problem.load(tester.exporter.instance_problem_path)
    out_jsons, out_dirs = convert_test_results_to_coco.main(predictions_outdir, groundtruth_outdir, problem_config,
                                                            out_dirs_root)

    # Evaluate
    collated_stats_per_image_file = evaluate.main(out_jsons['gt'], out_jsons['pred'], out_dirs['gt'],
                                                  out_dirs['pred'], problem_config)

    visualize_pq_stats.main(collated_stats_per_image_file)

    return collated_stats_per_image_file


if __name__ == '__main__':
    args = parse_args()

    collated_stats_per_image_per_cat_file = main(**args.__dict__)

    pass
