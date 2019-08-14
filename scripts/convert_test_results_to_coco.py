import os
from instanceseg.panoeval.utils import panoptic_converter_from_rgb_ids, check_annotation
from instanceseg.utils import instance_utils
import glob
import argparse
import yaml


def main(predictions_dir, groundtruth_dir, problem_config, out_dirs_root, overwrite=False):
    out_dirs = {}
    out_jsons = {}
    if not os.path.exists(out_dirs_root):
        os.makedirs(out_dirs_root)
    else:
        print(Warning('{} already exists'.format(out_dirs_root)))
    for in_dir, label_type in zip([predictions_dir, groundtruth_dir], ('pred', 'gt')):
        dir_basename = 'panoptic_conversion_{}'.format(label_type)
        out_dir = os.path.join(out_dirs_root, dir_basename)
        file_list = sorted(glob.glob(os.path.join(in_dir, '*.png')))
        out_file = os.path.abspath(os.path.expanduser(os.path.join(out_dir, '..', dir_basename + '.json')))
        panoptic_converter_from_rgb_ids(out_dir, out_file, labels_file_list=file_list, problem_config=problem_config,
                                        overwrite=overwrite)
        out_dirs[label_type] = out_dir
        out_jsons[label_type] = out_file

    # Sanity check
    with open(out_jsons['gt'], 'r') as f:
        import json
        gt_json = json.load(f)
        gt_annotations = gt_json['annotations']
    with open(out_jsons['pred'], 'r') as f:
        pred_json = json.load(f)
        pred_annotations = pred_json['annotations']
    categories = {el['id']: el for el in gt_json['categories']}
    check_annotation(out_dirs, gt_annotations[0], pred_annotations[0], categories)
    print('Conversion successful.  Predictions and groundtruth written to {}'.format([v for k, v in out_dirs.items()]))
    return out_jsons, out_dirs


def get_paths(test_outdir):
    return {
        'pred_dir': os.path.join(test_outdir, 'predictions'),
        'gt_dir': os.path.join(test_outdir, 'groundtruth'),
        'scores_dir': os.path.join(test_outdir, 'scores'),
        'problem_config_file': os.path.join(test_outdir, 'instance_problem_config.yaml')
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_logdir', type=str, help='For test parent directory')
    args_ = parser.parse_args()
    return args_


def get_outdirs_cache_root(train_logdir, test_pred_outdir):
    test_logdir = os.path.dirname(test_pred_outdir)
    train_logdir, test_pred_outdir = train_logdir.rstrip('/'), test_pred_outdir.rstrip('/')
    dataset_name_from_logdir = os.path.basename(os.path.dirname(train_logdir))
    train_exp_id = os.path.basename(train_logdir)
    test_exp_id = os.path.basename(test_logdir)  # Assuming .../<expid>/'predictions'
    assert test_exp_id is not '', ValueError('test_pred_outdir not in form I thought')
    assert train_exp_id is not '', ValueError('train_pred_outdir not in form I thought')
    out_dirs_root = os.path.join('cache', '{}'.format(dataset_name_from_logdir), train_exp_id, test_exp_id)
    return out_dirs_root


def get_cache_dir_from_test_logdir(test_logdir):
    with open(os.path.join(test_logdir, 'train_logdir.txt'), 'r') as fid:
        train_logdir = fid.read().strip()
    test_pred_outdir = os.path.join(test_logdir, 'predictions')
    return get_outdirs_cache_root(train_logdir, test_pred_outdir)


if __name__ == '__main__':
    args = parse_args()
    files = get_paths(args.test_logdir)
    out_dirs_root = get_cache_dir_from_test_logdir(args.test_logdir)
    problem_config = instance_utils.InstanceProblemConfig.load(files['problem_config_file'])
    main(files['pred_dir'], files['gt_dir'], problem_config, out_dirs_root)
