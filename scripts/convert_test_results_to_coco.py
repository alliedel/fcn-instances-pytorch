import os
from instanceseg.panoeval.utils import panoptic_converter_from_rgb_ids, check_annotation
import glob
import argparse


def main(predictions_dir, groundtruth_dir, problem_config, out_dirs_root):
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
        panoptic_converter_from_rgb_ids(out_dir, out_file, labels_file_list=file_list, problem_config=problem_config)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='For choosing correct directory name out')
    parser.add_argument('--pred_dir', type=str, help='Directory path for predictions from testing')
    parser.add_argument('--gt_dir', type=str, help='Directory path for matched groundtruth from testing')
    parser.add_argument('--problem_config_file', type=str, help='Path to problem config file (.yaml)')

    return parser.parse_args()


def get_outdirs_cache_root(train_logdir, test_pred_outdir):
    dataset_name_from_logdir = os.path.split(os.path.split(train_logdir)[0])[1]
    train_exp_id = os.path.basename(train_logdir)
    test_exp_id = os.path.basename(os.path.dirname(test_pred_outdir))  # Assuming .../<expid>/'predictions'
    out_dirs_root = os.path.join('cache', '{}'.format(dataset_name_from_logdir), train_exp_id, test_exp_id)
    return out_dirs_root


if __name__ == '__main__':
    args = parse_args()
    out_dirs_root = get_outdirs_cache_root(args.logdir, args.pred_dir)
    main(args.pred_dir, args.gt_dir, args.problem_config, out_dirs_root)
