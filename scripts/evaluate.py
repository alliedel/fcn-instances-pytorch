import argparse

import numpy as np
import os
import os.path as osp

from instanceseg.panoeval import compute
from instanceseg.panoeval.utils import collate_pq_into_pq_compute_per_imageNxS
from instanceseg.utils import instance_utils

here = osp.dirname(osp.abspath(__file__))


def main(gt_json_file, pred_json_file, gt_folder, pred_folder, problem_config):
    # checkpoint_path = '/usr0/home/adelgior/code/experimental-instanceseg/scripts/logs/synthetic/' \
    #                   'train_instances_filtered_2019-06-24-163353_VCS-8df0680'
    print('evaluating from {}, {}'.format(gt_json_file, pred_json_file))
    print('evaluating from {}, {}'.format(gt_folder, pred_folder))
    out_dirs_root = os.path.dirname(pred_json_file)
    class_avgs_per_image = compute.pq_compute_per_image(gt_json_file=gt_json_file, pred_json_file=pred_json_file,
                                                        gt_folder=gt_folder, pred_folder=pred_folder)
    # isthing = problem_config.has_instances
    categories = problem_config.semantic_vals
    collated_stats_per_image_per_cat = collate_pq_into_pq_compute_per_imageNxS(class_avgs_per_image, categories)
    # for semantic_id in results['per_class'].keys():
    #     results['per_class'][semantic_id]['name'] = problem_config.semantic_class_names[int(semantic_id)]
    # print(results['per_class'])

    outfile_collated = os.path.join(out_dirs_root, 'collated_stats_per_img_per_cat.npz')
    # Saving more than I need in case I need it for reproducibility later
    np.savez(outfile_collated,
             collated_stats_per_image_per_cat=collated_stats_per_image_per_cat,
             categories=categories, problem_config=problem_config, gt_json_file=gt_json_file,
             pred_json_file=pred_json_file, gt_folder=gt_folder, pred_folder=pred_folder)
    print('Stats (and categories/problem config) saved to {}'.format(outfile_collated))

    return outfile_collated


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_config_file', type=str,
                        help="JSON file with ground truth data", required=True)
    parser.add_argument('--gt_json_file', type=str, default=None,
                        help="JSON file with ground truth data")
    parser.add_argument('--pred_json_file', type=str, default=None,
                        help="JSON file with predictions data")
    parser.add_argument('--gt_folder', type=str, default=None,
                        help="Folder with ground truth COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--pred_folder', type=str, default=None,
                        help="Folder with prediction COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    problem_config = instance_utils.InstanceProblemConfig.load(args.problem_config_file)
    collated_stats_per_image_per_cat_file = main(gt_json_file=args.gt_json_file,
                                                 pred_json_file=args.pred_json_file,
                                                 gt_folder=args.gt_folder, pred_folder=args.pred_folder,
                                                 problem_config=problem_config)
