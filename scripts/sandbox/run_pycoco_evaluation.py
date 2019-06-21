import os
if 'panopticapi' not in os.environ['PYTHONPATH']:
    os.environ['PYTHONPATH'] += ':' + os.path.abspath(os.path.expanduser('./instanceseg/ext'))
    os.environ['PYTHONPATH'] += ':' + os.path.abspath(os.path.expanduser('./instanceseg/ext/panopticapi'))

import json
import shutil
from instanceseg.ext.panopticapi import evaluation


# python converters/2channels2panoptic_coco_format.py \
#   --source_folder sample_data/panoptic_examples_2ch_format \
#   --images_json_file sample_data/images_info_examples.json \
#   --prediction_json_file converted_data/panoptic_coco_from_2ch.json


def make_short_json_example(split='val'):
    # Load input
    converted_cityscapes_root = os.path.expanduser('~/data/datasets/cityscapes_data')
    gt_folder = os.path.join(converted_cityscapes_root, 'cityscapes_panoptic_{}/'.format(split))
    pred_folder = gt_folder
    gt_json_file = os.path.join(converted_cityscapes_root, 'cityscapes_panoptic_{}.json'.format(split))
    json_file_example = gt_json_file
    json_data_example = json.load(open(json_file_example, 'r'))

    # Create shortened version
    json_data_example_shortened = {
        'images': json_data_example['images'][:3],
        'annotations': json_data_example['annotations'][:3],
        'categories': json_data_example['categories']
    }

    # Get output filenames
    shortened_out_directory = '/tmp/short_cityscapes_panoptic_{}'.format(split)
    if not os.path.isdir(shortened_out_directory):
        os.makedirs(shortened_out_directory)
    gt_shortened_nm, pred_shortened_nm = (os.path.join(shortened_out_directory, 'short_example_{}_{}'.format(split, x))
                                          for x in ('gt', 'pred'))
    gt_folder_shortened, pred_folder_shortened = gt_shortened_nm, pred_shortened_nm
    gt_json_file_shortened, pred_json_file_shortened = gt_folder_shortened + '.json', pred_folder_shortened + '.json'

    # Generate output
    # GT

    out_folder = gt_folder_shortened
    in_folder = gt_folder
    out_json_file = gt_json_file_shortened
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    json.dump(json_data_example_shortened, open(out_json_file, 'w'))
    for image in json_data_example_shortened['annotations']:
        shutil.copyfile(os.path.join(in_folder, image['file_name']), os.path.join(out_folder, image['file_name']))

    # pred
    out_folder = pred_folder_shortened
    in_folder = pred_folder
    out_json_file = pred_json_file_shortened
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    json.dump(json_data_example_shortened, open(out_json_file, 'w'))
    for image in json_data_example_shortened['annotations']:
        shutil.copyfile(os.path.join(in_folder, image['file_name']), os.path.join(out_folder, image['file_name']))

    return gt_json_file_shortened, pred_json_file_shortened, gt_folder_shortened, pred_folder_shortened


def main():
    split = 'val'
    gt_json_file, pred_json_file, gt_folder, pred_folder = make_short_json_example(split=split)
    # json.dump(json_data, pred_json_file)
    evaluation.pq_compute(gt_json_file, pred_json_file, gt_folder=gt_folder, pred_folder=pred_folder)


if __name__ == '__main__':
    main()
