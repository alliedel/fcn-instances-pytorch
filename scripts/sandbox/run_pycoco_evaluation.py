import os
if 'panopticapi' not in os.environ['PYTHONPATH']:
    os.environ['PYTHONPATH'] += ':' + os.path.abspath(os.path.expanduser('./instanceseg/ext'))
    os.environ['PYTHONPATH'] += ':' + os.path.abspath(os.path.expanduser('./instanceseg/ext/panopticapi'))

import json

from panopticapi import evaluation


# python converters/2channels2panoptic_coco_format.py \
#   --source_folder sample_data/panoptic_examples_2ch_format \
#   --images_json_file sample_data/images_info_examples.json \
#   --prediction_json_file converted_data/panoptic_coco_from_2ch.json


def main():
    split = 'val'
    converted_cityscapes_root = os.path.expanduser('~/data/datasets/cityscapes_data')
    pred_json_file = '/tmp/pred_json_file.json'
    ground_truth_directory = os.path.join(converted_cityscapes_root, 'cityscapes_panoptic_{}/'.format(split))
    gt_json_file = os.path.join(converted_cityscapes_root, 'cityscapes_panoptic_{}.json'.format(split))
    json_file_example = gt_json_file
    json_data_example = json.load(open(json_file_example, 'r'))

    pred_json_file = gt_json_file
    pred_folder = gt_folder

    # json.dump(json_data, pred_json_file)
    evaluation.pq_compute(gt_json_file, pred_json_file, gt_folder=gt_folder, pred_folder=pred_folder)
    #
    # pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    # matched_annotations_list = []
    # for gt_ann in gt_json['annotations']:
    #     image_id = gt_ann['image_id']
    #     if image_id not in pred_annotations:
    #         raise Exception('no prediction for the image with id: {}'.format(image_id))
    #     matched_annotations_list.append((gt_ann, pred_annotations[image_id]))
    #
    # pq_stat = evaluation.pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories)
    #


if __name__ == '__main__':
    main()
