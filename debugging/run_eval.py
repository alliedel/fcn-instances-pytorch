import glob

import numpy as np
import os
from PIL import Image

import shutil
from instanceseg.ext.panopticapi import evaluation
from instanceseg.ext.panopticapi.utils import IdGenerator, save_json, rgb2id
# from cityscapesscripts.helpers.labels import labels, id2label
from instanceseg.utils import instance_utils
from instanceseg.utils import parse
from scripts import evaluate

if 'panopticapi' not in os.environ['PYTHONPATH']:
    os.environ['PYTHONPATH'] += ':' + os.path.abspath(os.path.expanduser('./instanceseg/ext'))
    os.environ['PYTHONPATH'] += ':' + os.path.abspath(os.path.expanduser('./instanceseg/ext/panopticapi'))


def panoptic_converter_from_rgb_ids(out_folder, out_json_file, labels_file_list,
                                    problem_config: instance_utils.InstanceProblemConfig):
    # Replace split with file_list
    labels_table = problem_config.labels_table
    if not os.path.isdir(out_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(out_folder))
        os.mkdir(out_folder)

    categories_dict = {cat.id: cat for cat in labels_table}
    images = []
    annotations = []
    n_files = len(labels_file_list)
    for working_idx, label_f in enumerate(labels_file_list):
        if working_idx % 10 == 0:
            print(working_idx, n_files)

        rgb_format = np.array(Image.open(label_f), dtype=np.uint8)

        assert len(rgb_format.shape) == 3, 'Image should be in rgb format'

        file_name = label_f.split('/')[-1]
        image_id = file_name.rsplit('_', 2)[1]
        image_filename = '{}_image.png'.format(image_id)
        # image entry, id for image is its filename without extension
        images.append({"id": image_id,
                       "width": rgb_format.shape[1],
                       "height": rgb_format.shape[0],
                       "file_name": image_filename})

        id_generator = IdGenerator(categories_dict)

        idx = 0
        present_channel_colors = np.unique(rgb_format.reshape(-1, rgb_format.shape[2]), axis=0)
        pan_format = np.zeros((rgb_format.shape[0], rgb_format.shape[1], 3), dtype=np.uint8)
        segm_info = []
        pan_ids = np.zeros((rgb_format.shape[0], rgb_format.shape[1]))
        for color_val in present_channel_colors:
            channel_idx = rgb2id(color_val)
            semantic_id = problem_config.semantic_instance_class_list[channel_idx]
            instance_count_id = problem_config.instance_count_id_list[channel_idx]
            is_crowd = instance_count_id < 1
            if semantic_id not in categories_dict:
                continue
            if categories_dict[semantic_id]['isthing'] == 0:
                is_crowd = 0
            mask = (rgb_format == color_val).all(axis=2)
            area, bbox = get_bounding_box(mask)

            segment_id, pan_format_color = id_generator.get_id_and_color(semantic_id)
            pan_format[mask, :] = pan_format_color
            pan_ids[mask] = segment_id

            segm_info.append({"id": int(segment_id),
                              "category_id": int(semantic_id),
                              "area": area,
                              "bbox": bbox,
                              "iscrowd": is_crowd})
        Image.fromarray(pan_format).save(os.path.join(out_folder, file_name))
        # Reverse the process and ensure we get the right id
        reloaded_pan_img = np.array(Image.open(os.path.join(out_folder, file_name)), dtype=np.uint32)
        reloaded_pan_id = rgb2id(reloaded_pan_img)
        assert np.all(reloaded_pan_id == pan_ids)
        if len(segm_info) == 0:
            raise Exception('No segments in this image')

        annotations.append({'image_id': image_id,
                            'file_name': file_name,
                            "segments_info": segm_info})

        shutil.copy(label_f, os.path.join(out_folder, file_name))

        assert len(segm_info) == len(present_channel_colors)

    d = {
        'images': images,
        'annotations': annotations,
        'categories': [l.__dict__ for l in labels_table],
    }

    save_json(d, out_json_file)


def get_bounding_box(mask):
    area = np.sum(mask)  # segment area computation
    # bbox computation for a segment
    hor = np.sum(mask, axis=0)
    hor_idx = np.nonzero(hor)[0]
    x = hor_idx[0]
    width = hor_idx[-1] - x + 1
    vert = np.sum(mask, axis=1)
    vert_idx = np.nonzero(vert)[0]
    y = vert_idx[0]
    height = vert_idx[-1] - y + 1
    bbox = [x, y, width, height]
    return area, bbox


def check_annotation(out_dirs, gt_annotation, pred_annotation, categories, VOID=0):
    pan_gt = np.array(Image.open(os.path.join(out_dirs['gt'], gt_annotation['file_name'])), dtype=np.uint32)
    pan_gt = rgb2id(pan_gt)
    pan_pred = np.array(Image.open(os.path.join(out_dirs['pred'], pred_annotation['file_name'])), dtype=np.uint32)
    pan_pred = rgb2id(pan_pred)

    gt_segms = {el['id']: el for el in gt_annotation['segments_info']}
    pred_segms = {el['id']: el for el in pred_annotation['segments_info']}

    # predicted segments area calculation + prediction sanity checks
    pred_labels_set = set(el['id'] for el in pred_annotation['segments_info'])
    labels, labels_cnt = np.unique(pan_pred, return_counts=True)
    for label, label_cnt in zip(labels, labels_cnt):
        if label not in pred_segms:
            if label == VOID:
                continue
            raise KeyError(
                'In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(
                    gt_annotation['image_id'], label))
        pred_segms[label]['area'] = label_cnt
        pred_labels_set.remove(label)
        if pred_segms[label]['category_id'] not in categories:
            raise KeyError(
                'In the image with ID {} segment with ID {} has unknown category_id {}.'.format(
                    gt_annotation['image_id'], label, pred_segms[label]['category_id']))
    if len(pred_labels_set) != 0:
        raise KeyError(
            'In the image with ID {} the following segment IDs {} are presented in JSON and not presented in '
            'PNG.'.format(gt_annotation['image_id'], list(pred_labels_set)))


def main():
    checkpoint_path = '/usr0/home/adelgior/code/experimental-instanceseg/scripts/logs/synthetic/' \
                      'train_instances_filtered_2019-06-24-163353_VCS-8df0680'
    dataset_name = 'synthetic'
    config = dict(dataset_name=dataset_name,
                  resume=checkpoint_path,
                  gpu=1,
                  config_idx=0,
                  sampler_name=None)

    commandline_arguments_list = parse.construct_args_list_to_replace_sys(**config)

    predictions_outdir, groundtruth_outdir, evaluator = evaluate.main(commandline_arguments_list)

    problem_config = evaluator.instance_problem

    experiment_identifier = os.path.basename(checkpoint_path)
    out_dirs = {}
    out_jsons = {}
    out_dirs_root = os.path.join('cache', '{}'.format(dataset_name), experiment_identifier)
    if not os.path.exists(out_dirs_root):
        os.makedirs(out_dirs_root)
    else:
        print(Warning('{} already exists'.format(out_dirs_root)))
    for in_dir, label_type in zip([predictions_outdir, groundtruth_outdir], ('pred', 'gt')):
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
        print('hey')
    categories = {el['id']: el for el in gt_json['categories']}
    check_annotation(out_dirs, gt_annotations[0], pred_annotations[0], categories)

    print('evaluating from {}, {}'.format(out_jsons['gt'], out_jsons['pred']))
    print('evaluating from {}, {}'.format(out_dirs['gt'], out_dirs['pred']))
    evaluation.pq_compute(out_jsons['gt'], out_jsons['pred'], gt_folder=out_dirs['gt'], pred_folder=out_dirs['pred'])


if __name__ == '__main__':
    main()
