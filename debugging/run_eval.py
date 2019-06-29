import glob

import numpy as np
import os
from PIL import Image

from instanceseg.ext.panopticapi import evaluation
from instanceseg.ext.panopticapi.utils import IdGenerator, save_json
# from cityscapesscripts.helpers.labels import labels, id2label
from instanceseg.utils import instance_utils
from instanceseg.utils import parse
from scripts import evaluate

if 'panopticapi' not in os.environ['PYTHONPATH']:
    os.environ['PYTHONPATH'] += ':' + os.path.abspath(os.path.expanduser('./instanceseg/ext'))
    os.environ['PYTHONPATH'] += ':' + os.path.abspath(os.path.expanduser('./instanceseg/ext/panopticapi'))


def panoptic_converter(out_folder, out_json_file, labels_file_list, labels_table,
                       problem_config: instance_utils.InstanceProblemConfig):
    # Replace split with file_list

    if not os.path.isdir(out_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(out_folder))
        os.mkdir(out_folder)

    categories_dict = {cat['id']: cat for cat in labels_table}

    images = []
    annotations = []
    n_files = len(labels_file_list)
    for working_idx, label_f in enumerate(labels_file_list):
        if working_idx % 10 == 0:
            print(working_idx, n_files)

        original_format = np.array(Image.open(label_f))

        file_name = label_f.split('/')[-1]
        image_id = file_name.rsplit('_', 2)[0]
        image_filename = '{}_image.png'.format(image_id)
        # image entry, id for image is its filename without extension
        images.append({"id": image_id,
                       "width": original_format.shape[1],
                       "height": original_format.shape[0],
                       "file_name": image_filename})

        pan_format = np.zeros((original_format.shape[0], original_format.shape[1], 3), dtype=np.uint8)
        id_generator = IdGenerator(categories_dict)

        idx = 0
        present_channel_vals = np.unique(original_format)

        segm_info = []
        for channel_val in present_channel_vals:
            channel_idx = problem_config.channel_values.index(channel_val)  # generally equals channel_val
            semantic_id = \
                problem_config.semantic_class_names[problem_config.semantic_instance_class_list[channel_idx]]
            instance_count_id = problem_config.instance_count_id_list[channel_idx]
            is_crowd = instance_count_id < 1
            if semantic_id not in categories_dict:
                continue
            if categories_dict[semantic_id]['isthing'] == 0:
                is_crowd = 0
            segment_id, color = id_generator.get_id_and_color(semantic_id)
            mask = original_format == channel_val
            area, bbox = get_bounding_box(mask)

            pan_format[mask] = color

            segm_info.append({"id": int(segment_id),
                              "category_id": int(semantic_id),
                              "area": area,
                              "bbox": bbox,
                              "iscrowd": is_crowd})

        annotations.append({'image_id': image_id,
                            'file_name': file_name,
                            "segments_info": segm_info})

        Image.fromarray(pan_format).save(os.path.join(out_folder, file_name))

    d = {
        'images': images,
        'annotations': annotations,
        'categories': labels_table,
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


def get_labels_table(dataset_type, dataset):
    if dataset_type == 'cityscapes':
        raise NotImplementedError
    if dataset_type != 'synthetic':
        raise NotImplementedError


# def get_synthetic_labels_table(dataset_for_labels_table):
#     labels_table = dataset_for_labels_table.labels_table
#     return labels_table
#
#
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

    labels_table = evaluator.dataloaders['test'].dataset.labels_table
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
        panoptic_converter(out_dir, out_file, labels_file_list=file_list, labels_table=labels_table,
                           problem_config=problem_config)
        out_dirs[label_type] = out_dir
        out_jsons[label_type] = out_file

    evaluation.pq_compute(out_jsons['gt'], out_jsons['pred'], gt_folder=out_dirs['gt'], pred_folder=out_dirs['pred'])


if __name__ == '__main__':
    main()
