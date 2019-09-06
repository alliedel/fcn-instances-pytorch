from debugging.coco_panoptic_converter import upsnet_panoptic_converter
from instanceseg.datasets import labels_table_cityscapes
from instanceseg.panoeval.utils import check_annotation
from instanceseg.utils import instance_utils
import pathlib
import os
import glob
import json

from scripts import evaluate
from scripts.visualizations import visualize_pq_stats, export_prediction_vs_gt_vis_sorted


def converter_main_upsnet(predictions_dir, groundtruth_dir, problem_config, out_dirs_root, overwrite=False):
    out_dirs = {}
    out_jsons = {}
    if not os.path.exists(out_dirs_root):
        os.makedirs(out_dirs_root)
    else:
        print(Warning('{} already exists'.format(out_dirs_root)))
    indirs = {'gt': groundtruth_dir, 'pred': predictions_dir}
    for label_type, in_dir in indirs.items():
        dir_basename = 'panoptic_conversion_{}'.format(label_type)
        out_dir = os.path.join(out_dirs_root, dir_basename)
        file_list = sorted(glob.glob(os.path.join(in_dir, '*.png')))
        out_file = os.path.abspath(os.path.expanduser(os.path.join(out_dir, '..', dir_basename + '.json')))
        upsnet_panoptic_converter(out_dir, out_file, labels_file_list=file_list,
                                  labels_table=problem_config.labels_table, overwrite=overwrite)
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


def get_labels_table_without_voids(original_labels_table, void_trainids=(255,-1)):
    return [l for l in original_labels_table if l['id'] not in void_trainids]


def upsnet_evaluate(upsnet_log_directory='/home/adelgior/code/upsnet/output/upsnet/'
                                         'cityscapes/upsnet_resnet50_cityscapes_4gpu/',
                    gt_folder='/home/adelgior/code/upsnet/data/cityscapes/panoptic', split='val',
                    eval_iou_threshold=0.5, overwrite=False):
    upsnet_results_directory = upsnet_log_directory + split + '/results/'
    upsnet_pano_path = pathlib.Path(upsnet_results_directory, 'pans_unified')
    gt_json_file = os.path.join(upsnet_pano_path, 'gt.json')
    old_pred_json_file = upsnet_pano_path / 'pred.json'
    pred_folder = upsnet_pano_path / 'pan'
    new_pred_json_file = pathlib.Path(str(old_pred_json_file).replace('.json', 'with_image_ids.json'))

    labels_table = get_labels_table_without_voids(
        labels_table_cityscapes.get_cityscapes_trainids_label_table_cocoform(void_trainids=(-1, 255)))

    problem_config = instance_utils.InstanceProblemConfig(n_instances_by_semantic_id=[None for _ in labels_table],
                                                          labels_table=labels_table)

    # Fix json format
    gt_json_format = json.load(open(gt_json_file, 'rb'))
    image_ids = [l['id'] for l in gt_json_format['images']]
    file_names = [l['file_name'] for l in gt_json_format['images']]
    pred_json_format = json.load(open(old_pred_json_file, 'rb'))
    assert len(image_ids) == len(pred_json_format['annotations']), \
        '{} image ids; {} annotations'.format(len(image_ids), len(pred_json_format['annotations']))
    assert os.path.join(pred_folder)
    for i in range(len(pred_json_format['annotations'])):
        pred_json_format['annotations'][i]['image_id'] = image_ids[i]
        pred_json_format['annotations'][i]['file_name'] = file_names[i].replace('_leftImg8bit', '')
    json.dump(pred_json_format, new_pred_json_file.open('w'))

    eval_pq_npz_file = evaluate.main_unwrapped(gt_json_file, new_pred_json_file, gt_folder, pred_folder, problem_config,
                                               iou_threshold=eval_iou_threshold, overwrite=overwrite)
    return eval_pq_npz_file


def main(visualize_pq_hists=True, export_sorted_perf_images=None, eval_iou_threshold=0.5):
    eval_pq_npz_file = upsnet_evaluate(eval_iou_threshold=eval_iou_threshold)

    export_sorted_perf_images = export_sorted_perf_images if export_sorted_perf_images is not None else \
        visualize_pq_hists

    if export_sorted_perf_images:
        export_prediction_vs_gt_vis_sorted.main(collated_stats_npz=eval_pq_npz_file)

    if visualize_pq_hists:
        visualize_pq_stats.main(eval_pq_npz_file)


if __name__ == '__main__':
    main()
