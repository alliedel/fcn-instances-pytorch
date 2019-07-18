import json
import multiprocessing
import time

import numpy as np
import os

from instanceseg.ext.panopticapi.evaluation import pq_compute_single_core, pq_compute_multi_core


def pq_compute_multi_core_per_image(matched_annotations_list, gt_folder, pred_folder, categories):
    # cpu_num = multiprocessing.cpu_count()
    cpu_num = min(len(matched_annotations_list), multiprocessing.cpu_count()-1)
    # cpu_num = len(matched_annotations_list)

    annotations_split = np.array_split(matched_annotations_list, len(matched_annotations_list))
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, categories))
        processes.append(p)

    pq_stats_per_image = combine_processes_to_stats_per_img(processes)
    assert len(pq_stats_per_image) == len(matched_annotations_list)
    # pq_stat = PQStat()
    # for p in processes:
    #     pq_stat += p.get()

    return pq_stats_per_image


def combine_processes_to_stats_per_img(processes):
    pq_stats = []
    for p in processes:
        pq_stats.append(p.get())
    return pq_stats


def pq_compute_per_image(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):

    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    categories = {el['id']: el for el in gt_json['categories']}

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            raise Exception('no prediction for the image with id: {}'.format(image_id))
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))
    pq_stats_per_image = pq_compute_multi_core_per_image(matched_annotations_list, gt_folder, pred_folder, categories)
    class_avgs_per_image = []
    for i in range(len(pq_stats_per_image)):
        total_avg, class_avg = pq_stats_per_image[i].pq_average(categories, isthing=None)
        class_avgs_per_image.append(class_avg)

    # metrics = [("All", None), ("Things", True), ("Stuff", False)]
    # results = {}
    # for name, isthing in metrics:
    #     results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
    #     if name == 'All':
    #         results['per_class'] = per_class_results
    # print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    # print("-" * (10 + 7 * 4))
    #
    # for name, _isthing in metrics:
    #     print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
    #         name,
    #         100 * results[name]['pq'],
    #         100 * results[name]['sq'],
    #         100 * results[name]['rq'],
    #         results[name]['n'])
    #     )
    #
    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return class_avgs_per_image