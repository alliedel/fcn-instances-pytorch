import numpy as np
import os
import tqdm
from PIL import Image

from instanceseg.ext.panopticapi.utils import IdGenerator, rgb2id, id2rgb, save_json
from instanceseg.utils import instance_utils
from instanceseg.utils.misc import y_or_n_input


# TODO(allie): Make converter into multiprocessing pool so we can convert the files much more quickly.


def panoptic_converter_from_rgb_ids(out_folder, out_json_file, labels_file_list,
                                    problem_config: instance_utils.InstanceProblemConfig, labels_table=None,
                                    VOID_RGB=(255, 255, 255), VOID_INSTANCE_G=255, overwrite=None):
    """
    Takes predictions output from tester (special Trainer type) and outputs coco panoptic format
    Inputs should represent the different channels, where rgb2id creates the channel id (R + G*255 + B*255*255)
    """
    # Replace split with file_list
    labels_table = labels_table or problem_config.labels_table
    if not os.path.isdir(out_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(out_folder))
        os.mkdir(out_folder)

    categories_dict = {cat.id: cat for cat in labels_table}
    import ipdb; ipdb.set_trace()
    images = []
    annotations = []
    n_files = len(labels_file_list)
    cocopano_ext = '_cocopano.png'

    all_files_already_exist = os.path.exists(out_json_file)
    file_exists = []
    for working_idx, label_f in tqdm.tqdm(enumerate(labels_file_list),
                                          desc='Finding files', total=n_files):
        file_name = label_f.split('/')[-1]
        out_file_name = file_name.replace('.png', cocopano_ext)
        file_exists.append(os.path.exists(os.path.join(out_folder, out_file_name)))
    all_files_already_exist = all_files_already_exist and all(file_exists)
    some_files_already_exist = any(file_exists)

    if all_files_already_exist:
        if overwrite is None:
            y_or_n = y_or_n_input('All files already exist.  Overwrite?')
            if y_or_n == 'n':
                return
        elif overwrite is False:
            print('All files already exist.')
            return
    elif some_files_already_exist:
        print('Warning: some ({}/{}) files already existed.  I may be overwriting them.'.format(
            sum(file_exists), len(file_exists)))

    for working_idx, label_f in tqdm.tqdm(enumerate(labels_file_list),
                                          desc='Converting to COCO panoptic format', total=n_files):
        rgb_format = np.array(Image.open(label_f), dtype=np.uint8)

        assert len(rgb_format.shape) == 3, 'Image should be in rgb format'

        file_name = label_f.split('/')[-1]
        out_file_name = file_name.replace('.png', cocopano_ext)
        if os.path.exists(out_file_name):
            if not overwrite:
                continue
        assert file_name.rsplit('_', 2)[2] == 'sem255instid2rgb.png'
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
        present_channel_colors = [c for c in present_channel_colors if rgb2id(c) != rgb2id(VOID_RGB) and c[1] !=
                                  VOID_INSTANCE_G]

        pan_format = np.zeros((rgb_format.shape[0], rgb_format.shape[1], 3), dtype=np.uint8)
        segm_info = []
        pan_ids = np.zeros((rgb_format.shape[0], rgb_format.shape[1]))
        semantic_ids_not_in_category_dict = []
        unique_segm_ids = []
        for color_val in present_channel_colors:
            semantic_id = color_val[0]
            instance_count_id = color_val[1]
            is_crowd = instance_count_id < 1
            if semantic_id not in categories_dict:
                if semantic_id not in semantic_ids_not_in_category_dict:
                    semantic_ids_not_in_category_dict.append(semantic_id)
                continue
            if categories_dict[semantic_id]['isthing'] == 0:
                is_crowd = 0
            mask = (rgb_format == color_val).all(axis=2)
            area, bbox = get_bounding_box(mask)

            segment_id = semantic_id * 1000 + instance_count_id
            pan_color = id2rgb(segment_id)
            pan_format[mask, :] = pan_color
            pan_ids[mask] = segment_id
            assert segment_id not in unique_segm_ids  # every segment should be unique
            unique_segm_ids.append(segment_id)
            segm_info.append({"id": int(segment_id),
                              "category_id": int(semantic_id),
                              "area": area,
                              "bbox": bbox,
                              "iscrowd": is_crowd})
        if len(semantic_ids_not_in_category_dict) > 0:
            print('The following semantic ids were present in the image, but not in the categories dict ({catdict}): '
                  '{semids}'.format(catdict=categories_dict, semids=semantic_ids_not_in_category_dict))
        Image.fromarray(pan_format).save(os.path.join(out_folder, out_file_name))
        # Reverse the process and ensure we get the right id
        reloaded_pan_img = np.array(Image.open(os.path.join(out_folder, out_file_name)), dtype=np.uint32)
        reloaded_pan_id = rgb2id(reloaded_pan_img)
        assert np.all(reloaded_pan_id == pan_ids)
        # print('Max pan id: {}'.format(reloaded_pan_id.max()))
        if len(segm_info) == 0:
            raise Exception('No segments in this image')

        annotations.append({'image_id': image_id,
                            'file_name': out_file_name,
                            "segments_info": segm_info})

        # shutil.copy(label_f, os.path.join(out_folder, file_name))

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


def check_annotation(out_dirs, gt_annotation, pred_annotation, categories, VOID=(0, rgb2id((255, 255, 255)))):
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


def collate_pq_into_pq_compute_per_imageNxS(class_avgs_per_image, categories):
    labels = list(class_avgs_per_image[0].keys())
    metric_names = class_avgs_per_image[0][labels[0]].keys()
    assert set(labels) == set(categories), 'labels: {}\n categories: {}'.format(labels, categories)
    n_categories = len(labels)
    n_images = len(class_avgs_per_image)
    empty_stats_per_image = np.zeros((n_images, n_categories))
    collated_stats_per_image_per_cat = {
        metric: empty_stats_per_image.copy() for metric in metric_names
    }
    for i in range(n_images):
        for j, (category, category_metrics) in enumerate(class_avgs_per_image[i].items()):
            for key, array in collated_stats_per_image_per_cat.items():
                collated_stats_per_image_per_cat[key][i, j] = category_metrics[key]
    return collated_stats_per_image_per_cat
