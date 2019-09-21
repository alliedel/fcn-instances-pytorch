import numpy as np
import os
import tqdm
from PIL import Image

from instanceseg.ext.panopticapi.utils import IdGenerator, rgb2id, id2rgb, save_json
from instanceseg.panoeval.utils import get_bounding_box
from instanceseg.utils.misc import y_or_n_input


def upsnet_panoptic_converter(out_folder, out_json_file, labels_file_list, labels_table,
                              VOID_RGB=(255, 255, 255), VOID_INSTANCE_G=255, overwrite=None):
    """
    Takes predictions output from tester (special Trainer type) and outputs coco panoptic format
    Inputs should represent the different channels, where rgb2id creates the channel id (R + G*255 + B*255*255)
    """
    # Replace split with file_list
    if not os.path.isdir(out_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(out_folder))
        os.mkdir(out_folder)

    categories_dict = {cat.id: cat for cat in labels_table}
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
        import ipdb; ipdb.set_trace()
        rgb_format = np.array(Image.open(label_f), dtype=np.uint8)

        assert len(rgb_format.shape) == 3, 'Image should be in rgb format'

        file_name = label_f.split('/')[-1]
        out_file_name = file_name.replace('.png', cocopano_ext)
        if os.path.exists(out_file_name):
            if not overwrite:
                continue
        image_id = file_name.rsplit('_', 1)[0]
        image_filename = '{}_image.png'.format(image_id)
        # image entry, id for image is its filename without extension
        images.append({"id": image_id,
                       "width": rgb_format.shape[1],
                       "height": rgb_format.shape[0],
                       "file_name": image_filename})

        id_generator = IdGenerator(categories_dict)

        idx = 0
        present_channel_colors = np.unique(rgb_format.reshape(-1, rgb_format.shape[2]), axis=0)
        present_channel_colors = [c for c in present_channel_colors if rgb2id(c) != rgb2id(VOID_RGB)
                                  and c[1] != VOID_INSTANCE_G]
        present_id_vals = [rgb2id(color_val) for color_val in present_channel_colors]

        pan_format = np.zeros((rgb_format.shape[0], rgb_format.shape[1], 3), dtype=np.uint8)
        segm_info = []
        pan_ids = np.zeros((rgb_format.shape[0], rgb_format.shape[1]))
        for color_val, inst_id in zip(present_channel_colors, present_id_vals):
            semantic_id = color_val[0]
            instance_count_id = color_val[1]
            is_crowd = instance_count_id < 1
            if semantic_id not in categories_dict:
                continue
            if categories_dict[semantic_id]['isthing'] == 0:
                is_crowd = 0
            mask = (rgb_format == color_val).all(axis=2)
            area, bbox = get_bounding_box(mask)

            segment_id = semantic_id * 1000 + instance_count_id
            pan_color = id2rgb(segment_id)
            pan_format[mask, :] = pan_color
            pan_ids[mask] = segment_id

            segm_info.append({"id": int(segment_id),
                              "category_id": int(semantic_id),
                              "area": area,
                              "bbox": bbox,
                              "iscrowd": is_crowd})
        assert len(segm_info) == len(present_channel_colors)

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

    d = {
        'images': images,
        'annotations': annotations,
        'categories': [l.__dict__ for l in labels_table],
    }

    save_json(d, out_json_file)