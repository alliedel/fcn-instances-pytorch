import argparse
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import json
import os
from PIL import Image
import tqdm

from instanceseg.analysis.visualization_utils import label_colormap, get_tile_image
from instanceseg.ext.panopticapi.utils import rgb2id, id2rgb
from instanceseg.utils import script_setup
from instanceseg.utils.misc import y_or_n_input, symlink
import pathlib

np.random.seed(42)

if os.path.basename(os.path.abspath('.')) == 'sandbox' or os.path.basename(os.path.abspath('.')) == 'scripts':
    os.chdir('../')


def my_hover(event, img_arr, line, fig, xybox, ab, im, x, y):
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        ind, = line.contains(event)[1]["ind"]
        # get the figure size
        w, h = fig.get_size_inches() * fig.dpi
        ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
        hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0] * ws, xybox[1] * hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy = (x[ind], y[ind])
        # set the image corresponding to that point
        im.set_data(img_arr[ind])
    else:
        # if the mouse is not over a scatter point
        ab.set_visible(False)
    fig.canvas.draw_idle()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('collated_stats_npz',
                        type=str,
                        help='Output file from eval; typically collated_stats_per_img_per_cat.npz in '
                             'the test output directory')
    # default='cache/cityscapes/'
    #         'train_2019-07-21-173140_VCS-8c43525_SAMPLER-person_car_2_4_SSET-person_car/'
    #         'test_2019-07-29-121603_VCS-193004c_SAMPLER-car_2_4__test_split-train/'
    #         'collated_stats_per_img_per_cat.npz')
    args = parser.parse_args()
    if not os.path.exists(args.collated_stats_npz):
        collated_stats_npz = os.path.abspath(os.path.join('../', args.collated_stats_npz))
        if os.path.exists(collated_stats_npz):
            args.collated_stats_npz = collated_stats_npz
    assert os.path.exists(args.collated_stats_npz), '{} does not exist'.format(args.collated_stats_npz)
    return args


def cityscapes_trainid_im_to_rgb(panoptic_id_im, labels_table):
    sem_id_im, inst_id_im = np.divmod(panoptic_id_im, 1000)
    out_img = get_sem_inst_rgb_img(sem_id_im, inst_id_im, labels_table)
    return out_img


def panoid_img_to_rgb(coco_id_im, labels_table):
    sem_inst_ch_im = id2rgb(coco_id_im)
    sem_id_im, inst_id_im = sem_inst_ch_im[:, :, 0], sem_inst_ch_im[:, :, 1]
    valid_sem_ids = [l['id'] for l in labels_table]
    assert all(x in valid_sem_ids for x in np.unique(sem_id_im))
    out_img = get_sem_inst_rgb_img(sem_id_im, inst_id_im, labels_table)
    return out_img


def get_sem_inst_rgb_img(sem_id_im, inst_id_im, labels_table):
    inst_label_colormap = label_colormap(255)
    out_sem_img = np.zeros((inst_id_im.shape[0], inst_id_im.shape[1], 3))
    out_inst_img = np.zeros_like(out_sem_img)
    max_inst_id = inst_id_im.max()
    label_table_ids = [l['id'] for l in labels_table]
    for sem_val in np.unique(sem_id_im):
        sem_table_idx = label_table_ids.index(sem_val)
        sem_rgb = labels_table[sem_table_idx]['color']
        out_sem_img[sem_id_im == sem_val, :] = sem_rgb
        for inst_val in np.unique(inst_id_im[sem_id_im == sem_val]):
            out_inst_img[np.bitwise_and(inst_id_im == inst_val, sem_id_im == sem_val), :] = \
                inst_label_colormap[sem_val * max_inst_id + inst_val, :] * 255
    out_img = np.concatenate([out_sem_img, out_inst_img], axis=0)
    return out_img


def load_images(image_dir, file_list, rgb_to_id=False, trainid_to_rgb=False, dtype=np.uint8, labels_table=None):
    image_list = []
    for file_name in tqdm.tqdm(file_list, desc='Loading images from <>/{}'.format(os.path.basename(image_dir)),
                               total=len(file_list)):
        # Hackery.
        possible_extensions = ['_sem255instid2rgb_cocopano.png', '_id2rgb_cocopano.png']
        if not os.path.exists(os.path.join(image_dir, file_name)) and '_gt' in image_dir:
            for ext in possible_extensions:
                new_file_name = 'groundtruth_{}'.format(file_name).replace('_image.png', ext)
                if os.path.join(image_dir, new_file_name):
                    file_name = new_file_name
                    break
            assert os.path.exists(os.path.join(image_dir, file_name)), '{} does not exist'.format(file_name)
        if not os.path.exists(os.path.join(image_dir, file_name)) and '_pred' in image_dir:
            for ext in possible_extensions:
                new_file_name = 'predictions_{}'.format(file_name).replace('_image.png', ext)
                if os.path.join(image_dir, new_file_name):
                    file_name = new_file_name
                    break
        assert os.path.exists(os.path.join(image_dir, file_name)), '{} does not exist'.format(file_name)

        if not os.path.exists(os.path.join(image_dir, file_name)) and 'images' in image_dir:
            file_name = 'image_{}'.format(file_name).replace('_image', '')
            assert rgb_to_id is False
            assert trainid_to_rgb is False
        assert os.path.exists(os.path.join(image_dir, file_name)), '{} does not exist'.format(os.path.join(
            image_dir, file_name))
        with Image.open(os.path.join(image_dir, file_name)) as img:
            pan_im = np.array(img, dtype=dtype)
            if rgb_to_id:
                assert trainid_to_rgb is False
                panoptic_id_im = rgb2id(pan_im)
                im = panoptic_id_im
            elif trainid_to_rgb:
                panoptic_id_im = rgb2id(pan_im)
                im = panoid_img_to_rgb(panoptic_id_im, labels_table)
            else:
                im = pan_im
            image_list.append(im)
    return image_list


def load_image_list(fullpath_filelist, dtype=np.uint8):
    image_list = []
    for file_name in tqdm.tqdm(fullpath_filelist, desc='Loading original images',
                               total=len(fullpath_filelist)):
        assert os.path.exists( file_name), '{} does not exist'.format(file_name)
        with Image.open(file_name) as img:
            im = np.array(img, dtype=dtype)
        image_list.append(im)
    return image_list


def get_stat_data(collated_stats, data_types, semantic_class_names):
    """
    Gets, for instance, ('sq', 'car'): semantic quality of car class
    """
    data_d = {}
    for data_type in data_types:
        if data_type in list(collated_stats.keys()):
            data = collated_stats[data_type]
        elif data_type is not str and data_type[0] in list(collated_stats.keys()):
            assert data_type[1] in semantic_class_names, 'You\'re asking for data_types {} but {} ' \
                                                                        'class doesnt exist in the problem ' \
                                                                        'config'.format(data_types, data_type[1])
            column_idx = semantic_class_names.index(data_type[1])
            data = collated_stats[data_type[0]][:, column_idx]
        else:
            raise ValueError('I dont know how to retrieve {}'.format(data_type))
        data_d[data_type] = data
    return data_d


def extract_n_instances_of_class(collated_stats, class_name, semantic_class_names):
    assert class_name in semantic_class_names, 'You\'re asking for data_types {} but {} ' \
                                                                'class doesnt exist in the problem ' \
                                                                'config'.format('n_inst', class_name)
    column_idx = semantic_class_names.index(class_name)
    data = collated_stats['n_inst'][:, column_idx]
    return data


def get_image_data(collated_stats_dict, img_types, labels_table):
    img_d = {}
    for img_type in img_types:
        if img_type in ('gt_inst_idx', 'pred_inst_idx', 'gt_pano_id_to_rgb', 'pred_pano_id_to_rgb'):
            if img_type == 'gt_inst_idx':
                json_list_file = collated_stats_dict['gt_json_file']
                file_names = [f['file_name'] for f in json.load(open(json_list_file, 'r'))['images']]
                img_d[img_type] = load_images(collated_stats_dict['gt_folder'],
                                              file_names, rgb_to_id=False,
                                              labels_table=labels_table)
            elif img_type == 'pred_inst_idx':
                json_list_file = collated_stats_dict['pred_json_file']
                file_names = [f['file_name'] for f in json.load(open(json_list_file, 'r'))['images']]
                img_d[img_type] = load_images(collated_stats_dict['pred_folder'],
                                              file_names, rgb_to_id=False,
                                              labels_table=labels_table)
            elif img_type == 'gt_pano_id_to_rgb':
                json_list_file = collated_stats_dict['gt_json_file']
                file_names = [f['file_name'] for f in json.load(open(json_list_file, 'r'))['images']]
                img_d[img_type] = load_images(collated_stats_dict['gt_folder'],
                                              file_names, rgb_to_id=False,
                                              labels_table=labels_table)
            elif img_type == 'pred_pano_id_to_rgb':
                json_list_file = collated_stats_dict['gt_json_file']
                file_names = [f['file_name'] for f in json.load(open(json_list_file, 'r'))['images']]
                img_d[img_type] = load_images(collated_stats_dict['gt_folder'],
                                              file_names, rgb_to_id=False,
                                              labels_table=labels_table)
            else:
                raise ValueError
    return img_d


def extract_variable(loaded_dict, key_name):
    value = loaded_dict[key_name]
    try:
        value_shape = value.shape
    except AttributeError:
        return value
    if value_shape == ():
        return value.item()
    else:
        return value


def make_interactive_plot(x, y, im_arr, x_type=None, y_type=None):
    # create figure and plot scatter
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(x, y, ls="", marker="o")
    plt.xlabel(x_type)
    plt.ylabel(y_type)

    # create the annotations box
    im = OffsetImage(im_arr[0], zoom=5)
    xybox = (50., 50.)
    ab = AnnotationBbox(im, (0, 0), xybox=xybox, xycoords='data',
                        boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)

    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event',
                           lambda event: my_hover(event, im_arr, line, fig, xybox, ab, im, x, y))
    plt.show()


def show_images_in_order_of(images, x, outdir='/tmp/sortedperf/', basename='image_', decreasing=False):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    idxs_sorted_by_x = np.argsort(x)
    if decreasing:
        idxs_sorted_by_x = idxs_sorted_by_x[::-1]

    x_sorted = [x[i] for i in idxs_sorted_by_x]
    assert x_sorted == sorted(x, reverse=decreasing)

    images_sorted = [images[i] for i in idxs_sorted_by_x]
    filepaths = []
    for i, (x_val, img) in tqdm.tqdm(enumerate(zip(x_sorted, images_sorted)),
                                     desc='Saving images to <>/{}'.format(os.path.basename(outdir)),
                                     total=len(x_sorted)):
        outfile = os.path.join(outdir, basename + '{}_{}'.format(i, x_val) + '.png')
        Image.fromarray(img).save(outfile)
        filepaths.append(outfile)
    return idxs_sorted_by_x, filepaths


def link_files_in_order_of(file_list, x, outdir='/tmp/sortedperf/', basename='image_', decreasing=False):
    """
    x[i] should correspond to the value associated with file_list[i]
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    idxs_sorted_by_x = np.argsort(x)
    if decreasing:
        idxs_sorted_by_x = idxs_sorted_by_x[::-1]

    x_sorted = [x[i] for i in idxs_sorted_by_x]

    files_sorted = [file_list[i] for i in idxs_sorted_by_x]

    for i, (x_val, filename) in tqdm.tqdm(enumerate(zip(x_sorted, files_sorted)),
                                          desc='Linking files to <>/{}'.format(os.path.basename(outdir)),
                                          total=len(x_sorted)):
        file_ext = os.path.splitext(filename)[1]
        outfile = os.path.join(outdir, basename + '{}_{}'.format(i, x_val) + file_ext)
        assert os.path.exists(filename), '{} does not exist'.format(filename)
        if os.path.islink(outfile):
            os.unlink(outfile)
        if os.path.islink(outfile):
            import ipdb;
            ipdb.set_trace()
        commonprefix = os.path.commonprefix([filename, outfile])

        symlink(os.path.relpath(filename, os.path.dirname(outfile)), outfile)

    return idxs_sorted_by_x


def get_image_file_list(json_file):
    with open(json_file, 'r') as f:
        json_list = json.load(f)
    file_names = []
    for i in json_list['images']:
        file_names.append(i['file_name'])
    return file_names


def get_tiled_pred_gt_images(collated_stats_dict, img_types, labels_table, sorted_perf_outdir, overwrite=None,
                             list_of_original_input_images=None):
    data_type_as_str = 'image_name_id'
    print('Saving images in order of {}'.format(data_type_as_str))
    image_id_outdir = os.path.join(sorted_perf_outdir, '{}'.format('image_id'))
    image_name_ids = [im_name.rstrip('.png')
                      for im_name in get_image_file_list(extract_variable(collated_stats_dict, 'gt_json_file'))]
    basename = '{}_'.format(data_type_as_str)
    image_names = [os.path.join(image_id_outdir, basename.format() + '{}_{}'.format(i, x_val) + '.png') for i, x_val in
                   enumerate(image_name_ids)]
    if os.path.exists(image_id_outdir) and all([os.path.exists(i) for i in image_names]) and \
            (overwrite is False or
             (overwrite is None and
              y_or_n_input('All files already exist in {}.  Would you like to overwrite? y/N', default='n') == 'n')):
        print('Using existing images from {}'.format(image_id_outdir))
    else:
        print('Loading image data')
        non_input_img_types = [i for i in img_types if i != 'input_image']
        img_d = get_image_data(collated_stats_dict, non_input_img_types, labels_table)
        if 'input_image' in img_types:
            assert list_of_original_input_images is not None
            img_d['input_image'] = load_image_list(list_of_original_input_images)
            img_d['input_image'] = [np.concatenate([im for _ in range(len(non_input_img_types))], axis=0)
                                    for im in img_d['input_image']]

        print('Tiling images')
        import ipdb; ipdb.set_trace()
        imgs_side_by_side = [get_tile_image(list(imgs), (1, len(imgs)), margin_color=(0, 0, 0), margin_size=2)
                             for imgs in zip(*[img_d[img_type] for img_type in img_types])]

        ids, image_names = show_images_in_order_of(imgs_side_by_side, image_name_ids,
                                                   outdir=image_id_outdir, basename='{}_'.format(data_type_as_str))
        print('Images saved to {}'.format(image_id_outdir))
    return image_names


def main(collated_stats_npz, original_image_list=None, overwrite_imgs=False):
    collated_stats_dict = dict(np.load(collated_stats_npz))
    for key in collated_stats_dict:
        collated_stats_dict[key] = extract_variable(collated_stats_dict, key)

    if original_image_list is None:
        test_dir = os.path.dirname(collated_stats_npz).replace('cache', 'scripts' + os.sep + 'logs')
        image_list_filename = os.path.join(test_dir, 'image_filenames.npz')
        if not os.path.exists(image_list_filename):
            if os.path.exists(test_dir):
                err_msg = 'I looked for a list in {} but the directory doesnt exist.'.format(test_dir)
            else:
                err_msg = 'I looked for a list in {} and the directory exists, but the file does not.'.format(test_dir)
            raise Exception('Please specify the original image list; ' + err_msg)
        else:
            image_filenames = np.load(image_list_filename)['image_filenames']
            original_image_list = [f['img'] for f in image_filenames]
    use_labels_table = True
    collated_stats = collated_stats_dict['collated_stats_per_image_per_cat']
    # Make sure we can directly index the semantic class names to get the correct column for the categories
    problem_config = collated_stats_dict['problem_config']
    labels_table = problem_config.labels_table if use_labels_table else None
    instance_classes = [l.name for l in labels_table if l.isthing]
    data_types = [('sq', cls_name) for cls_name in instance_classes] + \
                 [('rq', cls_name) for cls_name in instance_classes]
    if 'upsnet' in collated_stats_npz:
        img_types = ('gt_pano_id_to_rgb', 'pred_pano_id_to_rgb', 'input_image')
    else:
        img_types = ('gt_inst_idx', 'pred_inst_idx', 'input_image')
    assert len(collated_stats_dict['categories']) == len(problem_config.semantic_class_names)
    assert set(collated_stats_dict['categories']) == set(problem_config.semantic_vals)
    print('Loading stats data')
    sorted_perf_outdir = os.path.join(os.path.dirname(collated_stats_npz), 'sortedperf')

    data_d = get_stat_data(collated_stats, data_types, problem_config.semantic_class_names)

    image_names = get_tiled_pred_gt_images(collated_stats_dict, img_types, labels_table, sorted_perf_outdir,
                                           overwrite=overwrite_imgs,
                                           list_of_original_input_images=original_image_list)

    for data_type in data_types:
        data_type_as_str = '_'.join(d for d in data_type) if type(data_type) is tuple else data_type
        # Convert metrics for 0 instance images to NaN
        stat_type = data_type if data_type is str else data_type[0]
        class_name = None if data_type is str else data_type[1]
        n_instances_this_cls = extract_n_instances_of_class(collated_stats, class_name,
                                                            problem_config.semantic_class_names)
        data_d[data_type][n_instances_this_cls == 0] = np.NaN
        print('Saving images in order of {}'.format(data_type))
        outdir = os.path.join(sorted_perf_outdir, '{}'.format(data_type_as_str))
        idxs_sorted_by_x = link_files_in_order_of(image_names, data_d[data_type],
                                                  outdir=outdir, basename='{}_'.format(data_type_as_str))
        np.save(os.path.join(os.path.dirname(outdir), 'sorted_idxs_{}.npz'.format(data_type_as_str)), idxs_sorted_by_x)
        print('Images saved to {}'.format(outdir))
    return sorted_perf_outdir


if __name__ == '__main__':
    args = parse_args()
    outdir = main(collated_stats_npz=args.collated_stats_npz)
