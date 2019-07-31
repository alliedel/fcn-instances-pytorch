import argparse
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import json
import os
from PIL import Image
import tqdm
import six.moves


from instanceseg.analysis.visualization_utils import label_colormap, get_tile_image
from instanceseg.ext.panopticapi.utils import rgb2id

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


def panoid_im_to_rgb(panoptic_id_im, labels_table):
    inst_label_colormap = label_colormap(255)
    out_sem_img = np.zeros((panoptic_id_im.shape[0], panoptic_id_im.shape[1], 3))
    out_inst_img = np.zeros_like(out_sem_img)

    sem_id_im, inst_id_im = np.divmod(panoptic_id_im, 1000)
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


def load_images(image_dir, json_list_file, to_id=False, dtype=np.uint8, labels_table=None):
    image_list = []

    with open(json_list_file, 'r') as f:
        file_list = json.load(f)
        for i in tqdm.tqdm(file_list['images'], desc='Loading images from <>/{}'.format(os.path.basename(image_dir)),
                           total=len(file_list['images'])):
            file_name = i['file_name']
            # Hackery.
            if not os.path.exists(os.path.join(image_dir, file_name)) and '_gt' in image_dir:
                file_name = 'groundtruth_{}'.format(file_name).replace('_image.png', '_id2rgb_cocopano.png')
            if not os.path.exists(os.path.join(image_dir, file_name)) and '_pred' in image_dir:
                file_name = 'predictions_{}'.format(file_name).replace('_image.png', '_id2rgb_cocopano.png')
            if not os.path.exists(os.path.join(image_dir, file_name)) and 'orig_images' in image_dir:
                file_name = 'image_{}'.format(file_name).replace('_image', '')
            assert os.path.exists(os.path.join(image_dir, file_name)), '{} does not exist'.format(os.path.join(
                image_dir, file_name))
            with Image.open(os.path.join(image_dir, file_name)) as img:
                pan_im = np.array(img, dtype=dtype)
                if to_id or labels_table is not None:
                    panoptic_id_im = rgb2id(pan_im)
                    if to_id:
                        im = panoptic_id_im
                    else:
                        im = panoid_im_to_rgb(panoptic_id_im, labels_table)
                else:
                    im = pan_im
                image_list.append(im)
    return image_list


def get_data(collated_stats_dict, data_types=('sq', 'rq'),
             img_types=('gt_inst_idx', 'pred_inst_idx', 'input_image'), use_labels_table=True):
    """
    For a single class:
    data_types = (('sq', 'car'), ('rq', 'car'))
    """
    collated_stats = collated_stats_dict['collated_stats_per_image_per_cat'].item()
    # Make sure we can directly index the semantic class names to get the correct column for the categories
    problem_config = collated_stats_dict['problem_config'].item()
    labels_table = problem_config.labels_table if use_labels_table else None

    assert len(collated_stats_dict['categories']) == len(problem_config.semantic_class_names)
    assert set(collated_stats_dict['categories']) == set(problem_config.semantic_vals)
    data_d = {}
    for data_type in data_types:
        if data_type in ('sq', 'rq', 'pq'):
            data = collated_stats[data_type]
        elif data_type is not str and data_type[0] in ('sq', 'rq', 'pq'):
            assert data_type[1] in problem_config.semantic_class_names
            column_idx = problem_config.semantic_class_names.index(data_type[1])
            data = collated_stats[data_type[0]][:, column_idx]
        else:
            raise ValueError('I dont know how to retrieve {}'.format(data_type))
        data_d[data_type] = data

    img_d = {}
    for img_type in img_types:
        if img_type in ('gt_inst_idx', 'pred_inst_idx'):
            if img_type == 'gt_inst_idx':
                img_d[img_type] = load_images(collated_stats_dict['gt_folder'].item(),
                                              collated_stats_dict['gt_json_file'].item(), to_id=False,
                                              labels_table=labels_table)
            elif img_type == 'pred_inst_idx':
                img_d[img_type] = load_images(collated_stats_dict['pred_folder'].item(),
                                              collated_stats_dict['pred_json_file'].item(), to_id=False,
                                              labels_table=labels_table)
            else:
                raise ValueError
        elif img_type in ('input_image',):
            pth, nm = os.path.split(os.path.dirname(collated_stats_dict['pred_folder'].item().rstrip('/')))
            orig_img_dir = os.path.join(os.path.dirname(pth).replace('cache', 'scripts/logs'), nm, 'orig_images')
            img_d[img_type] = load_images(orig_img_dir,
                                          collated_stats_dict['gt_json_file'].item(), to_id=False)
        else:
            raise ValueError

    return data_d, img_d


def make_interactive_plot(x, y, im_arr):
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

    for i, (x_val, img) in tqdm.tqdm(enumerate(zip(x_sorted, images_sorted)),
                                     desc='Saving images to <>/{}'.format(os.path.basename(outdir)),
                                     total=len(x_sorted)):
        Image.fromarray(img).save(os.path.join(outdir, basename + '{}_{}'.format(i, x_val) + '.png'))
    return idxs_sorted_by_x


if __name__ == '__main__':
    args = parse_args()
    collated_stats_dict = np.load(args.collated_stats_npz)
    x_type = ('sq', 'car')
    y_type = ('rq', 'car')
    data_types = (x_type, y_type)
    img_types = ('gt_inst_idx', 'pred_inst_idx', 'input_image')
    print('Loading data')
    datas, im_arrs = get_data(collated_stats_dict, data_types=data_types, img_types=img_types)
    im_arrs['input_image'] = [np.concatenate([im, im], axis=0) for im in im_arrs['input_image']]
    print('Tiling images')
    imgs_side_by_side = [get_tile_image(list(imgs), (1, len(imgs)), margin_color=(0, 0, 0), margin_size=2)
                         for imgs in zip(*[im_arrs[img_type] for img_type in img_types])]

    x_ = datas[x_type]
    y_ = datas[y_type]
    im_arr_ = im_arrs[img_types[0]]
    # make_interactive_plot(x_, y_, im_arr_)

    for data_type in data_types:
        data_type_as_str = '_'.join(d for d in data_type) if type(data_type) is tuple else data_type
        print('Saving images in order of {}'.format(data_type))
        outdir = os.path.join(os.path.dirname(args.collated_stats_npz),
                              'sortedperf', '{}'.format(data_type_as_str))
        idxs_sorted_by_x = show_images_in_order_of(imgs_side_by_side, datas[data_type],
                                                   outdir=outdir, basename='{}_'.format(data_type_as_str))
        np.save(os.path.join(os.path.dirname(outdir), 'sorted_idxs_{}.npz'.format(data_type)), idxs_sorted_by_x)
        print('Images saved to {}'.format(outdir))
