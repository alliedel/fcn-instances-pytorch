import os
import os.path as osp
import shutil
import tqdm
import numpy as np
import torch
import PIL.Image

from instanceseg.analysis import visualization_utils
from instanceseg.datasets import labels_table_cityscapes, instance_dataset
from instanceseg.datasets.cityscapes_transformations import convert_to_p_mode_file
from instanceseg.datasets.runtime_transformations import SemanticAgreementForInstanceLabelsRuntimeDatasetTransformer
from instanceseg.train import trainer_exporter
from instanceseg.train.trainer import Trainer
from instanceseg.utils.misc import value_as_string
from instanceseg.utils import instance_utils

MAX_IMAGE_DIM = 5000


class DataloaderDataIntegrityChecker(object):
    def __init__(self, instance_problem: instance_utils.InstanceProblemConfig, void_value=-1):
        self.instance_problem = instance_problem
        self.void_val = void_value

    @property
    def thing_values(self):
        return self.instance_problem.thing_class_vals

    @property
    def stuff_values(self):
        return self.instance_problem.stuff_class_vals

    def check_batch_label_integrity(self, sem_lbl_batch, inst_lbl_batch):
        self.thing_constraint(sem_lbl_batch, inst_lbl_batch)
        self.stuff_constraint(sem_lbl_batch, inst_lbl_batch)
        self.matching_void_constraint(sem_lbl_batch, inst_lbl_batch)
        self.sem_value_constraint(sem_lbl_batch)

    def thing_constraint(self, sem_lbl, inst_lbl):
        for tv in self.thing_values:  # Things must have instance id != 0
            semantic_idx = self.instance_problem.semantic_vals.index(tv)
            sem_locs = sem_lbl == tv
            if sem_locs.sum() > 0:
                if self.instance_problem.map_to_semantic:
                    # I think we need thing value 1 (things we're actually predicting) and thing value 0 (things we're
                    # not bothering to predict)
                    assert (inst_lbl[sem_locs] == 1).sum() + (inst_lbl[sem_locs] == 0).sum() == sem_locs.sum()
                else:
                    assert (inst_lbl[sem_locs] == 0).sum() == 0, \
                        'Things should not have instance label 0 (error for class {}, {}'.format(
                            tv, self.instance_problem.semantic_class_names[semantic_idx])
                # assert inst_lbl[sem_locs].max() <= self.instance_problem.n_instances_by_semantic_id[semantic_idx]

    def stuff_constraint(self, sem_lbl, inst_lbl):
        for sv in self.stuff_values:
            assert (inst_lbl[sem_lbl == sv] != 0).sum() == 0, 'Stuff should only have instance label 0, ' \
                                                              'not {}'.format(inst_lbl[sem_lbl == sv].max())

    def sem_value_constraint(self, sem_lbl):
        if torch.is_tensor(sem_lbl):
            semvals = torch.unique(sem_lbl)
        else:
            semvals = np.unique(sem_lbl)
        for semval in semvals:
            assert semval in self.instance_problem.semantic_transformed_label_ids or semval == self.void_val, \
                'Semantic label contains value {} which is not in the following list: {}'.format(
                    semval, '\n'.join(['{}: {}'.format(v, n)
                                       for v, n in zip(self.instance_problem.semantic_transformed_label_ids,
                                                       self.instance_problem.semantic_class_names)]))

    def matching_void_constraint(self, sem_lbl, inst_lbl):
        if (sem_lbl[inst_lbl == self.void_val] != self.void_val).sum() > 0:
            raise Exception('Instance label was -1 where semantic label was not (e.g. - {})'.format(
                sem_lbl[inst_lbl == -1].max()))


def get_multiplier(orig_shape, max_image_dim):
    return float(max_image_dim) / float(max(orig_shape[:2]))


def keep_size_small(out_img, max_image_dim=MAX_IMAGE_DIM):
    if out_img.shape[0] <= max_image_dim and out_img.shape[1] <= max_image_dim:
        out_img_resized = out_img
    else:
        multiplier = get_multiplier(out_img.shape[:2], max_image_dim)
        out_img_resized = visualization_utils.resize_img_by_multiplier(out_img, multiplier=multiplier)
    return out_img_resized


def save_and_possibly_resize(out_file, out_img, max_image_dim=MAX_IMAGE_DIM):
    out_img_resized = keep_size_small(out_img, max_image_dim)
    visualization_utils.write_image(out_file, out_img_resized)


def transform_and_export_input_images(trainer: Trainer, dataloader,
                                      split='train', out_dir_parent=None, write_raw_images=True,
                                      write_transformed_images=True, n_debug_images=None, max_image_dim=MAX_IMAGE_DIM):
    out_dir_parent = out_dir_parent or trainer.exporter.out_dir
    if write_transformed_images:
        transformed_image_export(dataloader, max_image_dim, n_debug_images, out_dir_parent, split, trainer)

    if write_raw_images:
        try:
            dataloader.dataset.raw_dataset.files[0].keys()
        except AttributeError:
            print('Warning: cant write raw images because I cant access original files through '
                  'dataset.raw_dataset.files')
            write_raw_images = False

    if write_raw_images:
        raw_image_exporter(dataloader, n_debug_images, out_dir_parent)

    # TODO(allie): Make side-by-side comparison

    return out_dir_parent


def raw_image_exporter(dataloader, n_debug_images, out_dir_parent):
    out_dir_raw_rgb = osp.join(out_dir_parent, 'debug_viz_raw_rgb')
    out_dir_raw_decomposed = osp.join(out_dir_parent, 'debug_viz_raw_decomposed')
    for out_dir_raw_ in [out_dir_raw_rgb, out_dir_raw_decomposed]:
        if not os.path.exists(out_dir_raw_):
            os.makedirs(out_dir_raw_)
    try:
        filetypes = dataloader.dataset.raw_dataset.files[0].keys()
        for filetype in filetypes:
            suboutdir = os.path.join(out_dir_raw_rgb, filetype)
            if not os.path.exists(suboutdir):
                os.makedirs(suboutdir)
    except AttributeError:
        raise Exception('Dataset isnt in a format where I can retrieve the raw image files easily to copy them '
                        'over.  Expected to be able to access dataloader.dataset.raw_dataset[0].files.keys')
    instance_palette = labels_table_cityscapes.get_instance_palette_image()
    decomposed_image_paths_raw = []
    t = tqdm.tqdm(enumerate(dataloader.sampler.indices),
                  total=len(dataloader) if n_debug_images is None else n_debug_images, ncols=120, leave=False)
    labels_table = dataloader.dataset.raw_dataset.labels_table
    cmap_dict_by_sem_val = {l.id: l.color for l in labels_table}
    sem_names_dict = {l.id: '({}){}'.format(l.id, l.name) for l in labels_table}
    for image_idx, idx_into_dataset in t:
        filename_d = dataloader.dataset.raw_dataset.files[idx_into_dataset]
        for filetype, filename in filename_d.items():
            out_name = '{}_'.format(image_idx) + os.path.basename(filename)
            if filetype is not 'inst_lbl':
                shutil.copyfile(filename, os.path.join(out_dir_raw_rgb, filetype, out_name))
            else:
                new_inst_lbl_file = os.path.join(out_dir_raw_rgb, filetype, 'modified_modep_' + out_name)
                if not osp.isfile(new_inst_lbl_file):
                    assert osp.isfile(filename), '{} does not exist'.format(filename)
                    convert_to_p_mode_file(filename, new_inst_lbl_file, palette=instance_palette,
                                           assert_inside_palette_range=True)
        sem_lbl = np.array(PIL.Image.open(filename_d['sem_lbl'], 'r'))
        inst_lbl = np.array(PIL.Image.open(filename_d['inst_lbl'], 'r'))
        input_image = np.array(PIL.Image.open(filename_d['img'], 'r'))
        assert len(sem_lbl.shape) == 2
        assert len(inst_lbl.shape) == 2
        out_img = create_instancewise_decomposed_label(sem_lbl, inst_lbl, input_image=input_image,
                                                       cmap_dict_by_sem_val=cmap_dict_by_sem_val,
                                                       sem_names_dict=sem_names_dict, max_image_dim=MAX_IMAGE_DIM,
                                                       sort_by_sem_val=True)
        imname = os.path.basename(filename_d['img']).rstrip('.png')
        decomposed_image_path = os.path.join(out_dir_raw_decomposed, 'decomposed_%012d' % image_idx + imname + '.png')
        save_and_possibly_resize(decomposed_image_path, out_img)
        decomposed_image_paths_raw.append(decomposed_image_path)
        if n_debug_images is not None and (image_idx + 1) >= n_debug_images:
            break


def transformed_image_export(dataloader, max_image_dim, n_debug_images, out_dir_parent, split, trainer):
    t = tqdm.tqdm(enumerate(dataloader), total=len(dataloader) if n_debug_images is None else n_debug_images,
                  ncols=120, leave=False)
    image_idx = 0
    out_dir_rgb = osp.join(out_dir_parent, 'debug_viz')
    out_dir_decomposed = osp.join(out_dir_parent, 'debug_viz_decomposed')
    for out_dir_ in [out_dir_rgb, out_dir_decomposed]:
        if not os.path.exists(out_dir_):
            os.makedirs(out_dir_)
    try:
        void_value = dataloader.dataset.raw_dataset.void_val
    except:
        void_value = trainer.instance_problem.void_value
    integrity_checker = DataloaderDataIntegrityChecker(trainer.instance_problem, void_value)
    decomposed_image_paths_transformed = []
    labels_table = dataloader.dataset.labels_table
    sem_vals = trainer.instance_problem.semantic_transformed_label_ids
    idxs_into_labels_table = [[l.name for l in labels_table].index(name)
                              for name in trainer.instance_problem.semantic_class_names]
    cmap_dict_by_sem_val = {v: labels_table[table_idx].color
                            for v, table_idx in zip(sem_vals, idxs_into_labels_table)}
    sem_names_dict = {v: '({})'.format(v) + n
                      for v, n in zip(sem_vals, trainer.instance_problem.semantic_class_names)}
    for batch_idx, (img_data_b, (sem_lbl_b, inst_lbl_b)) in t:
        integrity_checker.check_batch_label_integrity(sem_lbl_b, inst_lbl_b)

        for datapoint_idx in range(img_data_b.size(0)):
            img_data = img_data_b[datapoint_idx, ...]
            sem_lbl, inst_lbl = sem_lbl_b[datapoint_idx, ...], inst_lbl_b[datapoint_idx, ...]

            # Transform back to numpy format (rather than tensor that's formatted for model)
            data_to_img_transformer = lambda i, l: trainer.exporter.untransform_data(
                trainer.dataloaders[split], i, l)
            img_untransformed, lbl_untransformed = data_to_img_transformer(
                img_data, (sem_lbl, inst_lbl)) \
                if data_to_img_transformer is not None else (img_data, (sem_lbl, inst_lbl))

            segmentation_viz = trainer_exporter.visualization_utils.visualize_segmentations_as_rgb_imgs(
                gt_sem_inst_lbl_tuple=lbl_untransformed, pred_channelwise_lbl=None, margin_color=(255, 255, 255),
                overlay=True, img=img_untransformed, void_val=-1)

            visualization_utils.export_visualizations(segmentation_viz, out_dir_rgb,
                                                      trainer.exporter.tensorboard_writer,
                                                      image_idx, basename='loader_' + split + '_', tile=True)

            input_image_resized = visualization_utils.resize_img_to_sz(img_untransformed, sem_lbl.shape[0],
                                                                       sem_lbl.shape[1])
            out_img = create_instancewise_decomposed_label(sem_lbl, inst_lbl, input_image=input_image_resized,
                                                           sem_names_dict=sem_names_dict,
                                                           cmap_dict_by_sem_val=cmap_dict_by_sem_val,
                                                           max_image_dim=max_image_dim)
            decomposed_image_path = os.path.join(out_dir_decomposed, 'decomposed_%012d.png' % image_idx)
            save_and_possibly_resize(decomposed_image_path, out_img)
            decomposed_image_paths_transformed.append(decomposed_image_path)
            image_idx += 1
            if n_debug_images is not None and image_idx >= n_debug_images:
                break
        if n_debug_images is not None and image_idx >= n_debug_images:
            break


def unique(tensor_or_np_arr):
    if torch.is_tensor(tensor_or_np_arr):
        unique_vals = torch.unique(tensor_or_np_arr)
        unique_vals = [i.item() for i in unique_vals]
    else:
        unique_vals = np.unique(tensor_or_np_arr)

    return unique_vals


def decompose_into_mask_image_per_instance(sem_lbl, inst_lbl, to_np=True):
    sem_vals = unique(sem_lbl)
    inst_vals_by_sem_val = {}
    instance_masks = {sem_val: [] for sem_val in sem_vals}
    for sem_val in sem_vals:
        sem_cls_locs = sem_lbl == sem_val
        inst_vals_by_sem_val[sem_val] = unique(inst_lbl[sem_cls_locs])
        for inst_val in inst_vals_by_sem_val[sem_val]:
            mask = sem_cls_locs & (inst_lbl == inst_val)
            instance_masks[sem_val].append(mask.numpy() if to_np and torch.is_tensor(mask) else mask)
    return instance_masks, inst_vals_by_sem_val


def visualize_masks_by_sem_cls(instance_mask_dict, inst_vals_dict, cmap_dict_by_sem_val=None,
                               sem_names_dict=None, input_image=None,
                               margin_size_small=3, margin_size_large=6, void_vals=(-1, 255),
                               margin_color=(255, 255, 255), downsample_multiplier=1.0, sort_by_sem_val=True):
    """
    dicts' keys are the semantic values of each of the instances
    """
    assert set(instance_mask_dict.keys()) == set(inst_vals_dict.keys())
    assert all([len(lst1) == len(lst2) for lst1, lst2 in zip(instance_mask_dict.values(), inst_vals_dict.values())])

    use_funky_void_pixels = True

    sem_vals = list(instance_mask_dict)  # .keys()
    if sort_by_sem_val:
        sem_vals = sorted(sem_vals)

    if sem_names_dict is not None:
        missing_vals = [s for s in sem_vals if s not in sem_names_dict and s not in void_vals]
        assert missing_vals == [], \
            'semantic values present: {}; values given a label: {}; assignments missing: {}'.format(
                sem_vals, list(sem_names_dict), missing_vals)
    if cmap_dict_by_sem_val is not None:
        missing_vals = [s for s in sem_vals if s not in cmap_dict_by_sem_val and s not in void_vals]
        assert missing_vals == [], \
            'semantic values present: {}; values given a label: {}'.format(
                sem_vals, list(cmap_dict_by_sem_val), missing_vals)

    n_sem_classes = len(sem_vals)

    if sem_names_dict is None:
        sem_names_dict = {sem_val: '{},'.format(sem_val if sem_val not in void_vals else '{} (void),'.format(sem_val))
                          for sem_val in sem_vals}

    if cmap_dict_by_sem_val is None:
        cmap_dict_by_sem_val = visualization_utils.label_colormap(n_sem_classes) * 255
    elif any([sum(c) != 0 and max(c) <= 1 for c in cmap_dict_by_sem_val.values()]):
        print('Warning: colormap should be in the range [0, 255], not [0,1]')
        import ipdb;
        ipdb.set_trace()

    sem_cls_rows = []
    for sem_val in sem_vals:

        if input_image is not None:
            input_image_resized = visualization_utils.resize_img_by_multiplier(input_image, downsample_multiplier)
            true_label_masks = [input_image_resized]
            colormaps = [input_image_resized]
        else:
            true_label_masks, colormaps = [], []
        for inst_val, inst_mask in zip(inst_vals_dict[sem_val], instance_mask_dict[sem_val]):
            if sem_val in sem_names_dict:
                sem_name = sem_names_dict[sem_val]
            else:
                assert sem_val in void_vals
                sem_name = 'void,'

            mask_label = '{} {}'.format(sem_name, inst_val)
            assert len(inst_mask.shape) == 2
            rgb_mask = np.repeat(inst_mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8) * 255
            if use_funky_void_pixels and sem_val in void_vals:
                void_inst_mask = inst_mask
                viz_void = (
                        np.random.random((void_inst_mask.shape[0], void_inst_mask.shape[1], 3)) * 255
                ).astype(np.uint8)
                rgb_mask[void_inst_mask] = viz_void[void_inst_mask]
                colormap = viz_void.copy()
            else:
                color = np.array(cmap_dict_by_sem_val[sem_val], dtype='uint8')
                colormap = np.ones_like(rgb_mask) * color

            if downsample_multiplier != 1:
                colormap = visualization_utils.resize_img_by_multiplier(colormap, downsample_multiplier)
                rgb_mask = visualization_utils.resize_img_by_multiplier(rgb_mask, downsample_multiplier)

            # font_scale = 2.0 * downsample_multiplier
            visualization_utils.write_word_in_img_center(colormap, mask_label, font_scale=max(0.5,
                                                                                              2.0 *
                                                                                              downsample_multiplier))

            true_label_masks.append(rgb_mask)
            colormaps.append(colormap)
        assert all(c.shape == colormaps[0].shape for c in colormaps)
        assert all(i.shape == true_label_masks[0].shape for i in true_label_masks)

        true_label_mask_row = visualization_utils.get_tile_image_1d(true_label_masks, concat_direction='horizontal',
                                                                    margin_color=margin_color,
                                                                    margin_size=margin_size_small)
        colormap_row = visualization_utils.get_tile_image_1d(colormaps, concat_direction='horizontal',
                                                             margin_color=margin_color,
                                                             margin_size=margin_size_small)
        row_with_colormaps = visualization_utils.get_tile_image_1d([colormap_row, true_label_mask_row],
                                                                   concat_direction='vertical',
                                                                   margin_color=margin_color,
                                                                   margin_size=margin_size_small)
        sem_cls_rows.append(row_with_colormaps)

    max_width = max([scr.shape[1] for scr in sem_cls_rows])
    sem_cls_rows = [visualization_utils.pad_image_to_right_and_bottom(sem_cls_row, dst_width=max_width)
                    for sem_cls_row in sem_cls_rows]
    tiled_masks = visualization_utils.get_tile_image_1d(sem_cls_rows, concat_direction='vertical',
                                                        margin_color=margin_color,
                                                        margin_size=margin_size_large)
    return tiled_masks


def create_instancewise_decomposed_label(sem_lbl, inst_lbl, input_image=None, sem_names_dict=None,
                                         cmap_dict_by_sem_val=None, max_image_dim=MAX_IMAGE_DIM, sort_by_sem_val=True):
    assert (sem_lbl.shape[0], sem_lbl.shape[1]) == (inst_lbl.shape[0], inst_lbl.shape[1])
    if input_image is not None:
        assert (sem_lbl.shape[0], sem_lbl.shape[1]) == (input_image.shape[0], input_image.shape[1])
    instance_mask_dict, inst_vals_dict = decompose_into_mask_image_per_instance(sem_lbl, inst_lbl)
    n_images_wide = max([len(ivs) for sv, ivs in inst_vals_dict.items()]) + (1 if input_image is not None else 0)
    n_images_tall = 2 * len(instance_mask_dict)  # 2 * for titles
    out_img_width = sem_lbl.shape[1] * n_images_wide
    out_img_height = sem_lbl.shape[0] * n_images_tall
    downsample_multiplier = max(1, get_multiplier((out_img_height, out_img_width), max_image_dim=max_image_dim))

    out_img = visualize_masks_by_sem_cls(instance_mask_dict, inst_vals_dict, cmap_dict_by_sem_val=cmap_dict_by_sem_val,
                                         sem_names_dict=sem_names_dict, input_image=input_image,
                                         downsample_multiplier=downsample_multiplier, sort_by_sem_val=True)
    return out_img


def write_dataloader_properties(dataloader, outfile):
    precomputed_file_transformation = dataloader.dataset.precomputed_file_transformation
    runtime_transformation = dataloader.dataset.runtime_transformation
    transformer_tag = ''
    for tr in [precomputed_file_transformation, runtime_transformation]:
        attributes = tr.get_attribute_items() if tr is not None else {}.items()
        transformer_tag += '__'.join(['{}-{}'.format(k, value_as_string(v)) for k, v in attributes])
    with open(outfile, 'w') as fid:
        fid.write(transformer_tag)
    return


def debug_dataloader(trainer: Trainer, split='train', out_dir=None, n_debug_images=None, max_image_dim=MAX_IMAGE_DIM):
    # TODO(allie, someday): Put cap on num images

    data_loader = trainer.dataloaders[split]
    out_dir = trainer.exporter.out_dir or out_dir
    outfile = osp.join(out_dir, 'dataloader_info_' + split + '.txt')
    write_dataloader_properties(data_loader, outfile)
    print('Writing images to {}'.format(out_dir))
    data_loader = trainer.dataloaders[split]

    out_dir_parent = transform_and_export_input_images(trainer, data_loader, split, n_debug_images=n_debug_images,
                                                       max_image_dim=max_image_dim)
    print('Wrote images into {}'.format(out_dir_parent))
