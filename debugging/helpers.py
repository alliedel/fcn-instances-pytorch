import os
import os.path as osp
import shutil
import tqdm
import numpy as np
import torch

from instanceseg.analysis import visualization_utils
from instanceseg.datasets import labels_table_cityscapes
from instanceseg.datasets.cityscapes_transformations import convert_to_p_mode_file
from instanceseg.train import trainer_exporter
from instanceseg.train.trainer import Trainer
from instanceseg.utils.misc import value_as_string
from instanceseg.utils import instance_utils


class DataloaderDataIntegrityChecker(object):
    def __init__(self, instance_problem: instance_utils.InstanceProblemConfig):
        self.instance_problem = instance_problem
        self.void_val = -1

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
            assert (inst_lbl[sem_lbl == tv] == 0).sum() == 0, \
                'Things should not have instance label 0 (error for class {}, {}'.format(
                    tv, self.instance_problem.semantic_class_names[semantic_idx])
            assert inst_lbl[sem_lbl == tv].max() <= self.instance_problem.n_instances_by_semantic_id[semantic_idx]

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
            assert semval in self.instance_problem.semantic_vals, \
                'Semantic label contains value {} which is not in the following list: {}'.format(
                    semval, zip(self.instance_problem.semantic_vals, self.instance_problem.semantic_class_names))

    def matching_void_constraint(self, sem_lbl, inst_lbl):
        if (sem_lbl[inst_lbl == self.void_val] != self.void_val).sum() > 0:
            raise Exception('Instance label was -1 where semantic label was not (e.g. - {})'.format(
                sem_lbl[inst_lbl == -1].max()))


def transform_and_export_input_images(trainer: Trainer, dataloader, split='train', out_dir=None, out_dir_raw=None,
                                      write_raw_images=True, write_transformed_images=True):
    if write_raw_images:
        try:
            filetypes = dataloader.dataset.raw_dataset.files[0].keys()
        except AttributeError:
            print('Warning: cant write raw images because I cant access original files through '
                  'dataset.raw_dataset.files')
        write_raw_images = False

    if write_raw_images:

        out_dir_raw = out_dir_raw or osp.join(trainer.exporter.out_dir, 'debug_viz_raw')
        if not os.path.exists(out_dir_raw):
            os.makedirs(out_dir_raw)
        try:
            filetypes = dataloader.dataset.raw_dataset.files[0].keys()
            for filetype in filetypes:
                suboutdir = os.path.join(out_dir_raw, filetype)
                if not os.path.exists(suboutdir):
                    os.makedirs(suboutdir)
        except AttributeError:
            raise Exception('Dataset isnt in a format where I can retrieve the raw image files easily to copy them '
                            'over.  Expected to be able to access dataloader.dataset.raw_dataset[0].files.keys')

        t = tqdm.tqdm(enumerate(dataloader.sampler.indices), total=len(dataloader), ncols=150, leave=False)
        for image_idx, idx_into_dataset in t:
            filename_d = dataloader.dataset.raw_dataset.files[idx_into_dataset]
            for filetype, filename in filename_d.items():
                out_name ='{}_'.format(image_idx) + os.path.basename(filename)
                if filetype is not 'inst_lbl':
                    shutil.copyfile(filename,
                                    os.path.join(out_dir_raw, filetype, out_name))
                else:
                    instance_palette = labels_table_cityscapes.get_instance_palette_image()
                    new_inst_lbl_file = os.path.join(out_dir_raw, filetype, 'modified_modep_' + out_name)
                    if not osp.isfile(new_inst_lbl_file):
                        assert osp.isfile(filename), '{} does not exist'.format(filename)
                        convert_to_p_mode_file(filename, new_inst_lbl_file, palette=instance_palette,
                                               assert_inside_palette_range=True)

            image_idx += 1
        # if pt is not None:
        #     dataloader.dataset.precomputed_file_transformation.transformer_sequence = pt
        # if rt is not None:
        #     dataloader.dataset.runtime_transformation.transformer_sequence = rt

    if write_transformed_images:
        t = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=150, leave=False)
        image_idx = 0
        out_dir = out_dir or osp.join(trainer.exporter.out_dir, 'debug_viz')
        integrity_checker = DataloaderDataIntegrityChecker(trainer.instance_problem)
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
                sem_lbl_np, inst_lbl_np = lbl_untransformed
                lt_combined = trainer.exporter.gt_tuple_to_combined(sem_lbl_np, inst_lbl_np)

                segmentation_viz = trainer_exporter.visualization_utils.visualize_segmentation(
                    lbl_true=lt_combined, img=img_untransformed, n_class=trainer.instance_problem.n_classes,
                    overlay=False)

                visualization_utils.export_visualizations(segmentation_viz, out_dir,
                                                          trainer.exporter.tensorboard_writer,
                                                          image_idx,
                                                          basename='loader_' + split + '_', tile=True)
                image_idx += 1

    return out_dir, out_dir_raw


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


def debug_dataloader(trainer: Trainer, split='train', out_dir=None):
    # TODO(allie, someday): Put cap on num images

    data_loader = trainer.dataloaders[split]
    out_dir = trainer.exporter.out_dir or out_dir
    outfile = osp.join(out_dir, 'dataloader_info_' + split + '.txt')
    write_dataloader_properties(data_loader, outfile)
    print('Writing images to {}'.format(out_dir))
    image_idx = 0
    data_loader = trainer.dataloaders[split]

    out_dir, out_dir_raw = transform_and_export_input_images(trainer, data_loader, split)
    print('Wrote images as loaded into {}, originals in {}'.format(out_dir, out_dir_raw))
