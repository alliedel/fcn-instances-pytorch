import os.path as osp
import tqdm

from instanceseg.analysis import visualization_utils
from instanceseg.datasets.cityscapes_transformations import convert_to_p_mode_file
from instanceseg.datasets.dataset_generator_registry import get_default_datasets_for_instance_counts
from instanceseg.datasets.instance_dataset import TransformedInstanceDataset
from instanceseg.train import trainer_exporter
from instanceseg.train.trainer import Trainer
from instanceseg.utils.misc import value_as_string
from instanceseg.datasets import runtime_transformations, precomputed_file_transformations, labels_table_cityscapes
import shutil
import os


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
        for batch_idx, (img_data_b, (sem_lbl_b, inst_lbl_b)) in t:
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
    data_loader = trainer.dataloaders['train']

    out_dir, out_dir_raw = transform_and_export_input_images(trainer, data_loader, split)
    print('Wrote images as loaded into {}, originals in {}'.format(out_dir, out_dir_raw))
