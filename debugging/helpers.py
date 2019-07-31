import os.path as osp
import tqdm

from instanceseg.analysis import visualization_utils
from instanceseg.train import trainer_exporter
from instanceseg.train.trainer import Trainer
from instanceseg.utils.misc import value_as_string
from instanceseg.datasets import runtime_transformations, precomputed_file_transformations

def transform_and_export_input_images(trainer: Trainer, dataloader, split='train', out_dir=None, out_dir_raw=None,
                                      write_raw_images=True, write_transformed_images=True):
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

    if write_raw_images:
        try:
            pt = dataloader.dataset.precomputed_file_transformation.transformer_sequence
            dataloader.dataset.precomputed_file_transformation.transformer_sequence = \
                [tr for tr in pt if type(tr) in ()]
        except AttributeError:
            assert dataloader.dataset.precomputed_file_transformation is None
            pt = None
        try:
            rt = dataloader.dataset.precomputed_file_transformation.transformer_sequence
            dataloader.dataset.runtime_transformation.transformer_sequence = \
                [tr for tr in pt if type(tr) in (runtime_transformations.ResizeRuntimeDatasetTransformer,
                                                 runtime_transformations.BasicRuntimeDatasetTransformer)]
        except AttributeError:
            assert dataloader.dataset.precomputed_file_transformation is None
            rt = None
        # (precomputed_file_transformations.InstanceOrderingPrecomputedDatasetFileTransformation)

        t = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=150, leave=False)
        out_dir_raw = out_dir_raw or osp.join(trainer.exporter.out_dir, 'debug_viz_raw')
        image_idx = 0
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
                out_dir = out_dir or osp.join(trainer.exporter.out_dir, 'debug_viz')

                visualization_utils.export_visualizations(segmentation_viz, out_dir_raw,
                                                          trainer.exporter.tensorboard_writer,
                                                          image_idx,
                                                          basename='loader_' + split + '_', tile=True)
                image_idx += 1
        if pt is not None:
            dataloader.dataset.precomputed_file_transformation.transformer_sequence = pt
        if rt is not None:
            dataloader.dataset.runtime_transformation.transformer_sequence = rt
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

