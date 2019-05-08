import tqdm
import torch
import numpy as np

from instanceseg.train import trainer_exporter
from instanceseg.train.trainer import Trainer


def transform_and_export_input_images(trainer: Trainer, img_data, sem_lbl, inst_lbl, split='train'):
    # Transform back to numpy format (rather than tensor that's formatted for model)
    data_to_img_transformer = lambda i, l: trainer.exporter.untransform_data(
        trainer.train_loader, i, l)
    img_untransformed, lbl_untransformed = data_to_img_transformer(
        img_data, (sem_lbl, inst_lbl)) \
        if data_to_img_transformer is not None else (img_data, (sem_lbl, inst_lbl))
    sem_lbl_np, inst_lbl_np = lbl_untransformed
    lt_combined = trainer.exporter.gt_tuple_to_combined(sem_lbl_np, inst_lbl_np)

    segmentation_viz = trainer_exporter.visualization_utils.visualize_segmentation(
        lbl_true=lt_combined, img=img_untransformed, n_class=trainer.instance_problem.n_classes,
        overlay=False)
    trainer.exporter.export_score_and_seg_images(segmentation_viz, None,
                                                 trainer.state.iteration, split)


def debug_dataloader(trainer, split='train'):
    # TODO(allie, someday): Put cap on num images

    data_loader = {'train': trainer.train_loader, 'val': trainer.val_loader}[split]

    t = tqdm.tqdm(
        enumerate(data_loader), total=len(data_loader),
        ncols=150, leave=False)
    for batch_idx, (img_data, (sem_lbl, inst_lbl)) in t:
        for datapoint_idx in range(img_data.size(0)):
            transform_and_export_input_images(
                trainer, img_data[batch_idx, ...], sem_lbl[batch_idx, ...],
                inst_lbl[batch_idx, ...], split)
