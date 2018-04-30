import collections
import datetime
import math
import os
import os.path as osp
import shutil

import numpy as np
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import tqdm
import torch.nn.functional as F

import torchfcn
from torchfcn import losses
from torchfcn import visualization_utils, instance_utils
from torchfcn.visualization_utils import log_images

MY_TIMEZONE = 'America/New_York'

DEBUG_ASSERTS = True


class Trainer(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter, instance_problem,
                 size_average=False, interval_validate=None, matching_loss=True,
                 tensorboard_writer=None, train_loader_for_val=None, loader_semantic_lbl_only=False,
                 use_semantic_loss=False, export_analytics_every=1):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone(MY_TIMEZONE))
        self.size_average = size_average
        self.matching_loss = matching_loss
        self.tensorboard_writer = tensorboard_writer
        self.train_loader_for_val = train_loader_for_val
        self.loader_semantic_lbl_only = loader_semantic_lbl_only
        self.instance_problem = instance_problem
        self.which_heatmaps_to_visualize = 'same semantic'  # 'all'
        self.use_semantic_loss = use_semantic_loss

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0
        # TODO(allie): clean up max combined class... computing accuracy shouldn't need it.
        self.n_combined_class = int(sum(self.model.semantic_instance_class_list)) + 1
        self.export_analytics_every = export_analytics_every

    def my_cross_entropy(self, score, sem_lbl, inst_lbl, **kwargs):
        if not (sem_lbl.size() == inst_lbl.size() == (score.size(0), score.size(2),
                                                      score.size(3))):
            import ipdb;
            ipdb.set_trace()
            raise Exception('Sizes of score, targets are incorrect')
        permutations, loss = losses.cross_entropy2d(
            score, sem_lbl, inst_lbl,
            semantic_instance_labels=self.instance_problem.semantic_instance_class_list,
            instance_id_labels=self.instance_problem.instance_count_id_list,
            matching=self.matching_loss,
            size_average=self.size_average, break_here=False, recompute_optimal_loss=False, **kwargs)
        return permutations, loss

    def validate(self, split='val', write_metrics=None, write_analytics=True, save_checkpoint=None,
                 update_best_checkpoint=None, should_export_visualizations=True):
        """
        If split == 'val': write_metrics, save_checkpoint, update_best_checkpoint default to True.
        If split == 'train': write_metrics, save_checkpoint, update_best_checkpoint default to
            False.
        """
        metrics, visualizations = None, None
        analytics = None
        write_metrics = (split == 'val') if write_metrics is None else write_metrics
        save_checkpoint = (split == 'val') if save_checkpoint is None else save_checkpoint
        update_best_checkpoint = save_checkpoint if update_best_checkpoint is None \
            else update_best_checkpoint
        should_compute_metrics = write_metrics or save_checkpoint or update_best_checkpoint

        assert split in ['train', 'val']
        if split == 'train':
            data_loader = self.train_loader_for_val
        else:
            data_loader = self.val_loader

        # eval instead of training mode temporarily
        training = self.model.training
        self.model.eval()

        val_loss = 0
        segmentation_visualizations, score_visualizations = [], []
        label_trues, label_preds, scores, pred_permutations = [], [], [], []
        visualizations_need_to_be_exported = True if should_export_visualizations else False
        num_images_to_visualize = min(len(data_loader), 9)
        for batch_idx, (data, lbls) in tqdm.tqdm(
                enumerate(data_loader), total=len(data_loader),
                desc='Valid iteration (split=%s)=%d' % (split, self.iteration), ncols=80,
                leave=False):
            if not self.loader_semantic_lbl_only:
                (sem_lbl, inst_lbl) = lbls
                if self.use_semantic_loss:
                    inst_lbl = torch.zeros_like(sem_lbl)
                    inst_lbl[sem_lbl == -1] = -1
            else:
                assert self.use_semantic_loss, 'Can''t run instance loss if loader is semantic labels only.  Set ' \
                                               'use_semantic_loss to True'
                assert type(lbls) is not tuple
                sem_lbl = lbls
                inst_lbl = torch.zeros_like(sem_lbl)
                inst_lbl[sem_lbl == -1] = -1

            should_visualize = len(segmentation_visualizations) < num_images_to_visualize
            if not (should_compute_metrics or should_visualize):
                # Don't waste computation if we don't need to run on the remaining images
                continue
            true_labels_sb, pred_labels_sb, score_sb, pred_permutations_sb, val_loss_sb, \
            segmentation_visualizations_sb, score_visualizations_sb = \
                self.validate_single_batch(data, sem_lbl, inst_lbl, data_loader=data_loader,
                                           should_visualize=should_visualize)
            if visualizations_need_to_be_exported and len(segmentation_visualizations) == num_images_to_visualize:
                self.export_visualizations(segmentation_visualizations, 'seg_' + split, tile=True)
                self.export_visualizations(score_visualizations, 'score_' + split, tile=False)
                visualizations_need_to_be_exported = False

            label_trues += true_labels_sb
            label_preds += pred_labels_sb
            val_loss += val_loss_sb
            scores += [score_sb]
            pred_permutations += [pred_permutations_sb]
            segmentation_visualizations += segmentation_visualizations_sb
            score_visualizations += score_visualizations_sb

        if visualizations_need_to_be_exported and len(segmentation_visualizations) == num_images_to_visualize:
            if should_export_visualizations:
                self.export_visualizations(segmentation_visualizations, 'seg_' + split, tile=True)
                self.export_visualizations(score_visualizations, 'score_' + split, tile=False)

        val_loss /= len(data_loader)

        if should_compute_metrics:
            metrics = self.compute_metrics(label_trues, label_preds, pred_permutations)
            if write_metrics:
                self.write_metrics(metrics, val_loss, split)
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar('metrics/{}/loss'.format(split),
                                                       val_loss, self.iteration)
                    self.tensorboard_writer.add_scalar('metrics/{}/mIOU'.format(split), metrics[2],
                                                       self.iteration)

            if save_checkpoint:
                self.save_checkpoint()
            if update_best_checkpoint:
                self.update_best_checkpoint_if_best(mean_iu=metrics[2])

        # Restore training settings set prior to function call
        if training:
            self.model.train()

        visualizations = (segmentation_visualizations, score_visualizations)
        return metrics, visualizations

    def permute_scores(self, score, pred_permutations):
        score_permuted_to_match = score.clone()
        for ch in range(score.size(1)):  # NOTE(allie): iterating over channels, but maybe should iterate over
            # batch size?
            score_permuted_to_match[:, ch, :, :] = score[:, pred_permutations[:, ch], :, :]
        return score_permuted_to_match

    def permute_labels(self, label_preds, permutations):
        if torch.is_tensor(label_preds):
            label_preds_permuted = label_preds.clone()
        else:
            label_preds_permuted = label_preds.copy()
        for idx in range(permutations.shape[0]):
            permutation = permutations[idx, :]
            for old_channel, new_channel in enumerate(permutation):
                label_preds_permuted[label_preds == old_channel] = new_channel
        return label_preds_permuted

    def compute_metrics(self, label_trues, label_preds, permutations=None):
        if permutations is not None:
            assert type(permutations) == list, NotImplementedError('I''m assuming permutations are a list of ndarrays '
                                                                   'from multiple batches')
            label_preds_permuted = [self.permute_labels(label_pred, perms)
                                    for label_pred, perms in zip(label_preds, permutations)]
        else:
            label_preds_permuted = label_preds
        metrics = torchfcn.utils.label_accuracy_score(label_trues, label_preds_permuted, n_class=self.n_combined_class)
        return metrics
    def write_metrics(self, metrics, loss, split):
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                    datetime.datetime.now(pytz.timezone(MY_TIMEZONE)) -
                    self.timestamp_start).total_seconds()
            if split == 'val':
                log = [self.epoch, self.iteration] + [''] * 5 + \
                      [loss] + list(metrics) + [elapsed_time]
            elif split == 'train':
                log = [self.epoch, self.iteration] + [loss] + \
                      metrics.tolist() + [''] * 5 + [elapsed_time]
            else:
                raise ValueError('split not recognized')
            log = map(str, log)
            f.write(','.join(log) + '\n')

    def export_visualizations(self, visualizations, basename='val_', tile=True, outdir=None):
        outdir = outdir or osp.join(self.out, 'visualization_viz')
        export_visualizations(visualizations, outdir, self.tensorboard_writer, self.iteration, basename=basename,
                              tile=tile)

    def save_checkpoint(self):
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))

    def update_best_checkpoint_if_best(self, mean_iu):
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

    def validate_single_batch(self, data, sem_lbl, inst_lbl, data_loader, should_visualize):
        true_labels = []
        pred_labels = []
        pred_permutations = []
        segmentation_visualizations = []
        score_visualizations = []
        val_loss = 0
        if self.cuda:
            data, (sem_lbl, inst_lbl) = data.cuda(), (sem_lbl.cuda(), inst_lbl.cuda())
        # TODO(allie): Don't turn target into variables yet here? (Not yet sure if this works
        # before we've actually combined the semantic and instance labels...)
        data, sem_lbl, inst_lbl = Variable(data, volatile=True), \
                                  Variable(sem_lbl), Variable(inst_lbl)
        score = self.model(data)
        pred_permutations, loss = self.my_cross_entropy(score, sem_lbl, inst_lbl)
        if np.isnan(float(loss.data[0])):
            raise ValueError('loss is nan while validating')
        val_loss += float(loss.data[0]) / len(data)

        imgs = data.data.cpu()
        softmax_scores = F.softmax(score, dim=1).data.cpu().numpy()
        inst_lbl_pred = score.data.max(dim=1)[1].cpu().numpy()[:, :, :]

        # TODO(allie): convert to sem, inst visualizations.
        lbl_true_sem, lbl_true_inst = (sem_lbl.data.cpu(), inst_lbl.data.cpu())
        for idx, (img, sem_lbl, inst_lbl, lp) in enumerate(zip(imgs, lbl_true_sem, lbl_true_inst, inst_lbl_pred)):
            img = data_loader.dataset.untransform_img(img)
            pp = pred_permutations[idx, :]
            try:
                (sem_lbl, inst_lbl) = (data_loader.dataset.untransform_lbl(sem_lbl),
                                       data_loader.dataset.untransform_lbl(inst_lbl))
            except:
                import ipdb;
                ipdb.set_trace()
                raise

            lt_combined = self.gt_tuple_to_combined(sem_lbl, inst_lbl)
            true_labels.append(lt_combined)
            pred_labels.append(lp)
            if should_visualize:
                # Segmentations
                viz = visualization_utils.visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt_combined, pred_permutations=pp, img=img, n_class=self.n_combined_class,
                    overlay=False)
                segmentation_visualizations.append(viz)
                # Scores
                sp = softmax_scores[idx, :, :, :]

                # TODO(allie): Fix this -- bug(?!)
                lp = np.argmax(sp, axis=0)
                # try:
                #     assert np.all(np.argmax(sp, axis=0) == lp)
                # except:
                #     import ipdb; ipdb.set_trace()
                if self.which_heatmaps_to_visualize == 'same semantic':
                    inst_sem_classes_present = torch.np.unique(true_labels)
                    inst_sem_classes_present = inst_sem_classes_present[inst_sem_classes_present != -1]
                    sem_classes_present = np.unique([self.instance_problem.semantic_instance_class_list[c]
                                                     for c in inst_sem_classes_present])
                    channels_for_these_semantic_classes = [inst_idx for inst_idx, sem_cls in enumerate(
                        self.instance_problem.semantic_instance_class_list) if sem_cls in sem_classes_present]
                    channels_to_visualize = channels_for_these_semantic_classes
                elif self.which_heatmaps_to_visualize == 'all':
                    channels_to_visualize = list(range(sp.shape[0]))
                else:
                    raise ValueError('which heatmaps to visualize is not recognized: {}'.format(
                        self.which_heatmaps_to_visualize))
                channel_labels = self.instance_problem.get_channel_labels('{} {}')
                viz = visualization_utils.visualize_heatmaps(scores=sp,
                                                             lbl_true=lt_combined,
                                                             lbl_pred=lp,
                                                             pred_permutations=pp,
                                                             n_class=self.n_combined_class,
                                                             score_vis_normalizer=sp.max(),
                                                             channel_labels=channel_labels,
                                                             channels_to_visualize=channels_to_visualize)
                score_visualizations.append(viz)
        return true_labels, pred_labels, score, pred_permutations, val_loss, segmentation_visualizations, \
               score_visualizations

    def gt_tuple_to_combined(self, sem_lbl, inst_lbl):
        semantic_instance_class_list = self.instance_problem.semantic_instance_class_list
        instance_count_id_list = self.instance_problem.instance_count_id_list
        return instance_utils.combine_semantic_and_instance_labels(sem_lbl, inst_lbl,
                                                                   semantic_instance_class_list,
                                                                   instance_count_id_list)

    def train_epoch(self):
        self.model.train()

        # n_class = len(self.train_loader.dataset.class_names)
        # n_class = self.model.n_classes

        for batch_idx, (data, target) in tqdm.tqdm(  # tqdm: progress bar
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration
            if self.iteration % self.interval_validate == 0:
                if self.train_loader_for_val is not None:
                    self.validate('train')
                self.validate()

            assert self.model.training
            if not self.loader_semantic_lbl_only:
                (sem_lbl, inst_lbl) = target
            else:
                assert type(target) is not tuple
                sem_lbl = target
                inst_lbl = torch.zeros_like(sem_lbl)
                inst_lbl[sem_lbl == -1] = -1

            if self.cuda:
                data, (sem_lbl, inst_lbl) = data.cuda(), (sem_lbl.cuda(), inst_lbl.cuda())
            data, sem_lbl, inst_lbl = Variable(data), \
                                      Variable(sem_lbl), Variable(inst_lbl)
            self.optim.zero_grad()
            score = self.model(data)

            pred_permutations, loss = self.my_cross_entropy(score, sem_lbl, inst_lbl)
            loss /= len(data)
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('metrics/training_loss', loss.data[0],
                                                   self.iteration)
            loss.backward()
            self.optim.step()

            inst_lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true_sem, lbl_true_inst = sem_lbl.data.cpu().numpy(), inst_lbl.data.cpu().numpy()
            metrics = []
            for sem_lbl, inst_lbl, lp in zip(lbl_true_sem, lbl_true_inst, inst_lbl_pred):
                lt_combined = self.gt_tuple_to_combined(sem_lbl, inst_lbl)
                acc, acc_cls, mean_iu, fwavacc = \
                    self.compute_metrics(label_trues=[lt_combined], label_preds=[lp], permutations=[pred_permutations])
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)
            self.write_metrics(metrics, loss, split='train')
            if np.mod(batch_idx, self.export_analytics_every) == 0:
                lt_combined = self.gt_tuple_to_combined(lbl_true_sem, lbl_true_inst)
            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80, leave=True):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break


def export_visualizations(visualizations, outdir, tensorboard_writer, iteration, basename='val_', tile=True):
    if not osp.exists(outdir):
        os.makedirs(outdir)
    if tile:
        out_img = visualization_utils.get_tile_image(visualizations, margin_color=[255, 255, 255],
                                                     margin_size=50)
        tag = '{}images'.format(basename)
        if tensorboard_writer is not None:
            log_images(tensorboard_writer, tag, [out_img], iteration, numbers=[0])
        out_subdir = osp.join(outdir, tag)
        if not osp.exists(out_subdir):
            os.makedirs(out_subdir)
        out_file = osp.join(out_subdir, 'iter-%012d.jpg' % iteration)
        scipy.misc.imsave(out_file, out_img)
    else:
        tag = '{}images'.format(basename)
        out_subdir = osp.join(outdir, tag)
        if not osp.exists(out_subdir):
            os.makedirs(out_subdir)
        for img_idx, out_img in enumerate(visualizations):
            if tensorboard_writer is not None:
                log_images(tensorboard_writer, tag, [out_img], iteration, numbers=[img_idx])
            out_subsubdir = osp.join(out_subdir, str(img_idx))
            if not osp.exists(out_subsubdir):
                os.makedirs(out_subsubdir)
            out_file = osp.join(out_subsubdir, 'iter-%012d.jpg' % iteration)
            scipy.misc.imsave(out_file, out_img)


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
