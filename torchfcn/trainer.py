import datetime
import math
import os
import os.path as osp
import shutil

import numpy as np
import pytz
import scipy.misc
import torch
import tqdm
from torch.autograd import Variable

import torchfcn
from torchfcn import losses
from torchfcn import visualization_utils
from torchfcn.visualization_utils import log_images

MY_TIMEZONE = 'America/New_York'


class Trainer(object):
    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter,
                 size_average=False, interval_validate=None, matching_loss=True,
                 tensorboard_writer=None, visualize_overlay=False, visualize_confidence=True):
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
        self.visualize_overlay = visualize_overlay
        self.visualize_confidence = visualize_confidence

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

    def validate(self, split='val', write_metrics=None, save_checkpoint=None,
                 update_best_checkpoint=None):
        """
        If split == 'val': write_metrics, save_checkpoint, update_best_checkpoint default to True.
        If split == 'train': write_metrics, save_checkpoint, update_best_checkpoint default to
            False.
        """
        write_metrics = (split == 'val') if write_metrics is None else write_metrics
        save_checkpoint = (split == 'val') if save_checkpoint is None else save_checkpoint
        update_best_checkpoint = save_checkpoint if update_best_checkpoint is None \
            else update_best_checkpoint
        compute_metrics = write_metrics or save_checkpoint or update_best_checkpoint

        assert split in ['train', 'val']
        if split == 'train':
            data_loader = self.train_loader
        else:
            data_loader = self.val_loader

        # Turn off shuffling temporarily
        shuffle = data_loader.shuffle
        data_loader.shuffle = False

        # eval instead of training mode temporarily
        training = self.model.training
        self.model.eval()

        n_class = self.model.n_classes

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(data_loader), total=len(data_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            should_visualize = len(visualizations) < 9
            if not(compute_metrics or should_visualize):
                # Don't waste computation if we don't need to run on the remaining images
                break
            true_labels_single_batch, pred_labels_single_batch, val_loss_single_batch, \
                visualizations_single_batch = self.validate_single_batch(
                    data, target, data_loader=data_loader, n_class=n_class,
                    should_visualize=should_visualize)
            label_trues.append(true_labels_single_batch)
            label_preds.append(pred_labels_single_batch)
            val_loss += val_loss_single_batch
            visualizations += visualizations_single_batch
        val_loss /= len(data_loader)

        self.export_visualizations(visualizations, split)

        if compute_metrics:
            metrics = torchfcn.utils.label_accuracy_score(
                label_trues, label_preds, n_class)
            if write_metrics:
                self.write_metrics(metrics, val_loss, split)
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar('data/{}_loss'.format(split),
                                                       val_loss, self.iteration)
            if save_checkpoint:
                self.save_checkpoint()
            if update_best_checkpoint:
                self.update_best_checkpoint_if_best(mean_iu=metrics[2])

        # Restore shuffle, training settings set prior to function call
        self.train_loader.shuffle = shuffle
        if training:
            self.model.training()

    def export_visualizations(self, visualizations, split='val_'):
        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        out_img = visualization_utils.get_tile_image(visualizations, margin_color=[255, 255, 255],
                                                     margin_size=50)
        scipy.misc.imsave(out_file, out_img)
        if self.tensorboard_writer is not None:
            basename = split
            tag = '{}images'.format(basename, 0)
            log_images(self.tensorboard_writer, tag, [out_img], self.iteration, numbers=[0])

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

    def validate_single_batch(self, data, target, data_loader, n_class, should_visualize):
        true_labels = []
        pred_labels = []
        visualizations = []
        val_loss = 0
        if self.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        score = self.model(data)

        gt_permutations, loss = losses.cross_entropy2d(
            score, target, semantic_instance_labels=self.model.semantic_instance_class_list,
            matching=self.matching_loss, size_average=self.size_average)
        if np.isnan(float(loss.data[0])):
            raise ValueError('loss is nan while validating')
        val_loss += float(loss.data[0]) / len(data)

        imgs = data.data.cpu()
        # numpy_score = score.data.numpy()
        lbl_pred = score.data.max(dim=1)[1].cpu().numpy()[:, :, :]
        # confidence = numpy_score[numpy_score == numpy_score.max(dim=1)[0]]
        lbl_true = target.data.cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = data_loader.dataset.untransform(img, lt)
            true_labels.append(lt)
            pred_labels.append(lp)
            if should_visualize:
                viz = visualization_utils.visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
                    overlay=self.visualize_overlay)
                visualizations.append(viz)
        return true_labels, pred_labels, val_loss, visualizations

    def train_epoch(self):
        self.model.train()

        n_class = self.model.n_classes

        for batch_idx, (data, target) in tqdm.tqdm(  # tqdm: progress bar
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()
                self.validate(split='train')

            assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optim.zero_grad()
            score = self.model(data)

            gt_permutations, loss = losses.cross_entropy2d(
                score, target,
                matching=self.matching_loss,
                size_average=self.size_average,
                semantic_instance_labels=self.model.semantic_instance_class_list)
            loss /= len(data)
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('data/training_loss', loss.data[0],
                                                   self.iteration)
            loss.backward()
            self.optim.step()

            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            metrics = []
            for lt, lp in zip(lbl_true, lbl_pred):
                acc, acc_cls, mean_iu, fwavacc = \
                    torchfcn.utils.label_accuracy_score(
                        [lt], [lp], n_class=n_class)
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            self.write_metrics(metrics, loss, split='train')
            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
