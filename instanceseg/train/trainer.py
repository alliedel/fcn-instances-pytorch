import math

import numpy as np
import torch
import tqdm
from torch.autograd import Variable

import instanceseg
import instanceseg.losses.loss
import instanceseg.utils.export
import instanceseg.utils.misc
from instanceseg.models.model_utils import is_nan, any_nan
from instanceseg.train import trainer_exporter
from instanceseg.utils import datasets, instance_utils
import os.path as osp

DEBUG_ASSERTS = True

BINARY_AUGMENT_MULTIPLIER = 100.0
BINARY_AUGMENT_CENTERED = True


class TrainerState(object):
    def __init__(self, max_iter=None):
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter

    def done_training(self):
        return self.iteration > self.max_iter


class Trainer(object):
    def __init__(self, cuda, model, optimizer, train_loader, val_loader, max_iter,
                 instance_problem: instance_utils.InstanceProblemConfig,
                 size_average=True,
                 interval_validate=None,
                 loss_type='cross_entropy', matching_loss=True, use_semantic_loss=False,
                 exporter: trainer_exporter.TrainerExporter = None,
                 train_loader_for_val=None, loader_semantic_lbl_only=False,
                 augment_input_with_semantic_masks=False,
                 generate_new_synthetic_data_each_epoch=False):
        # System parameters
        self.cuda = cuda

        # Model
        self.model = model

        # Problem setup objects
        self.instance_problem = instance_problem

        # Loss
        self.size_average = size_average
        self.matching_loss = matching_loss
        self.use_semantic_loss = use_semantic_loss
        self.loss_type = loss_type
        self.loss_fcn = self.build_my_loss_function()
        self.loss_fcn_matching_override = self.build_my_loss_function(matching_override=True)

        # Optimizer
        self.optim = optimizer

        # Data loading
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loader_for_val = train_loader_for_val
        self.loader_semantic_lbl_only = loader_semantic_lbl_only
        self.augment_input_with_semantic_masks = augment_input_with_semantic_masks
        self.generate_new_synthetic_data_each_epoch = generate_new_synthetic_data_each_epoch

        # Exporting objects
        self.exporter = exporter

        self.state = TrainerState(max_iter=max_iter)
        self.interval_validate = interval_validate or len(self.train_loader)

    def train(self):
        start_epoch = self.state.epoch  # 0 unless we loaded existing log
        max_epoch = int(math.ceil(1. * self.state.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(start_epoch, max_epoch, desc='Train', ncols=80, leave=True):
            self.state.epoch = epoch
            self.train_epoch()
            if self.state.done_training():
                break

    def train_epoch(self):
        self.model.train()

        if self.generate_new_synthetic_data_each_epoch:
            seed = np.random.randint(100)
            self.train_loader.dataset.raw_dataset.initialize_locations_per_image(seed)
            self.train_loader_for_val.dataset.raw_dataset.initialize_locations_per_image(seed)

        for batch_idx, (img_data, target) in tqdm.tqdm(  # tqdm: progress bar
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.state.epoch, ncols=80, leave=False):
            # Update state (iteration)
            iteration = batch_idx + self.state.epoch * len(self.train_loader)
            if self.state.iteration != 0 and (iteration - 1) != self.state.iteration:
                continue  # for resuming
            self.state.iteration = iteration

            # Validate
            if self.state.iteration % self.interval_validate == 0:
                val_metrics, _ = self.validate()

            self.train_minibatch(img_data, target)
            if self.state.done_training():
                break

    def train_minibatch(self, img_data, target):
        # Prepare input
        assert self.model.training
        full_input, sem_lbl, inst_lbl = self.prepare_data_for_forward_pass(img_data, target)

        # Reset gradient (a Pytorch thing, so we don't accumulate gradients from every step)
        self.optim.zero_grad()

        # Run forward pass (get prediction); compute loss
        score = self.model(full_input)
        pred_permutations, total_loss, _ = self.compute_loss(score, sem_lbl, inst_lbl)
        avg_loss = total_loss / len(full_input)

        # Compute gradient
        avg_loss.backward()

        # Improve model
        self.optim.step()

        # Update quantities of interest
        self.exporter.update_after_train_minibatch(self, full_input, score, sem_lbl, inst_lbl,
                                                   pred_permutations, avg_loss)

    def prepare_data_for_forward_pass(self, img_data, target, requires_grad=True):
        """
        Loads data and transforms it into Variable based on GPUs, input augmentations, and loader type (if semantic)
        requires_grad: True if training; False if you're not planning to backprop through (for validation / metrics)
        """
        if not self.loader_semantic_lbl_only:
            (sem_lbl, inst_lbl) = target
        else:
            assert self.use_semantic_loss, 'Can''t run instance losses if loader is semantic labels only.  Set ' \
                                           'use_semantic_loss to True'
            assert type(target) is not tuple
            sem_lbl = target
            inst_lbl = torch.zeros_like(sem_lbl)
            inst_lbl[sem_lbl == -1] = -1

        if self.cuda:
            img_data, (sem_lbl, inst_lbl) = img_data.cuda(), (sem_lbl.cuda(), inst_lbl.cuda())
        full_input = img_data if not self.augment_input_with_semantic_masks \
            else self.augment_image(img_data, sem_lbl)
        full_input, sem_lbl, inst_lbl = \
            Variable(full_input, volatile=requires_grad), \
            Variable(sem_lbl, requires_grad=requires_grad), \
            Variable(inst_lbl, requires_grad=requires_grad)
        return full_input, sem_lbl, inst_lbl

    def build_my_loss_function(self, matching_override=None):
        # permutations, loss, loss_components = f(scores, sem_lbl, inst_lbl)
        matching = matching_override if matching_override is not None else self.matching_loss
        my_loss_fcn = instanceseg.losses.loss.loss_2d_factory(  # f(scores, sem_lbl, inst_lbl)
            self.loss_type, self.instance_problem.semantic_instance_class_list,
            self.instance_problem.instance_count_id_list,
            return_loss_components=True, matching=matching)
        return my_loss_fcn

    def compute_loss(self, score, sem_lbl, inst_lbl, val_matching_override=False):
        # permutations, loss, loss_components = f(scores, sem_lbl, inst_lbl)
        map_to_semantic = self.instance_problem.map_to_semantic
        if not (sem_lbl.size() == inst_lbl.size() == (score.size(0), score.size(2), score.size(3))):
            raise Exception('Sizes of score, targets are incorrect')

        if map_to_semantic:
            inst_lbl[inst_lbl > 1] = 1
        loss_fcn = self.loss_fcn if not val_matching_override else self.loss_fcn_matching_override

        permutations, loss, loss_components = loss_fcn(score, sem_lbl, inst_lbl)

        if is_nan(loss.data[0]):
            raise ValueError('losses after forward pass')
        if loss.data[0] > 1e4:
            print('WARNING: losses={} at iteration {}'.format(loss.data[0], self.state.iteration))
        if any_nan(score.data):
            raise ValueError('score is nan after forward pass')

        return permutations, loss, loss_components

    def augment_image(self, img, sem_lbl):
        semantic_one_hot = datasets.labels_to_one_hot(sem_lbl, self.instance_problem.n_semantic_classes)
        return datasets.augment_channels(img, BINARY_AUGMENT_MULTIPLIER * semantic_one_hot -
                                         (0.5 if BINARY_AUGMENT_CENTERED else 0), dim=1)

    def validate(self, split='val'):
        """
        If split == 'val': write_metrics, save_checkpoint, update_best_checkpoint default to True.
        If split == 'train': write_metrics, save_checkpoint, update_best_checkpoint default to
            False.
        """
        assert split in ['train', 'val']

        training = self.model.training
        self.model.eval()

        # Get the data loader
        if split == 'train':
            data_loader = self.train_loader_for_val
        else:
            data_loader = self.val_loader

        val_loss = 0
        for batch_idx, (img_data, target) in tqdm.tqdm(
                enumerate(data_loader), total=len(data_loader),
                desc='Valid iteration (split=%s)=%d' % (split, self.state.iteration), ncols=80, leave=False):

                # Don't waste computation if we don't need to run on the remaining images
            if not self.exporter.continue_validation_iterations(split):
                continue
            full_input, sem_lbl, inst_lbl = self.prepare_data_for_forward_pass(img_data, target, requires_grad=False)
            score = self.model(full_input)
            pred_permutations, loss, _ = self.compute_loss(score, sem_lbl, inst_lbl, val_matching_override=True)
            val_loss_mb = float(loss.data[0]) / len(full_input)

            self.exporter.update_after_val_minibatch(full_input, sem_lbl, inst_lbl, score, pred_permutations, loss,
                                                     data_loader.dataset.runtime_transformation, split, img=img_data)
            val_loss += val_loss_mb

        val_loss /= len(data_loader)
        val_metrics = self.exporter.update_after_validation_epoch(self, split)

        if training:
            self.model.train()

        return val_metrics

        # return true_labels, pred_labels, score, pred_permutations, val_loss, segmentation_visualizations,
        # score_visualizations

    @property
    def is_running_validation(self):
        return not self.model.training
