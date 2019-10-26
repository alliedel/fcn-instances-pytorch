import math
import os
import subprocess

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

import instanceseg
import instanceseg.losses.loss
import instanceseg.utils.export
from instanceseg.datasets import dataset_statistics
from instanceseg.models.fcn8s_instance import FCN8sInstance
from instanceseg.models.model_utils import is_nan, any_nan
from instanceseg.train import metrics, trainer_exporter
from instanceseg.utils import datasets
from instanceseg.utils import misc
from instanceseg.utils.instance_utils import InstanceProblemConfig
import time


WATCH_VAL_SUBDIR = 'watching_validator'


DEBUG_ASSERTS = True
DEBUG_MEMORY_ISSUES = False

BINARY_AUGMENT_MULTIPLIER = 100.0
BINARY_AUGMENT_CENTERED = True


class ValProgressWatcher(object):
    def __init__(self, watcher_log_directory, trainer_model_directory):
        assert os.path.exists(watcher_log_directory)
        self.watcher_log_directory = watcher_log_directory
        self.trainer_model_directory = trainer_model_directory
        self.prev_total = self.get_total()
        self.t = self.make_new_progress_bar(self.prev_total)
        self.prev_count = 0

    def make_new_progress_bar(self, total=None):
        total = total or self.get_total()
        return tqdm.tqdm(total=total, desc=self.desc, ncols=80, leave=True)

    @property
    def desc(self):
        in_progress_files = self.get_in_progress_files()
        in_progress_files = in_progress_files if len(in_progress_files) != 1 else in_progress_files[0]
        return 'Val progress: {n_finished}/{n_trained} models (In progress: {in_progress_files})'.format(
            n_finished=self.get_finished_count(), n_trained=self.get_total(),
            in_progress_files=in_progress_files)

    def get_total(self):
        return len(self.get_trained_model_list())

    def get_trained_model_list(self):
        trained_model_list = os.listdir(self.trainer_model_directory)
        return [f for f in trained_model_list if f.endswith('.pth') or f.endswith('.pth.tar')]

    def get_watcher_log_files(self):
        return os.listdir(os.path.join(self.watcher_log_directory))

    def get_finished_files(self):
        return [f for f in self.get_watcher_log_files() if os.path.basename(f).startswith('finished-')]

    def get_in_progress_files(self):
        return [f for f in self.get_watcher_log_files() if os.path.basename(f).startswith('started-')]

    def get_queued_files(self):
        return [f for f in self.get_watcher_log_files() if os.path.basename(f).startswith('queue-')]

    def get_finished_count(self):
        return len(self.get_finished_files())

    def finished(self):
        return self.get_total() == len(self.get_finished_files())

    def update(self):
        count = self.get_finished_count()
        total = self.get_total()
        if self.get_total() != self.prev_total:
            self.make_new_progress_bar(total)
            self.prev_count = 0
        self.t.update(count - self.prev_count)
        self.prev_count = count
        self.prev_total = total

    def close(self):
        self.t.close()


class TrainingState(object):
    def __init__(self, max_iteration):
        self.iteration = 0
        self.epoch = 0
        self.max_iteration = max_iteration

    def training_complete(self):
        return self.iteration >= self.max_iteration


class Trainer(object):
    def __init__(self, cuda, model: FCN8sInstance, optimizer: Optimizer or None, dataloaders,
                 out_dir, max_iter,
                 instance_problem: InstanceProblemConfig,
                 size_average=True, interval_validate=None, loss_type='cross_entropy',
                 matching_loss=True,
                 tensorboard_writer=None, loader_semantic_lbl_only=False,
                 use_semantic_loss=False, augment_input_with_semantic_masks=False,
                 write_instance_metrics=True,
                 generate_new_synthetic_data_each_epoch=False,
                 export_activations=False, activation_layers_to_export=(),
                 lr_scheduler: ReduceLROnPlateau = None,
                 n_model_checkpoints=None,
                 skip_validation=False, skip_model_checkpoint_saving=False):

        # System parameters
        self.cuda = cuda
        self.skip_validation = skip_validation  # If another process is doing it for us, or we're going to do it later.
        self.skip_model_checkpoint_saving = skip_model_checkpoint_saving  # Generally if we're alreadyloading from a
        # checkpoint

        # Model objects
        self.model = model

        # Training objects
        self.optim = optimizer

        # Dataset objects
        self.dataloaders = dataloaders

        # Problem setup objects
        self.instance_problem = instance_problem

        # Exporting objects
        # Exporting parameters

        # Loss parameters
        self.size_average = size_average
        self.matching_loss = matching_loss
        self.loss_type = loss_type

        # Data loading parameters
        self.loader_semantic_lbl_only = loader_semantic_lbl_only

        self.use_semantic_loss = use_semantic_loss
        self.augment_input_with_semantic_masks = augment_input_with_semantic_masks
        self.generate_new_synthetic_data_each_epoch = generate_new_synthetic_data_each_epoch

        self.write_instance_metrics = write_instance_metrics

        # Stored values
        self.last_val_loss = None

        self.interval_validate = interval_validate or (
            len(self.dataloaders['train']) if 'train' in self.dataloaders else None)

        self.state = TrainingState(max_iteration=max_iter)
        self.best_mean_iu = 0
        # TODO(allie): clean up max combined class... computing accuracy shouldn't need it.

        if self.loss_type is not None:
            self.loss_object = self.build_my_loss()
            self.eval_loss_object_with_matching = self.build_my_loss(
                matching_override=True)  # Uses matching
        else:
            self.loss_object = None
            self.eval_loss_object_with_matching = None
        self.lr_scheduler = lr_scheduler

        metric_maker_kwargs = {
            'problem_config': self.instance_problem,
            'component_loss_function': self.eval_loss_fcn_with_matching,
            'augment_function_img_sem': self.augment_image
            if self.augment_input_with_semantic_masks else None
        }
        metric_makers = {
            split: metrics.InstanceMetrics(self.dataloaders[split], **metric_maker_kwargs)
            for split in self.dataloaders.keys()
        }

        export_config = trainer_exporter.ExportConfig(interval_validate=interval_validate,
                                                      export_activations=export_activations,
                                                      activation_layers_to_export=activation_layers_to_export,
                                                      write_instance_metrics=write_instance_metrics,
                                                      max_n_saved_models=n_model_checkpoints,
                                                      skip_model_checkpoint_saving=skip_model_checkpoint_saving)
        self.exporter = trainer_exporter.TrainerExporter(
            out_dir=out_dir, instance_problem=instance_problem,
            export_config=export_config, tensorboard_writer=tensorboard_writer,
            metric_makers=metric_makers)
        self.t_val = None  # We need to initialize this when we make our validation watcher.

    def get_validation_progress_bar(self) -> ValProgressWatcher:
        watcher_log_dir = os.path.join(self.exporter.out_dir, 'model_checkpoints-val-log')
        if not os.path.exists(watcher_log_dir):
            print('Waiting for validator to start..')
            time.sleep(10)  # give subprocess a chance to make its log directory.
        if not os.path.exists(watcher_log_dir):
            t_val = None
            print('No directory exists at {}'.format(watcher_log_dir))
            if self.skip_validation:
                misc.color_text("Validation might not be happening: Couldn't find a watcher log directory at {}".format(
                    watcher_log_dir))
        else:
            t_val = ValProgressWatcher(watcher_log_directory=watcher_log_dir,
                                       trainer_model_directory=self.exporter.model_history_saver.model_checkpoint_dir)
        return t_val

    @property
    def eval_loss_fcn_with_matching(self):
        return None if self.loss_type is None else self.eval_loss_object_with_matching.loss_fcn

    def loss_fcn(self, *args, **kwargs):
        loss_result = self.loss_object.loss_fcn(*args, **kwargs)
        return loss_result

    def prepare_data_for_forward_pass(self, img_data, target, requires_grad=True):
        """
        Loads data and transforms it into Variable based on GPUs, input augmentations, and loader
        type (if semantic)
        requires_grad: True if training; False if you're not planning to backprop through (for
        validation / metrics)
        """
        if not self.loader_semantic_lbl_only:
            (sem_lbl, inst_lbl) = target
        else:
            assert self.use_semantic_loss, 'Can''t run instance losses if loader is semantic ' \
                                           'labels only.  Set ' \
                                           'use_semantic_loss to True'
            assert type(target) is not tuple
            sem_lbl = target
            inst_lbl = torch.zeros_like(sem_lbl)
            inst_lbl[sem_lbl == -1] = -1

        if self.cuda:
            img_data, (sem_lbl, inst_lbl) = img_data.cuda(), (sem_lbl.cuda(), inst_lbl.cuda())
        full_input = img_data if not self.augment_input_with_semantic_masks \
            else self.augment_image(img_data, sem_lbl)
        if requires_grad:
            # APD: Still not sure it's okay to set requires_grad to False on all labels (was True
            #  previously)
            full_input.requires_grad = True
            sem_lbl.requires_grad = False
            inst_lbl.requires_grad = False
        else:
            with torch.no_grad():  # volatile replacement
                full_input.requires_grad = False
                sem_lbl.requires_grad = False
                inst_lbl.requires_grad = False

        assert not self.instance_problem.include_instance_channel0, NotImplementedError
        for sem_id in self.instance_problem.thing_class_ids:  # inst_val == 0 for thing classes
            # get mapped to 255
            inst_lbl[(sem_lbl == sem_id) * (inst_lbl == 0)] = self.instance_problem.void_value

        return full_input, sem_lbl, inst_lbl

    def build_my_loss(self, matching_override=None):
        # permutations, loss, loss_components = f(scores, sem_lbl, inst_lbl)

        matching = matching_override if matching_override is not None else self.matching_loss

        my_loss_object = instanceseg.losses.loss.loss_object_factory(
            self.loss_type,
            self.instance_problem.model_channel_semantic_ids,
            self.instance_problem.instance_count_id_list,
            matching, self.size_average)
        return my_loss_object

    def compute_loss(self, score, sem_lbl, inst_lbl, cap_sizes=True,
                     val_matching_override=False) -> instanceseg.losses.loss.MatchingLossResult:
        """
        Returns assignments, total_loss, loss_components_by_channel
        """
        # permutations, loss, loss_components = f(scores, sem_lbl, inst_lbl)
        map_to_semantic = self.instance_problem.map_to_semantic
        if not (sem_lbl.size() == inst_lbl.size() == (score.size(0), score.size(2), score.size(3))):
            raise Exception('Sizes of score, targets are incorrect')

        if DEBUG_ASSERTS:
            unique_sem_vals = torch.unique(sem_lbl)
            for sem_val in unique_sem_vals:
                assert sem_val in self.instance_problem.semantic_ids or sem_val == self.instance_problem.void_value

        train_inst_lbl = None
        if cap_sizes:
            removed_gt_inst_tuples = []
            assert len(sem_lbl.shape) == 3
            for i in range(sem_lbl.shape[0]):
                removed_gt_inst_tuples_img_i = []
                for sem_val, n_channels_allocated in zip(
                        self.instance_problem.semantic_ids,
                        self.instance_problem.model_n_instances_by_semantic_id):
                    if sem_val in self.instance_problem.thing_class_ids:
                        inst_vals, inst_sizes = dataset_statistics.get_instance_sizes(sem_lbl[i, ...],
                                                                                      inst_lbl[i, ...], sem_val)
                        sorted_inst_vals = [inst_vals[i] for i in np.argsort(inst_sizes)][::-1]
                        bad_inst_vals = sorted_inst_vals[n_channels_allocated:]
                        if DEBUG_ASSERTS:
                            assert (len(bad_inst_vals) == max(0, len(inst_vals) - n_channels_allocated))
                        for inst_val in bad_inst_vals:
                            if train_inst_lbl is None:  # allocate new inst lbl if we need to
                                train_inst_lbl = torch.clone(inst_lbl)
                            train_inst_lbl[(sem_lbl == sem_val) * (inst_lbl == inst_val)] = \
                                self.instance_problem.void_value
                        removed_gt_inst_tuples_img_i.extend([(sem_val, iv) for iv in bad_inst_vals])
                    else:
                        assert sem_val in self.instance_problem.stuff_class_ids
                removed_gt_inst_tuples.append(removed_gt_inst_tuples_img_i)
        else:
            removed_gt_inst_tuples = None
        if train_inst_lbl is None:
            train_inst_lbl = inst_lbl
        if map_to_semantic:
            train_inst_lbl[train_inst_lbl > 1] = 1
        # print('APD: Running loss fcn')
        if val_matching_override:
            loss_result = self.eval_loss_fcn_with_matching(score, sem_lbl, train_inst_lbl)
        else:
            loss_result = self.loss_fcn(score, sem_lbl, train_inst_lbl)
        loss_result.avg_loss = loss_result.total_loss / score.size(0)

        if cap_sizes:
            for i, l in enumerate(loss_result.assignments.unassigned_gt_sem_inst_tuples):
                loss_result.assignments.unassigned_gt_sem_inst_tuples[i].extend(removed_gt_inst_tuples[i])
        return loss_result

    def augment_image(self, img, sem_lbl):
        semantic_one_hot = datasets.labels_to_one_hot(sem_lbl,
                                                      self.instance_problem.n_semantic_classes)
        return datasets.augment_channels(img, BINARY_AUGMENT_MULTIPLIER * semantic_one_hot -
                                         (0.5 if BINARY_AUGMENT_CENTERED else 0), dim=1)

    def save_checkpoint_and_update_if_best(self, mean_iu, save_by_iteration=True):
        current_checkpoint_file = self.exporter.save_checkpoint(self.state.epoch,
                                                                self.state.iteration, self.model,
                                                                self.optim, self.best_mean_iu,
                                                                mean_iu)
        if mean_iu > self.best_mean_iu or self.best_mean_iu == 0:
            self.best_mean_iu = mean_iu
            self.exporter.copy_checkpoint_as_best(current_checkpoint_file)

    def test(self, test_outdir, split='test', save_scores=False):
        """
        If split == 'val': write_metrics, save_checkpoint, update_best_checkpoint default to True.
        If split == 'train': write_metrics, save_checkpoint, update_best_checkpoint default to
            False.
        """
        self.exporter.instance_problem.save(self.exporter.instance_problem_path)
        predictions_outdir = os.path.join(test_outdir, 'predictions')
        groundtruth_outdir = os.path.join(test_outdir, 'groundtruth')
        scores_outdir = None if not save_scores else predictions_outdir.replace('predictions',
                                                                                'scores')
        images_outdir = os.path.join(test_outdir, 'images')
        for my_dir in [predictions_outdir, groundtruth_outdir, images_outdir, scores_outdir]:
            if my_dir is None:
                continue
            if not os.path.exists(my_dir):
                os.makedirs(my_dir)
            else:
                print(Warning('I didnt expect the directory {} to already exist.'.format(my_dir)))
        data_loader = self.dataloaders[split]
        if data_loader.sampler.sequential:
            indices = [i for i in data_loader.sampler]
        else:
            raise Exception(
                'We need the sampler to be sequential to know which images we\'re testing')
        image_filenames = [data_loader.dataset.get_image_file(i) for i in indices]
        np.savez(os.path.join(test_outdir, 'image_filenames.npz'), image_filenames=image_filenames)
        with torch.set_grad_enabled(False):
            t = tqdm.tqdm(
                enumerate(data_loader), total=len(data_loader),
                desc='Test iteration (split=%s)=%d' %
                     (split, self.state.iteration), ncols=150,
                leave=False)
            batch_img_idx = 0
            for batch_idx, data_dict in t:
                img_data, lbls = data_dict['image'], (data_dict['sem_lbl'], data_dict['inst_lbl'])
                full_input, sem_lbl, inst_lbl = self.prepare_data_for_forward_pass(
                    img_data, lbls, requires_grad=False)
                batch_sz = full_input.size(0)
                score = self.model(full_input)
                sem_lbl_np = sem_lbl.data.cpu().numpy()
                inst_lbl_np = inst_lbl.data.cpu().numpy()
                label_pred = score.data.max(dim=1)[1].cpu().numpy()[:, :, :]
                img_idxs = list(range(batch_img_idx, batch_img_idx + batch_sz))
                if save_scores:
                    score_names = ['scores_{:06d}.pt'.format(img_idx) for img_idx in img_idxs]
                    assert len(score.size()) == 4
                    for idx_into_batch, outf in enumerate(score_names):
                        torch.save(score[idx_into_batch, ...], os.path.join(scores_outdir, outf))

                # TODO(allie): remap both to original semantic values
                prediction_names = ['predictions_{:06d}_sem255instid2rgb.png'.format(img_idx) for
                                    img_idx in img_idxs]
                self.exporter.export_channelvals2d_as_id2rgb(label_pred, predictions_outdir,
                                                             prediction_names)
                groundtruth_names = ['groundtruth_{:06d}_sem255instid2rgb.png'.format(img_idx) for
                                     img_idx in img_idxs]
                self.exporter.export_inst_sem_lbls_as_id2rgb(sem_lbl_np,
                                                             inst_lbl_np, groundtruth_outdir,
                                                             groundtruth_names)
                if 1:
                    image_names = ['image_{:06d}.png'.format(img_idx) for img_idx in img_idxs]
                    for ii, img_idx in enumerate(img_idxs):
                        orig_image, _ = self.exporter.untransform_data(data_loader, img_data[ii],
                                                                       None)
                        out_file = os.path.join(images_outdir, image_names[ii])
                        self.exporter.write_rgb_image(out_file, orig_image)
                    batch_img_idx += batch_sz
        return predictions_outdir, groundtruth_outdir, images_outdir, scores_outdir

    def map_sem_ids_to_sem_vals(self, sem_lbl_channel_ids):
        """
        When training, we load sem_vals as channel idxs rather than their original semantic val (
        which in retrospect
        was dumb, but we're rolling with it for now...)
        This transformation function creates a copy of the 'dumb' semantic label and maps it onto
        the original
        semantic values (e.g. - from the labels table).
        """
        sem_vals = np.nan * np.ones_like(sem_lbl_channel_ids)
        for sem_channel_lbl, sem_val in zip(
                self.instance_problem.semantic_transformed_label_ids,
                self.instance_problem.semantic_vals):
            sem_vals[sem_lbl_channel_ids == sem_channel_lbl] = sem_val
        sem_vals[
            sem_lbl_channel_ids == self.instance_problem.void_value] = \
            self.instance_problem.void_value
        try:
            assert np.all(~np.isnan(sem_vals))
        except AssertionError:
            channel_ids_not_found = set(np.unique(sem_lbl_channel_ids).tolist()) - set(
                self.instance_problem.semantic_transformed_label_ids)
            print('Could not map {} to a semantic value'.format(channel_ids_not_found))
            raise
        return sem_vals

    def validate_split(self, split='val', write_basic_metrics=None, write_instance_metrics=None,
                       should_export_visualizations=None):
        """
        If split == 'val': write_metrics, save_checkpoint, update_best_checkpoint default to True.
        If split == 'train': write_metrics, save_checkpoint, update_best_checkpoint default to
            False.
        """
        if should_export_visualizations is None:
            if self.exporter.conservative_export_decider.is_prev_or_next_export_iteration(self.state.iteration):
                should_export_visualizations = True
            else:
                import ipdb;
                ipdb.set_trace()
                should_export_visualizations = False

        val_metrics = None
        save_checkpoint = (split == 'val') and not self.skip_model_checkpoint_saving
        write_instance_metrics = (split == 'val') and self.write_instance_metrics \
            if write_instance_metrics is None else write_instance_metrics
        write_basic_metrics = True if write_basic_metrics is None else write_basic_metrics
        should_compute_basic_metrics = write_basic_metrics or write_instance_metrics or \
                                       save_checkpoint
        assert split in ['train', 'val']
        if split == 'train':
            data_loader = self.dataloaders['train_for_val']
        else:
            data_loader = self.dataloaders['val']

        # panoeval instead of training mode temporarily
        training = self.model.training
        self.model.eval()

        val_loss = 0
        segmentation_visualizations, score_visualizations = [], []
        label_trues, label_preds, assignments = [], [], []
        num_images_to_visualize = min(len(data_loader), 9)
        memory_allocated_before = torch.cuda.memory_allocated(device=None)
        identifiers = []
        # GPUtil.showUtilization()

        with torch.set_grad_enabled(False):
            t = tqdm.tqdm(
                enumerate(data_loader), total=len(data_loader),
                desc='Valid iteration (split=%s)=%d' %
                     (split, self.state.iteration), ncols=150,
                leave=False)
            for batch_idx, data_dict in t:
                identifiers.append(data_dict['image_id'])
                img_data, lbls = data_dict['image'], (data_dict['sem_lbl'], data_dict['inst_lbl'])
                memory_allocated = sum(torch.cuda.memory_allocated(device=d) for d in
                                       range(torch.cuda.device_count()))
                description = 'Valid iteration=%d, %g GB (%g GB at start)' % \
                              (self.state.epoch, memory_allocated / 1e9, memory_allocated_before
                               / 1e9)
                t.set_description_str(description)

                should_visualize = len(segmentation_visualizations) < num_images_to_visualize
                if not (should_compute_basic_metrics or should_visualize):
                    # Don't waste computation if we don't need to run on the remaining images
                    continue

                score_sb, val_loss_sb, assignments_sb, segmentation_visualizations_sb, \
                score_visualizations_sb = \
                    self.validate_single_batch(img_data, lbls[0], lbls[1], data_loader=data_loader,
                                               should_visualize=should_visualize)
                # print('APD: Memory allocated after validating {} GB'.format(memory_allocated /
                # 1e9))
                val_loss += val_loss_sb
                # scores += [score_sb]  # This takes up way too much memory
                assignments += [assignments_sb]
                segmentation_visualizations += segmentation_visualizations_sb
                score_visualizations += score_visualizations_sb
                # num_collected, mem_collected = torch_utils.garbage_collect(verbose=True)
                if not should_visualize:
                    vars_to_delete = ['score_sb']
                    for var in vars_to_delete:
                        del var

        if should_export_visualizations:
            self.exporter.export_visualizations(segmentation_visualizations, self.state.iteration,
                                                basename='seg_' + split, tile=True)
            if score_visualizations is not None:
                self.exporter.export_visualizations(score_visualizations, self.state.iteration,
                                                    basename='score_' + split, tile=False)
        val_loss /= len(data_loader)
        self.last_val_loss = val_loss

        if should_compute_basic_metrics:
            if write_basic_metrics:
                #         self.exporter.write_eval_metrics(val_metrics, val_loss, split,
                #         epoch=self.state.epoch,
                #                                          iteration=self.state.iteration)
                if self.exporter.tensorboard_writer is not None:
                    self.exporter.tensorboard_writer.add_scalar(
                        'A_eval_metrics/{}/losses'.format(split), val_loss,
                        self.state.iteration)
                    # self.exporter.tensorboard_writer.add_scalar('A_eval_metrics/{
                    # }/mIOU'.format(split), val_metrics[2],
                    #                                             self.state.iteration)
        #
        if write_instance_metrics:
            self.exporter.compute_and_write_instance_metrics(model=self.model,
                                                             iteration=self.state.iteration)
        if save_checkpoint:
            # self.save_checkpoint_and_update_if_best(mean_iu=val_metrics[2],
            #                                         save_by_iteration=save_checkpoint_by_itr_name)
            self.save_checkpoint_and_update_if_best(mean_iu=-val_loss)

        # Restore training settings set prior to function call
        if training:
            self.model.train()

        visualizations = (segmentation_visualizations, score_visualizations)
        return val_loss, val_metrics, visualizations

    def validate_single_batch(self, img_data, sem_lbl, inst_lbl, data_loader, should_visualize):
        with torch.no_grad():
            full_input, sem_lbl, inst_lbl = self.prepare_data_for_forward_pass(
                img_data, (sem_lbl, inst_lbl), requires_grad=False)
            imgs = img_data.cpu()

            score = self.model(full_input)
            # print('APD: Computing loss')
            loss_result = self.compute_loss(score, sem_lbl, inst_lbl, val_matching_override=True)
            assignments, avg_loss, loss_components_by_channel = \
                loss_result.assignments, loss_result.avg_loss, loss_result.loss_components_by_channel
            # print('APD: Finished computing loss')
            val_loss = float(avg_loss.item())
            segmentation_visualizations, score_visualizations = \
                self.exporter.run_post_val_iteration(
                    imgs, sem_lbl, inst_lbl, score, assignments, should_visualize,
                    data_to_img_transformer=lambda i, l: self.exporter.untransform_data(data_loader, i, l))

            # print('APD: Finished iteration')
        return score, val_loss, assignments, segmentation_visualizations, score_visualizations

    def train_epoch(self):
        self.model.train()
        if self.lr_scheduler is not None:
            val_loss, val_metrics, _ = self.validate_split('val')
            self.lr_scheduler.step(val_loss, epoch=self.state.epoch)

        if self.generate_new_synthetic_data_each_epoch:
            seed = np.random.randint(100)
            self.dataloaders['train'].dataset.raw_dataset.initialize_locations_per_image(seed)
            self.dataloaders['train_for_val'].dataset.raw_dataset.initialize_locations_per_image(
                seed)

        t = tqdm.tqdm(  # tqdm: progress bar
            enumerate(self.dataloaders['train']), total=len(self.dataloaders['train']),
            desc='Train epoch=%d' % self.state.epoch, ncols=80, leave=False)

        for batch_idx, data_dict in t:
            if self.t_val is not None:
                self.t_val.update()
            memory_allocated = torch.cuda.memory_allocated(device=None)
            description = 'Train epoch=%d, %g GB' % (self.state.epoch, memory_allocated / 1e9)
            t.set_description_str(description)

            # Check/update iteration
            iteration = batch_idx + self.state.epoch * len(self.dataloaders['train'])
            if self.state.iteration != 0 and (iteration - 1) != self.state.iteration:
                continue  # for resuming
            self.state.iteration = iteration

            # Run validation epochs if it's time
            if self.state.iteration % self.interval_validate == 0:
                if not (self.exporter.export_config.validate_only_on_vis_export and
                        not self.exporter.conservative_export_decider.is_prev_or_next_export_iteration(
                            self.state.iteration)):
                    if not self.skip_validation:
                        self.validate_all_splits()
                    elif not self.skip_model_checkpoint_saving:
                        current_checkpoint_file = self.exporter.save_checkpoint(self.state.epoch,
                                                                                self.state.iteration, self.model,
                                                                                self.optim, self.best_mean_iu,
                                                                                None)

            # Run training iteration
            self.train_iteration(data_dict)

            if self.state.training_complete():
                self.validate_all_splits()
                break

    def train_iteration(self, data_dict):
        assert self.model.training
        img_data = data_dict['image']
        target = (data_dict['sem_lbl'], data_dict['inst_lbl'])
        full_input, sem_lbl, inst_lbl = self.prepare_data_for_forward_pass(img_data, target,
                                                                           requires_grad=True)
        self.optim.zero_grad()
        score = self.model(full_input)
        loss_result = self.compute_loss(score, sem_lbl, inst_lbl, cap_sizes=True)
        avg_loss, loss_components_by_channel = loss_result.avg_loss, loss_result.loss_components_by_channel
        debug_check_values_are_valid(avg_loss, score, self.state.iteration)

        avg_loss.backward()
        self.optim.step()

        if self.exporter.run_loss_updates:
            self.model.eval()
            new_score = self.model(full_input)
            new_loss_result = self.compute_loss(new_score, sem_lbl, inst_lbl)
            self.model.train()
        else:
            new_loss_result = None

        group_lrs = []
        for grp_idx, param_group in enumerate(self.optim.param_groups):
            group_lr = self.optim.param_groups[grp_idx]['lr']
            group_lrs.append(group_lr)
        self.exporter.run_post_train_iteration(
            full_input=full_input, loss_result=loss_result,
            epoch=self.state.epoch, iteration=self.state.iteration,
            new_loss_result=new_loss_result, get_activations_fcn=self.model.module.get_activations
            if isinstance(self.model, torch.nn.DataParallel) else self.model.get_activations,
            lrs_by_group=group_lrs, semantic_names_by_val=self.instance_problem.semantic_class_names_by_model_id)

    def train(self):
        max_epoch = int(math.ceil(1. * self.state.max_iteration / len(self.dataloaders['train'])))
        if self.t_val is None:
            self.t_val = self.get_validation_progress_bar()

        for epoch in tqdm.trange(self.state.epoch, max_epoch,
                                 desc='Train', ncols=80, leave=False):
            self.state.epoch = epoch
            self.train_epoch()
            if self.state.training_complete():
                break
        if self.t_val is not None:
            self.t_val.close()
            if not self.t_val.finished():
                misc.color_text('Validation is continuing.', color='WARNING')
            else:
                misc.color_text('Validation is continuing.', color='OKGREEN')

    def validate_all_splits(self):
        val_loss, val_metrics, _ = self.validate_split('val')
        if self.dataloaders['train_for_val'] is not None:
            train_loss, train_metrics, _ = self.validate_split('train')
        else:
            train_loss, train_metrics = None, None
        if train_loss is not None:
            self.exporter.update_mpl_joint_train_val_loss_figure(train_loss, val_loss, self.state.iteration)
        if self.exporter.tensorboard_writer is not None:
            self.exporter.tensorboard_writer.add_scalar(
                'C_intermediate_metrics/val_minus_train_loss', val_loss - train_loss,
                self.state.iteration)
        return train_metrics, train_loss, val_metrics, val_loss


def debug_check_values_are_valid(loss, score, iteration):
    if is_nan(loss.data.item()):
        raise ValueError('losses is nan while training')
    if loss.data.item() > 1e4:
        print('WARNING: losses={} at iteration {}'.format(loss.data.item(), iteration))
    if any_nan(score.data):
        raise ValueError('score is nan while training')


def argsort_lst(lst):
    alst = np.argsort(lst)
    return alst, [lst[a] for a in alst]


