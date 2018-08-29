import datetime
import os
import os.path as osp
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytz
import torch
import torch.nn.functional as F
import tqdm

import instanceseg.utils.display as display_pyutils
import instanceseg.utils.export
import instanceseg.utils.misc
from instanceseg.analysis import visualization_utils
from instanceseg.analysis.visualization_utils import export_visualizations
from instanceseg.datasets import runtime_transformations
from instanceseg.models.model_utils import any_nan
from instanceseg.train import metrics
from instanceseg.utils import instance_utils
from instanceseg.utils.misc import flatten_dict

MY_TIMEZONE = 'America/New_York'


# TODO(allie): add semantic visualizations (even from instance segmentation runs)


def should_write_activations(iteration, epoch, interval_validate):
    if iteration < 3000:
        return True
    else:
        return False


class MinibatchResult(object):
    """
    A glorified struct to enforce structure
    """
    full_input = None
    img = None
    sem_lbl = None
    inst_lbl = None
    scores = None
    pred_permutations = None
    loss = None

    def __init__(self):
        self.clear()

    def clear(self):
        variables_to_preserve = []
        for attr_name in self.__dict__.keys():
            if attr_name in variables_to_preserve:
                continue
            setattr(self, attr_name, None)

    def set(self, full_input, sem_lbl, inst_lbl, scores, loss, pred_permutations, img=None):
        self.clear()
        self.full_input = full_input
        self.sem_lbl = sem_lbl
        self.inst_lbl = inst_lbl
        self.scores = scores
        self.loss = loss
        self.pred_permutations = pred_permutations
        self.img = img


class ValidationEpochResult(object):
    true_labels_list = None
    predicted_labels_list = None
    pred_permutations_list = None
    num_minibatches_stored = 0
    loss_sum = 0

    def __init__(self):
        self.clear()

    def clear(self):
        self.num_minibatches_stored = 0
        self.true_labels_list = []
        self.predicted_labels_list = []
        self.pred_permutations_list = []
        self.loss_sum = 0

    def append(self, true_labels_combined_mb, predicted_labels_combined_mb, pred_permutations_mb, loss):
        """
        _mb postfix refers to minibatch
        """
        self.true_labels_list.append(true_labels_combined_mb)
        self.predicted_labels_list.append(predicted_labels_combined_mb)
        self.pred_permutations_list.append(pred_permutations_mb)
        self.num_minibatches_stored += 1
        self.loss_sum += loss

    @property
    def loss_avg(self):
        return self.loss_sum / self.num_minibatches_stored


class TrainerExporter(object):
    log_headers = [
        'epoch',
        'iteration',
        'train/losses',
        'train/acc',
        'train/acc_cls',
        'train/mean_iu',
        'train/fwavacc',
        'valid/losses',
        'valid/acc',
        'valid/acc_cls',
        'valid/mean_iu',
        'valid/fwavacc',
        'elapsed_time',
    ]

    def __init__(self, out_dir, trainer, tensorboard_writer=None, export_activations=False,
                 activation_layers_to_export=(), write_activation_condition=should_write_activations,
                 write_instance_metrics=True):

        # 'Parent' object for proxies
        self._trainer = trainer

        # Helper objects
        self.tensorboard_writer = tensorboard_writer

        # Log directory / log files
        self.out_dir = out_dir
        if not osp.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if not osp.exists(osp.join(self.out_dir, 'log.csv')):
            with open(osp.join(self.out_dir, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        # Logging parameters
        self.timestamp_start = datetime.datetime.now(pytz.timezone(MY_TIMEZONE))

        # Stored values
        self.last_val_loss = None
        self.val_losses_stored = []
        self.train_losses_stored = []
        self.joint_train_val_loss_mpl_figure = None  # figure for plotting losses on same plot
        self.state.iterations_for_losses_stored = []

        # Writing activations
        self.export_activations = export_activations
        self.activation_layers_to_export = activation_layers_to_export
        self.write_activation_condition = write_activation_condition
        self.write_instance_metrics = write_instance_metrics

        # What to do after validation
        self.write_train_evaluation_metrics = True
        self.write_val_evaluation_metrics = True
        self.save_checkpoint = None
        self.update_best_checkpoint = None
        self.num_images_to_visualize_per_split = 9

        # Exporting parameters
        self.which_heatmaps_to_visualize = 'same semantic'  # 'all'

        metric_maker_kwargs = {
            'problem_config': self.instance_problem,
            'component_loss_function': self._trainer.loss_fcn,
            'augment_function_img_sem': self._trainer.augment_image if self._trainer.augment_input_with_semantic_masks
            else None
        }
        self.metric_makers = {
            'val': metrics.InstanceMetrics(self._trainer.val_loader, **metric_maker_kwargs),
            'train_for_val': metrics.InstanceMetrics(self._trainer.train_loader_for_val, **metric_maker_kwargs)
        }

        # Current state
        self.last_val_iteration = None
        self.last_train_iteration = None
        self.last_val_minibatch_result = MinibatchResult()
        self.last_train_minibatch_result = MinibatchResult()
        self.val_epoch_result = ValidationEpochResult()
        self.train_for_val_epoch_result = ValidationEpochResult()
        self.segmentation_visualizations = {
            'train': [],
            'val': []
        }
        self.score_visualizations = {
            'train': [],
            'val': []
        }
        self.best_mean_iu = 0

    def continue_validation_iterations(self, split):
        return self.write_val_evaluation_metrics or self.update_best_checkpoint or \
               len(self.segmentation_visualizations[split]) < self.num_images_to_visualize_per_split

    @property
    def state(self):
        return self._trainer.state

    @property
    def model(self):
        return self._trainer.model

    @property
    def instance_problem(self):
        return self._trainer.instance_problem

    def val_visualization_quota_has_been_met(self, split):
        return len(self.segmentation_visualizations[split]) >= self.num_images_to_visualize_per_split

    def update_after_validation_epoch(self, split):
        val_metrics = None
        write_instance_metrics = (split == 'train') and self.write_instance_metrics
        write_basic_metrics = \
            True if ((split == 'train' and self.write_train_evaluation_metrics)
                     or (split == 'val' and self.write_val_evaluation_metrics)) \
                else False
        save_checkpoint = (split == 'val') and self.save_checkpoint
        update_best_checkpoint = (split == 'val') and self.update_best_checkpoint
        should_compute_basic_metrics = \
            write_basic_metrics or write_instance_metrics or save_checkpoint or update_best_checkpoint

        if should_compute_basic_metrics:
            val_metrics = self.compute_evaluation_metrics(self.val_epoch_result.true_labels_list,
                                                          self.val_epoch_result.predicted_labels_list,
                                                          self.val_epoch_result.pred_permutations_list)
            if write_basic_metrics:
                self.write_metrics(val_metrics, self.val_epoch_result.loss_avg, split)
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar('metrics/{}/losses'.format(split),
                                                       self.val_epoch_result.loss_avg, self.state.iteration)
                    self.tensorboard_writer.add_scalar('metrics/{}/mIOU'.format(split), val_metrics[2],
                                                       self.state.iteration)
            if save_checkpoint:
                self.save_current_checkpoint()
            if update_best_checkpoint:
                self.update_best_model_and_checkpoint_if_best(mean_iu=val_metrics[2])
        if write_instance_metrics:
            self.compute_and_write_instance_metrics()

        self.export_visualizations(self.segmentation_visualizations[split], 'seg_' + split, tile=True)
        self.export_visualizations(self.score_visualizations[split], 'score_' + split, tile=False)

        val_loss = self.last_val_loss
        if self._trainer.train_loader_for_val is not None:
            train_metrics, _ = self._trainer.validate('train')  # TODO(allie): Don't violate the 'don't run forward
            # pass inside exporter' rule we set for ourselves!!
            train_loss = self.last_val_loss
        else:
            print('Warning: cannot generate train vs. val plots if we dont have access to the training losses '
                  'via train_for_val dataloader')
            train_loss = None
        if train_loss is not None:
            self.update_mpl_joint_train_val_loss_figure(train_loss, val_loss)
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('val_minus_train_loss', val_loss - train_loss,
                                                   self.state.iteration)

    def compute_evaluation_metrics(self, label_trues, label_preds, permutations=None, single_batch=False):
        if permutations is not None:
            if single_batch:
                permutations = [permutations]
            assert type(permutations) == list, \
                NotImplementedError('I''m assuming permutations are a list of ndarrays from multiple batches, '
                                    'not type {}'.format(type(permutations)))
            label_preds_permuted = [instance_utils.permute_labels(label_pred, perms)
                                    for label_pred, perms in zip(label_preds, permutations)]
        else:
            label_preds_permuted = label_preds
        metrics_list = instanceseg.utils.misc.label_accuracy_score(label_trues, label_preds_permuted,
                                                                   n_class=self.instance_problem.n_classes)
        return metrics_list

    def write_metrics(self, metrics_list, loss, split):
        with open(osp.join(self.out_dir, 'log.csv'), 'a') as f:
            elapsed_time = (
                    datetime.datetime.now(pytz.timezone(MY_TIMEZONE)) -
                    self.timestamp_start).total_seconds()
            if split == 'val':
                log = [self.state.epoch, self.state.iteration] + [''] * 5 + [loss] + list(metrics_list) + [elapsed_time]
            elif split == 'train':
                try:
                    metrics_as_list = metrics_list.tolist()
                except:
                    metrics_as_list = list(metrics_list)
                log = [self.state.epoch, self.state.iteration] + [loss] + metrics_as_list + [''] * 5 + [elapsed_time]
            else:
                raise ValueError('split not recognized')
            log = map(str, log)
            f.write(','.join(log) + '\n')

    def export_visualizations(self, visualizations, basename='val_', tile=True, outdir=None):
        outdir = outdir or osp.join(self.out_dir, 'visualization_viz')
        export_visualizations(visualizations, outdir, self.tensorboard_writer, self.state.iteration, basename=basename,
                              tile=tile)

    def save_current_checkpoint(self):
        torch.save({
            'epoch': self.state.epoch,
            'iteration': self.state.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self._trainer.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out_dir, 'checkpoint.pth.tar'))

    def update_best_model_and_checkpoint_if_best(self, mean_iu):
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
            shutil.copy(osp.join(self.out_dir, 'checkpoint.pth.tar'),
                        osp.join(self.out_dir, 'model_best.pth.tar'))

    def compute_and_write_instance_metrics(self):
        if self.tensorboard_writer is not None:
            for split, metric_maker in tqdm.tqdm(self.metric_makers.items(), desc='Computing instance metrics',
                                                 total=len(self.metric_makers.items()), leave=False):
                metric_maker.clear()
                metric_maker.compute_metrics(self.model)
                metrics_as_nested_dict = metric_maker.get_aggregated_scalar_metrics_as_nested_dict()
                metrics_as_flattened_dict = flatten_dict(metrics_as_nested_dict)
                for name, metric in metrics_as_flattened_dict.items():
                    self.tensorboard_writer.add_scalar('instance_metrics_{}/{}'.format(split, name), metric,
                                                       self.state.iteration)
                histogram_metrics_as_nested_dict = metric_maker.get_aggregated_histogram_metrics_as_nested_dict()
                histogram_metrics_as_flattened_dict = flatten_dict(histogram_metrics_as_nested_dict)
                if self.state.iteration != 0:  # screws up the axes if we do it on the first iteration with weird inits
                    # if 1:
                    for name, metric in tqdm.tqdm(histogram_metrics_as_flattened_dict.items(),
                                                  total=len(histogram_metrics_as_flattened_dict.items()),
                                                  desc='Writing histogram metrics', leave=False):
                        if torch.is_tensor(metric):
                            self.tensorboard_writer.add_histogram('instance_metrics_{}/{}'.format(split, name),
                                                                  metric.numpy(), self.state.iteration, bins='auto')
                        elif isinstance(metric, np.ndarray):
                            self.tensorboard_writer.add_histogram('instance_metrics_{}/{}'.format(split, name), metric,
                                                                  self.state.iteration, bins='auto')
                        elif metric is None:
                            import ipdb;
                            ipdb.set_trace()
                            pass
                        else:
                            raise ValueError('I\'m not sure how to write {} to tensorboard_writer (name is '
                                             ' '.format(type(metric), name))

    def retrieve_and_write_batch_activations(self, batch_input):
        if self.tensorboard_writer is not None:
            activations = self.model.get_activations(batch_input, self.activation_layers_to_export)
            histogram_activations = activations
            for name, activations in tqdm.tqdm(histogram_activations.items(),
                                               total=len(histogram_activations.items()),
                                               desc='Writing activation distributions', leave=False):
                if name == 'upscore8':
                    channel_labels = self.instance_problem.get_model_channel_labels('{}_{}')
                    assert activations.size(1) == len(channel_labels), '{} != {}'.format(activations.size(1),
                                                                                         len(channel_labels))
                    for c, channel_label in enumerate(channel_labels):
                        self.tensorboard_writer.add_histogram('batch_activations/{}/{}'.format(name, channel_label),
                                                              activations[:, c, :, :].cpu().numpy(),
                                                              self.state.iteration, bins='auto')
                elif name == 'conv1x1_instance_to_semantic':
                    channel_labels = self.instance_problem.get_channel_labels('{}_{}')
                    assert activations.size(1) == len(channel_labels)
                    for c, channel_label in enumerate(channel_labels):
                        try:
                            self.tensorboard_writer.add_histogram('batch_activations/{}/{}'.format(name, channel_label),
                                                                  activations[:, c, :, :].cpu().numpy(),
                                                                  self.state.iteration, bins='auto')
                        except IndexError as ie:
                            print('WARNING: Didn\'t write activations.  IndexError: {}'.format(ie))
                elif name == 'conv1_1':
                    # This is expensive to write, so we'll just write a representative set.
                    min = torch.min(activations)
                    max = torch.max(activations)
                    mean = torch.mean(activations)
                    representative_set = np.ndarray((100, 3))
                    representative_set[:, 0] = min
                    representative_set[:, 1] = max
                    representative_set[:, 2] = mean
                    self.tensorboard_writer.add_histogram('batch_activations/{}/min_mean_max_all_channels'.format(name),
                                                          representative_set, self.state.iteration, bins='auto')
                    continue

                self.tensorboard_writer.add_histogram('batch_activations/{}/all_channels'.format(name),
                                                      activations.cpu().numpy(), self.state.iteration, \
                                                      bins='auto')

    def update_mpl_joint_train_val_loss_figure(self, train_loss, val_loss):
        assert train_loss is not None, ValueError
        assert val_loss is not None, ValueError
        figure_name = 'train/val losses'
        ylim_buffer_size = 3
        self.train_losses_stored.append(train_loss)
        self.val_losses_stored.append(val_loss)

        self.state.iterations_for_losses_stored.append(self.state.iteration)
        if self.joint_train_val_loss_mpl_figure is None:
            self.joint_train_val_loss_mpl_figure = plt.figure(figure_name)
            display_pyutils.set_my_rc_defaults()

        h = plt.figure(figure_name)

        plt.clf()
        train_label = 'train losses: ' + 'last epoch of images: {}'.format(len(self._trainer.train_loader)) if \
            self._trainer.generate_new_synthetic_data_each_epoch else '{} images'.format(
            len(self._trainer.train_loader_for_val))
        val_label = 'val losses: ' + '{} images'.format(len(self._trainer.val_loader))

        plt.plot(self.state.iterations_for_losses_stored, self.train_losses_stored, label=train_label,
                 color=display_pyutils.GOOD_COLORS_BY_NAME['blue'])
        plt.plot(self.state.iterations_for_losses_stored, self.val_losses_stored, label=val_label,
                 color=display_pyutils.GOOD_COLORS_BY_NAME['aqua'])
        plt.xlabel('iteration')
        plt.legend()
        # Set y limits for just the last 10 datapoints
        last_x = max(len(self.train_losses_stored), len(self.val_losses_stored))
        if last_x >= 0:
            ymin = min(min(self.train_losses_stored[(last_x - ylim_buffer_size - 1):]),
                       min(self.val_losses_stored[(last_x - ylim_buffer_size - 1):]))
            ymax = max(max(self.train_losses_stored[(last_x - ylim_buffer_size - 1):]),
                       max(self.val_losses_stored[(last_x - ylim_buffer_size - 1):]))
        else:
            ymin, ymax = None, None
        if self.tensorboard_writer is not None:
            instanceseg.utils.export.log_plots(self.tensorboard_writer, 'joint_loss', [h], self.state.iteration)
        filename = os.path.join(self.out_dir, 'val_train_loss.png')
        h.savefig(filename)

        # zoom
        zoom_filename = os.path.join(self.out_dir, 'val_train_loss_zoom_last_{}.png'.format(ylim_buffer_size))
        if ymin is not None:
            plt.ylim(ymin=ymin, ymax=ymax)
            plt.xlim(xmin=(last_x - ylim_buffer_size - 1), xmax=last_x)
            if self.tensorboard_writer is not None:
                instanceseg.utils.export.log_plots(self.tensorboard_writer,
                                                   'joint_loss_last_{}'.format(ylim_buffer_size),
                                                   [h], self.state.iteration)
            h.savefig(zoom_filename)
        else:
            shutil.copyfile(filename, zoom_filename)

    def update_after_train_minibatch(self, full_input, score, sem_lbl, inst_lbl, pred_permutations, loss):
        """
        Happens every iteration.  Computed quantities / predictions, etc. that need to be kept around must be stored
        or written here before the trainer moves on to the next batch.
        """
        assert not self._trainer.is_running_validation

        self.last_train_minibatch_result.set(full_input, sem_lbl, inst_lbl, score, loss, pred_permutations)

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar('metrics/train_batch_loss', loss.data[0], self.state.iteration)

        inst_lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true_sem, lbl_true_inst = sem_lbl.data.cpu().numpy(), inst_lbl.data.cpu().numpy()
        metrics_list = []
        for sem_lbl_np, inst_lbl_np, lp in zip(lbl_true_sem, lbl_true_inst, inst_lbl_pred):
            lt_combined = self._trainer.gt_tuple_to_combined(sem_lbl_np, inst_lbl_np)
            acc, acc_cls, mean_iu, fwavacc = \
                self.compute_evaluation_metrics(label_trues=[lt_combined], label_preds=[lp],
                                                permutations=[pred_permutations])
            metrics_list.append((acc, acc_cls, mean_iu, fwavacc))
        metrics_list = np.mean(metrics_list, axis=0)
        self.write_metrics(metrics_list, loss, split='train')

        if self.tensorboard_writer is not None:
            self.model.eval()
            new_score = self.model(full_input)
            if any_nan(new_score.data):
                raise ValueError('new_score became nan while training')
            new_pred_permutations, new_loss, _ = self._trainer.compute_loss(new_score, sem_lbl, inst_lbl)
            new_loss /= len(full_input)
            loss_improvement = loss.data[0] - new_loss.data[0]
            self.model.train()

            self.tensorboard_writer.add_scalar('metrics/train_batch_loss_improvement', loss_improvement,
                                               self.state.iteration)
            self.tensorboard_writer.add_scalar('metrics/reassignment',
                                               np.sum(new_pred_permutations != pred_permutations),
                                               self.state.iteration)
            if self.export_activations \
                    and self.write_activation_condition(iteration=self.state.iteration, epoch=self.state.epoch,
                                                        interval_validate=self._trainer.interval_validate):
                self.retrieve_and_write_batch_activations(full_input)

    def update_after_val_minibatch(self, full_input, sem_lbl, inst_lbl, score, pred_permutations, loss,
                                   dataloader_runtime_transformation, split, img=None):

        self.last_val_minibatch_result.set(full_input, sem_lbl, inst_lbl, score, pred_permutations, loss, img=img)
        inst_lbl_pred = score.data.max(dim=1)[1].cpu().numpy()[:, :, :]

        lbl_true_sem, lbl_true_inst = (sem_lbl.data.cpu(), inst_lbl.data.cpu())
        list_true_labels_combined = []
        list_pred_labels_combined = []
        for idx, (img, sem_lbl, inst_lbl, lp) in enumerate(zip(img, lbl_true_sem, lbl_true_inst, inst_lbl_pred)):
            # runtime_transformation needs to still run the resize, even for untransformed img, lbl pair
            if dataloader_runtime_transformation is not None:
                runtime_transformation_undo = runtime_transformations.GenericSequenceRuntimeDatasetTransformer(
                    [t for t in (dataloader_runtime_transformation.transformer_sequence or [])
                     if isinstance(t, runtime_transformations.BasicRuntimeDatasetTransformer)])
                img_untransformed, lbl_untransformed = runtime_transformation_undo.untransform(img, (sem_lbl, inst_lbl))
            else:
                img_untransformed, lbl_untransformed = img, (sem_lbl, inst_lbl)
            sem_lbl_np = lbl_untransformed[0]
            inst_lbl_np = lbl_untransformed[1]

            pp = pred_permutations[idx, :]
            lt_combined = self._trainer.gt_tuple_to_combined(sem_lbl_np, inst_lbl_np)
            list_true_labels_combined.append(lt_combined)
            list_pred_labels_combined.append(lp)

            if not self.val_visualization_quota_has_been_met(split):
                # Generate per-image visualizations
                softmax_scores = F.softmax(score, dim=1).data.cpu().numpy()
                sp = softmax_scores[idx, :, :, :]
                viz_score, viz_segmentation = self.generate_visualizations_for_one_minibatch(
                    score, lp, lt_combined, pp, img_untransformed)

                self.score_visualizations[split].append(viz_score)
                self.segmentation_visualizations[split].append(viz_segmentation)

            # Store results
            if self.write_val_evaluation_metrics:
                self.val_epoch_result.append(true_labels_combined_mb=list_true_labels_combined,
                                             predicted_labels_combined_mb=list_pred_labels_combined,
                                             pred_permutations_mb=pred_permutations, loss=loss)

    def generate_visualizations_for_one_minibatch(self, sp, lp, lt_combined, pp, img_untransformed):
        viz_segmentation = visualization_utils.visualize_segmentation(
            lbl_pred=lp, lbl_true=lt_combined, pred_permutations=pp, img=img_untransformed,
            n_class=self.instance_problem.n_classes, overlay=False)

        # # Scores
        # softmax_scores = F.softmax(score, dim=1).data.cpu().numpy()
        # sp = softmax_scores[idx, :, :, :]
        #
        # # TODO(allie): Fix this -- bug(?!)
        # lp = np.argmax(sp, axis=0)

        if self.which_heatmaps_to_visualize == 'same semantic':
            inst_sem_classes_present = torch.np.unique(lt_combined)
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
        viz_score = visualization_utils.visualize_heatmaps(scores=sp,
                                                           lbl_true=lt_combined,
                                                           lbl_pred=lp,
                                                           pred_permutations=pp,
                                                           n_class=self._trainer.instance_problem.n_classes,
                                                           score_vis_normalizer=sp.max(),
                                                           channel_labels=channel_labels,
                                                           channels_to_visualize=channels_to_visualize,
                                                           input_image=img_untransformed)
        return viz_score, viz_segmentation


def gt_tuple_to_combined(self, sem_lbl, inst_lbl):
    semantic_instance_class_list = self.instance_problem.semantic_instance_class_list
    instance_count_id_list = self.instance_problem.instance_count_id_list
    return instance_utils.combine_semantic_and_instance_labels(sem_lbl, inst_lbl,
                                                               semantic_instance_class_list,
                                                               instance_count_id_list)
