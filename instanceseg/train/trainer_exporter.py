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
from instanceseg.analysis import visualization_utils
from instanceseg.datasets import runtime_transformations
from instanceseg.losses.loss import LossMatchAssignments
from instanceseg.utils import instance_utils
from instanceseg.utils.misc import flatten_dict
from instanceseg.utils.instance_utils import InstanceProblemConfig
from tensorboardX import SummaryWriter
from instanceseg.ext.panopticapi.utils import rgb2id, id2rgb

display_pyutils.set_my_rc_defaults()

MY_TIMEZONE = 'America/New_York'


def should_write_activations(iteration, epoch):
    if iteration < 3000:
        return True
    else:
        return False


DEBUG_ASSERTS = True


class ExportConfig(object):
    def __init__(self, export_activations=None, activation_layers_to_export=(), write_instance_metrics=False,
                 run_loss_updates=True):
        self.export_activations = export_activations
        self.activation_layers_to_export = activation_layers_to_export
        self.write_instance_metrics = write_instance_metrics
        self.run_loss_updates = run_loss_updates

        self.write_activation_condition = should_write_activations
        self.which_heatmaps_to_visualize = 'same semantic'  # 'all'

        self.downsample_multiplier_score_images = 0.5
        self.export_component_losses = True

        self.write_lr = True


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

    def __init__(self, out_dir, instance_problem: InstanceProblemConfig, export_config: ExportConfig = None,
                 tensorboard_writer: SummaryWriter=None, metric_makers=None):

        self.export_config = export_config or ExportConfig()

        # Copies of things the trainer was given access to
        self.instance_problem = instance_problem

        # Helper objects
        self.tensorboard_writer = tensorboard_writer

        # Log directory / log files
        self.out_dir = out_dir
        if not osp.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if not osp.exists(osp.join(self.out_dir, 'log.csv')):
            with open(osp.join(self.out_dir, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        # Save instance problem
        self.instance_problem.save(self.instance_problem_path)

        # Logging parameters
        self.timestamp_start = datetime.datetime.now(pytz.timezone(MY_TIMEZONE))

        self.val_losses_stored = []
        self.train_losses_stored = []
        self.joint_train_val_loss_mpl_figure = None  # figure for plotting losses on same plot
        self.iterations_for_losses_stored = []

        self.metric_makers = metric_makers

        # Writing activations

        self.run_loss_updates = True

    @property
    def instance_problem_path(self):
        return osp.join(self.out_dir, 'instance_problem_config.yaml')

    def write_eval_metrics(self, eval_metrics, loss, split, epoch, iteration):
        with open(osp.join(self.out_dir, 'log.csv'), 'a') as f:
            elapsed_time = (
                    datetime.datetime.now(pytz.timezone(MY_TIMEZONE)) -
                    self.timestamp_start).total_seconds()
            if split == 'val':
                log = [epoch, iteration] + [''] * 5 + \
                      [loss] + list(eval_metrics) + [elapsed_time]
            elif split == 'train':
                try:
                    eval_metrics_as_list = eval_metrics.tolist()
                except:
                    eval_metrics_as_list = list(eval_metrics)
                log = [epoch, iteration] + [loss] + eval_metrics_as_list + [''] * 5 + [elapsed_time]
            else:
                raise ValueError('split not recognized')
            log = map(str, log)
            f.write(','.join(log) + '\n')

    def update_mpl_joint_train_val_loss_figure(self, train_loss, val_loss, iteration):
        assert train_loss is not None, ValueError
        assert val_loss is not None, ValueError
        figure_name = 'train/val losses'
        ylim_buffer_size = 3
        self.train_losses_stored.append(train_loss)
        self.val_losses_stored.append(val_loss)

        self.iterations_for_losses_stored.append(iteration)
        if self.joint_train_val_loss_mpl_figure is None:
            self.joint_train_val_loss_mpl_figure = plt.figure(figure_name)

        h = plt.figure(figure_name)

        plt.clf()
        train_label = 'train losses'  # TODO(allie): record number of images somewhere.. (we deleted it from here)
        val_label = 'val losses'

        plt.plot(self.iterations_for_losses_stored, self.train_losses_stored, label=train_label,
                 color=display_pyutils.GOOD_COLORS_BY_NAME['blue'])
        plt.plot(self.iterations_for_losses_stored, self.val_losses_stored, label=val_label,
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
            instanceseg.utils.export.log_plots(self.tensorboard_writer, 'joint_loss', [h], iteration)
        filename = os.path.join(self.out_dir, 'val_train_loss.png')
        h.savefig(filename)

        # zoom
        zoom_filename = os.path.join(self.out_dir, 'val_train_loss_zoom_last_{}.png'.format(ylim_buffer_size))
        if ymin is not None:
            plt.ylim(bottom=ymin, top=ymax)  # ymin, ymax, xmin, xmax deprecated Matplotlib 3.0
            plt.xlim(left=(last_x - ylim_buffer_size - 1), right=last_x)
            if self.tensorboard_writer is not None:
                instanceseg.utils.export.log_plots(self.tensorboard_writer,
                                                   'joint_loss_last_{}'.format(ylim_buffer_size),
                                                   [h], iteration)
            h.savefig(zoom_filename)
        else:
            shutil.copyfile(filename, zoom_filename)

    def retrieve_and_write_batch_activations(self, batch_input, iteration,
                                             get_activations_fcn):
        """
        get_activations_fcn: example in FCN8sInstance.get_activations(batch_input, layer_names)
        """
        if self.tensorboard_writer is not None:
            activations = get_activations_fcn(batch_input, self.export_config.activation_layers_to_export)
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
                                                              iteration, bins='auto')
                elif name == 'conv1x1_instance_to_semantic':
                    channel_labels = self.instance_problem.get_channel_labels('{}_{}')
                    assert activations.size(1) == len(channel_labels)
                    for c, channel_label in enumerate(channel_labels):
                        try:
                            self.tensorboard_writer.add_histogram('batch_activations/{}/{}'.format(name, channel_label),
                                                                  activations[:, c, :, :].cpu().numpy(),
                                                                  iteration, bins='auto')
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
                                                          representative_set, iteration, bins='auto')
                    continue

                self.tensorboard_writer.add_histogram('batch_activations/{}/all_channels'.format(name),
                                                      activations.cpu().numpy(), iteration, bins='auto')

    def write_loss_updates(self, old_loss, new_loss, old_assignments: LossMatchAssignments,
                           new_assignments: LossMatchAssignments, iteration):
        loss_improvement = old_loss - new_loss
        num_reassignments = float(
            np.sum(new_assignments.assigned_gt_inst_vals != old_assignments.assigned_gt_inst_vals))
        self.tensorboard_writer.add_scalar('A_eval_metrics/train_minibatch_loss_improvement', loss_improvement,
                                           iteration)
        self.tensorboard_writer.add_scalar('A_eval_metrics/reassignment', num_reassignments, iteration)

    def compute_and_write_instance_metrics(self, model, iteration):
        if self.tensorboard_writer is not None:
            for split, metric_maker in tqdm.tqdm(self.metric_makers.items(), desc='Computing instance metrics',
                                                 total=len(self.metric_makers.items()), leave=False):
                metric_maker.clear()
                metric_maker.compute_metrics(model)
                metrics_as_nested_dict = metric_maker.get_aggregated_scalar_metrics_as_nested_dict()
                metrics_as_flattened_dict = flatten_dict(metrics_as_nested_dict)
                for name, metric in metrics_as_flattened_dict.items():
                    self.tensorboard_writer.add_scalar('C_{}_{}'.format(name, split), metric,
                                                       iteration)
                histogram_metrics_as_nested_dict = metric_maker.get_aggregated_histogram_metrics_as_nested_dict()
                histogram_metrics_as_flattened_dict = flatten_dict(histogram_metrics_as_nested_dict)
                if iteration != 0:  # screws up the axes if we do it on the first iteration with weird inits
                    for name, metric in tqdm.tqdm(histogram_metrics_as_flattened_dict.items(),
                                                  total=len(histogram_metrics_as_flattened_dict.items()),
                                                  desc='Writing histogram metrics', leave=False):
                        if torch.is_tensor(metric):
                            self.tensorboard_writer.add_histogram('C_instance_metrics_{}/{}'.format(split, name),
                                                                  metric.numpy(), iteration, bins='auto')
                        elif isinstance(metric, np.ndarray):
                            self.tensorboard_writer.add_histogram('C_instance_metrics_{}/{}'.format(split, name),
                                                                  metric, iteration, bins='auto')
                        elif metric is None:
                            import ipdb;
                            ipdb.set_trace()
                            pass
                        else:
                            raise ValueError('I\'m not sure how to write {} to tensorboard_writer (name is '
                                             '{}'.format(type(metric), name))

    def save_checkpoint(self, epoch, iteration, model, optimizer, best_mean_iu, mean_iu, out_dir=None,
                        save_by_iteration=True):
        out_name = 'checkpoint.pth.tar'
        out_dir = out_dir or os.path.join(self.out_dir)
        checkpoint_file = osp.join(out_dir, out_name)
        if hasattr(self, 'module'):
            model_state_dict = model.module.state_dict()  # nn.DataParallel
        else:
            model_state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'arch': model.__class__.__name__,
            'optim_state_dict': optimizer.state_dict(),
            'model_state_dict': model_state_dict,
            'best_mean_iu': best_mean_iu,
            'mean_iu': mean_iu
        }, checkpoint_file)

        if save_by_iteration:
            itr_out_name = 'checkpoint-{:09d}.pth.tar'.format(iteration)
            itr_out_dir = osp.join(out_dir, 'model_checkpoints')
            if not osp.isdir(itr_out_dir):
                os.makedirs(itr_out_dir)
            shutil.copyfile(checkpoint_file, osp.join(itr_out_dir, itr_out_name))
        return checkpoint_file

    def copy_checkpoint_as_best(self, current_checkpoint_file, out_dir=None, out_name='model_best.pth.tar'):
        out_dir = out_dir or self.out_dir
        best_checkpoint_file = osp.join(out_dir, out_name)
        shutil.copy(current_checkpoint_file, best_checkpoint_file)
        return best_checkpoint_file

    def visualize_one_img_prediction(self, img_untransformed, lp, lt_combined, softmax_scores,
                                     true_labels, idx):
        # Segmentations
        segmentation_viz = visualization_utils.visualize_segmentation(
            lbl_pred=lp, lbl_true=lt_combined, img=img_untransformed, n_class=self.instance_problem.n_classes,
            overlay=False)
        # Scores
        sp = softmax_scores[idx, :, :, :]
        # TODO(allie): Fix this -- bug(?!)
        lp = np.argmax(sp, axis=0)
        if self.export_config.which_heatmaps_to_visualize == 'same semantic':
            inst_sem_classes_present = torch.from_numpy(np.unique(true_labels))  # torch.np.unique
            inst_sem_classes_present = inst_sem_classes_present[inst_sem_classes_present != -1]
            sem_classes_present = np.unique([self.instance_problem.semantic_instance_class_list[c]
                                             for c in inst_sem_classes_present])
            channels_for_these_semantic_classes = [inst_idx for inst_idx, sem_cls in enumerate(
                self.instance_problem.semantic_instance_class_list) if sem_cls in sem_classes_present]
            channels_to_visualize = channels_for_these_semantic_classes
        elif self.export_config.which_heatmaps_to_visualize == 'all':
            channels_to_visualize = list(range(sp.shape[0]))
        else:
            raise ValueError('which heatmaps to visualize is not recognized: {}'.format(
                self.export_config.which_heatmaps_to_visualize))
        channel_labels = self.instance_problem.get_channel_labels('{} {}')
        score_viz = visualization_utils.visualize_heatmaps(scores=sp,
                                                           lbl_true=lt_combined,
                                                           lbl_pred=lp,
                                                           n_class=self.instance_problem.n_classes,
                                                           score_vis_normalizer=sp.max(),
                                                           channel_labels=channel_labels,
                                                           channels_to_visualize=channels_to_visualize,
                                                           input_image=img_untransformed)
        if self.export_config.downsample_multiplier_score_images != 1:
            score_viz = visualization_utils.resize_img_by_multiplier(
                score_viz, self.export_config.downsample_multiplier_score_images)
        return segmentation_viz, score_viz

    def export_score_and_seg_images(self, segmentation_visualizations, score_visualizations, iteration, split):
        self.export_visualizations(segmentation_visualizations, iteration, basename='seg_' + split, tile=True)
        if score_visualizations is not None:
            self.export_visualizations(score_visualizations, iteration, basename='score_' + split, tile=False)

    def export_visualizations(self, visualizations, iteration, basename='val_', tile=True, out_dir=None):
        if visualizations is None:
            return
        out_dir = out_dir or osp.join(self.out_dir, 'visualization_viz')
        visualization_utils.export_visualizations(visualizations, out_dir, self.tensorboard_writer, iteration,
                                                  basename=basename, tile=tile)

    def run_post_val_epoch(self, label_preds, label_trues, should_compute_basic_metrics, split, val_loss, val_metrics,
                           write_basic_metrics, write_instance_metrics, epoch, iteration, model):
        if should_compute_basic_metrics:
            val_metrics = self.compute_eval_metrics(label_trues, label_preds)
            if write_basic_metrics:
                self.write_eval_metrics(val_metrics, val_loss, split, epoch=epoch, iteration=iteration)
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar('A_eval_metrics/{}/losses'.format(split), val_loss, iteration)
                    self.tensorboard_writer.add_scalar('A_eval_metrics/{}/mIOU'.format(split), val_metrics[2],
                                                       iteration)

        if write_instance_metrics:
            self.compute_and_write_instance_metrics(model=model, iteration=iteration)
        return val_metrics

    def run_post_train_iteration(self, full_input, sem_lbl, inst_lbl, loss, loss_components,
                                 assignments: LossMatchAssignments, score,
                                 epoch,
                                 iteration, new_assignments=None, new_loss=None, get_activations_fcn=None,
                                 lrs_by_group=None):
        """
        get_activations_fcn=self.model.get_activations
        """
        inst_lbl_pred = score.detach().max(1)[1].cpu().numpy()[:, :, :]
        lbl_true_sem, lbl_true_inst = sem_lbl.detach().cpu().numpy(), inst_lbl.detach().cpu().numpy()
        eval_metrics = []
        for idx, (sem_lbl_np, inst_lbl_np, lp) in enumerate(zip(lbl_true_sem, lbl_true_inst, inst_lbl_pred)):
            assigned_sem_vals = [self.instance_problem.semantic_instance_class_list[c]
                                 for c in assignments.model_channels[idx, :]]
            assigned_inst_vals = [self.instance_problem.instance_count_id_list[c]
                                  for c in assignments.model_channels[idx, :]]
            lt_combined = self.gt_tuple_to_channelwise_combined(sem_lbl_np, inst_lbl_np,
                                                                assigned_sem_vals=assigned_sem_vals,
                                                                assigned_inst_vals=assigned_inst_vals)
            acc, acc_cls, mean_iu, fwavacc = \
                self.compute_eval_metrics(label_trues=[lt_combined], label_preds=[lp])
            eval_metrics.append((acc, acc_cls, mean_iu, fwavacc))
        eval_metrics = np.mean(eval_metrics, axis=0)
        self.write_eval_metrics(eval_metrics, loss, split='train', epoch=epoch, iteration=iteration)
        if self.tensorboard_writer is not None:
        # TODO(allie): Check dimensionality of loss to prevent potential bugs
            self.tensorboard_writer.add_scalar('A_eval_metrics/train_minibatch_loss', loss.detach().sum(),
                                               iteration)

        if self.export_config.write_lr:
            for group_idx, lr in enumerate(lrs_by_group):
                self.tensorboard_writer.add_scalar('Z_hyperparameters/lr_group{}'.format(group_idx), lr, iteration)

        if self.export_config.export_component_losses:
            for c_idx, c_lbl in enumerate(self.instance_problem.get_model_channel_labels('{}_{}')):
                self.tensorboard_writer.add_scalar('B_component_losses/train/{}'.format(c_lbl),
                                                   loss_components.detach()[:, c_idx].sum(),
                                                   iteration)

        if self.export_config.run_loss_updates:
            self.write_loss_updates(old_loss=loss.item(), new_loss=new_loss.sum(), old_assignments=assignments,
                                    new_assignments=new_assignments, iteration=iteration)

            if self.export_config.export_activations and \
                    self.export_config.write_activation_condition(iteration, epoch):
                self.retrieve_and_write_batch_activations(batch_input=full_input, iteration=iteration,
                                                          get_activations_fcn=get_activations_fcn)
        return eval_metrics

    def run_post_val_iteration(self, imgs, inst_lbl, score, sem_lbl, assignments: LossMatchAssignments,
                               should_visualize, data_to_img_transformer):
        """
        data_to_img_transformer: img_untransformed, lbl_untransformed = f(img, lbl) : e.g. - resizes, etc.
        """
        true_labels = []
        pred_labels = []
        segmentation_visualizations = []
        score_visualizations = []

        softmax_scores = F.softmax(score, dim=1).detach().cpu().numpy()
        inst_lbl_pred = score.detach().max(dim=1)[1].cpu().numpy()[:, :, :]
        lbl_true_sem, lbl_true_inst = (sem_lbl.detach().cpu(), inst_lbl.detach().cpu())
        if DEBUG_ASSERTS:
            assert inst_lbl_pred.shape == lbl_true_inst.shape
        for idx, (img, sem_lbl, inst_lbl, lp) in enumerate(zip(imgs, lbl_true_sem, lbl_true_inst, inst_lbl_pred)):
            # runtime_transformation needs to still run the resize, even for untransformed img, lbl pair
            img_untransformed, lbl_untransformed = data_to_img_transformer(img, (sem_lbl, inst_lbl)) \
                if data_to_img_transformer is not None \
                else (img, (sem_lbl, inst_lbl))
            sem_lbl_np, inst_lbl_np = lbl_untransformed
            assigned_sem_vals = [self.instance_problem.semantic_instance_class_list[c]
                                 for c in assignments.model_channels[idx, :].long()]
            assigned_inst_vals = [self.instance_problem.instance_count_id_list[c]
                                  for c in assignments.model_channels[idx, :]]
            lt_combined = self.gt_tuple_to_channelwise_combined(sem_lbl_np, inst_lbl_np,
                                                                assigned_sem_vals=assigned_sem_vals,
                                                                assigned_inst_vals=assigned_inst_vals)
            true_labels.append(lt_combined)
            pred_labels.append(lp)
            if should_visualize:
                segmentation_viz, score_viz = self.visualize_one_img_prediction(
                    img_untransformed, lp, lt_combined, softmax_scores, true_labels, idx)
                score_visualizations.append(score_viz)
                segmentation_visualizations.append(segmentation_viz)
        return true_labels, pred_labels, segmentation_visualizations, score_visualizations

    def compute_eval_metrics(self, label_trues, label_preds):
        eval_metrics_list = instanceseg.utils.misc.label_accuracy_score(label_trues, label_preds,
                                                                        n_class=self.instance_problem.n_classes)
        return eval_metrics_list

    def gt_tuple_to_channelwise_combined(self, sem_lbl, inst_lbl, assigned_sem_vals=None, assigned_inst_vals=None):
        # semantic_instance_class_list = self.instance_problem.semantic_instance_class_list
        # instance_count_id_list = self.instance_problem.instance_count_id_list
        return instance_utils.combine_semantic_and_instance_labels(sem_lbl, inst_lbl,
                                                                   assigned_sem_vals, assigned_inst_vals)

    @staticmethod
    def untransform_data(data_loader, img, lbl):
        if data_loader.dataset.runtime_transformation is not None:
            runtime_transformation_undo = runtime_transformations.GenericSequenceRuntimeDatasetTransformer(
                [t for t in (data_loader.dataset.runtime_transformation.transformer_sequence or [])
                 if isinstance(t, runtime_transformations.BasicRuntimeDatasetTransformer)])
            img_untransformed, lbl_untransformed = runtime_transformation_undo.untransform(img, lbl)
        else:
            img_untransformed, lbl_untransformed = img, lbl
        return img_untransformed, lbl_untransformed

    def export_channelvals2d_as_id2rgb(self, labels_as_batch_nparray, output_directory, image_names):
        batch_sz = labels_as_batch_nparray.shape[0]
        for img_idx in range(batch_sz):
            lbl = labels_as_batch_nparray[img_idx, ...]
            sem_l, inst_l = self.instance_problem.decompose_semantic_and_instance_labels_with_original_sem_ids(lbl)
            img = self.convert_lbl_to_image_with_id2rgb(255 * sem_l + inst_l)
            out_file = os.path.join(output_directory, image_names[img_idx])
            visualization_utils.write_image(out_file, img)

    def export_inst_sem_lbls_as_id2rgb(self, sem_lbls_as_batch_nparray, inst_lbls_as_batch_nparray, output_directory,
                                       image_names):
        batch_sz = sem_lbls_as_batch_nparray.shape[0]
        assert batch_sz == inst_lbls_as_batch_nparray.shape[0]
        for img_idx in range(batch_sz):
            sem_l = sem_lbls_as_batch_nparray[img_idx, ...]
            inst_l = inst_lbls_as_batch_nparray[img_idx, ...]
            img = self.convert_lbl_to_image_with_id2rgb(255 * sem_l + inst_l)
            out_file = os.path.join(output_directory, image_names[img_idx])
            visualization_utils.write_image(out_file, img)

    def export_rgb_images(self, images_as_batch_nparray, output_directory, image_names):
        batch_sz = images_as_batch_nparray.shape[0]
        for img_idx in range(batch_sz):
            img = images_as_batch_nparray[img_idx, ...]
            out_file = os.path.join(output_directory, image_names[img_idx])
            visualization_utils.write_image(out_file, img)

    def write_rgb_image(self, out_file, img):
        visualization_utils.write_image(out_file, img)

    @staticmethod
    def load_rgb_predictions_or_gt_to_id(in_file):
        rgb_img = visualization_utils.read_image(in_file)
        lbl = rgb2id(rgb_img)
        # convert void class
        lbl[255 + 256 * 255 + 256 * 256 * 255] = -1
        return

        # if self.exporter.export_config.export_activations:
        #     try:
        #         iteration = self.state.iteration
        #         upscore8_grad = self.model.upscore8.weight.grad
        #         for channel_idx, channel_name in enumerate(self.instance_problem.get_channel_labels()):
        #             self.exporter.tensorboard_writer.add_histogram(
        #                 'Z_upscore8_gradients/{}'.format(channel_idx),
        #                 upscore8_grad[channel_idx, :, :, :], self.state.iteration)
        #         score_pool4_weight_grad = self.model.score_pool4.weight.grad
        #         score_pool4_bias_grad = self.model.score_pool4.bias.grad
        #         assert len(score_pool4_bias_grad) == self.instance_problem.n_classes
        #         for channel_idx, channel_name in enumerate(self.instance_problem.get_channel_labels()):
        #             self.exporter.tensorboard_writer.add_histogram(
        #                 'Z_score_pool4_weight_gradients/{}'.format(channel_idx),
        #                 score_pool4_weight_grad[channel_idx, :, :, :], self.state.iteration)
        #             self.exporter.tensorboard_writer.add_histogram(
        #                 'score_pool4_bias_gradients/{}'.format(channel_idx),
        #                 score_pool4_bias_grad[channel_idx], self.state.iteration)
        #     except:
        #         import ipdb;
        #         ipdb.set_trace()
        #         raise

    @staticmethod
    def convert_lbl_to_image_with_id2rgb(lbl, permutation=None, void_value=-1):
        """
        Returns
        img_array: ndarray
            Visualized image.
        """

        # Generate funky pixels for void class
        mask_unlabeled = lbl == void_value
        lbl[mask_unlabeled] = 255
        # lbl_true[mask_unlabeled] = 0
        if permutation is not None:
            assert len(permutation.shape) == 1, 'Debug this -- assumed one image here.'
            lbl_permuted = instance_utils.permute_labels(lbl, permutation[np.newaxis, :])
        else:
            lbl_permuted = lbl
        viz = id2rgb(lbl_permuted)
        viz[:, :, 0][mask_unlabeled] = 255
        viz[:, :, 1][mask_unlabeled] = 255
        viz[:, :, 2][mask_unlabeled] = 255
        return viz


