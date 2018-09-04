from instanceseg.analysis import visualization_utils
import datetime
import os
import os.path as osp
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytz
import torch
import tqdm

import instanceseg.utils.display as display_pyutils
import instanceseg.utils.export
from instanceseg.utils.misc import flatten_dict

MY_TIMEZONE = 'America/New_York'


def should_write_activations(iteration, epoch, interval_validate):
    if iteration < 3000:
        return True
    else:
        return False


class ExportConfig(object):
    def __init__(self, export_activations=None, activation_layers_to_export=(), write_instance_metrics=False,
                 run_loss_updates=True):
        self.export_activations = export_activations
        self.activation_layers_to_export = activation_layers_to_export
        self.write_instance_metrics = write_instance_metrics
        self.run_loss_updates = run_loss_updates

        self.write_activation_condition = should_write_activations
        self.which_heatmaps_to_visualize = 'same semantic'  # 'all'


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

    def __init__(self, out_dir, instance_problem, export_config: ExportConfig = None,
                 tensorboard_writer=None, metric_makers=None):

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

        # Logging parameters
        self.timestamp_start = datetime.datetime.now(pytz.timezone(MY_TIMEZONE))

        self.val_losses_stored = []
        self.train_losses_stored = []
        self.joint_train_val_loss_mpl_figure = None  # figure for plotting losses on same plot
        self.iterations_for_losses_stored = []

        self.metric_makers = metric_makers

        # Writing activations

        self.run_loss_updates = True

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
            display_pyutils.set_my_rc_defaults()

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
            plt.ylim(ymin=ymin, ymax=ymax)
            plt.xlim(xmin=(last_x - ylim_buffer_size - 1), xmax=last_x)
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

    def write_loss_updates(self, old_loss, new_loss, old_pred_permutations, new_pred_permutations, iteration):
        loss_improvement = old_loss - new_loss
        num_reassignments = np.sum(new_pred_permutations != old_pred_permutations)
        self.tensorboard_writer.add_scalar('eval_metrics/train_batch_loss_improvement', loss_improvement,
                                           iteration)
        self.tensorboard_writer.add_scalar('eval_metrics/reassignment',
                                           num_reassignments, iteration)

    def compute_and_write_instance_metrics(self, model, iteration):
        if self.tensorboard_writer is not None:
            for split, metric_maker in tqdm.tqdm(self.metric_makers.items(), desc='Computing instance metrics',
                                                 total=len(self.metric_makers.items()), leave=False):
                metric_maker.clear()
                metric_maker.compute_metrics(model)
                metrics_as_nested_dict = metric_maker.get_aggregated_scalar_metrics_as_nested_dict()
                metrics_as_flattened_dict = flatten_dict(metrics_as_nested_dict)
                for name, metric in metrics_as_flattened_dict.items():
                    self.tensorboard_writer.add_scalar('instance_metrics_{}/{}'.format(split, name), metric, iteration)
                histogram_metrics_as_nested_dict = metric_maker.get_aggregated_histogram_metrics_as_nested_dict()
                histogram_metrics_as_flattened_dict = flatten_dict(histogram_metrics_as_nested_dict)
                if iteration != 0:  # screws up the axes if we do it on the first iteration with weird inits
                    # if 1:
                    for name, metric in tqdm.tqdm(histogram_metrics_as_flattened_dict.items(),
                                                  total=len(histogram_metrics_as_flattened_dict.items()),
                                                  desc='Writing histogram metrics', leave=False):
                        if torch.is_tensor(metric):
                            self.tensorboard_writer.add_histogram('instance_metrics_{}/{}'.format(split, name),
                                                                  metric.numpy(), iteration, bins='auto')
                        elif isinstance(metric, np.ndarray):
                            self.tensorboard_writer.add_histogram('instance_metrics_{}/{}'.format(split, name),
                                                                  metric, iteration, bins='auto')
                        elif metric is None:
                            import ipdb;
                            ipdb.set_trace()
                            pass
                        else:
                            raise ValueError('I\'m not sure how to write {} to tensorboard_writer (name is '
                                             '{}'.format(type(metric), name))

    def save_checkpoint(self, epoch, iteration, model, optimizer, best_mean_iu, out_dir=None,
                        out_name='checkpoint.pth.tar'):
        out_dir = out_dir or self.out_dir
        checkpoint_file = osp.join(out_dir, out_name)
        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'arch': model.__class__.__name__,
            'optim_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
            'best_mean_iu': best_mean_iu,
        }, checkpoint_file)
        return checkpoint_file

    def copy_checkpoint_as_best(self, current_checkpoint_file, out_dir=None, out_name='model_best.pth.tar'):
        out_dir = out_dir or self.out_dir
        best_checkpoint_file = osp.join(out_dir, out_name)
        shutil.copy(current_checkpoint_file, best_checkpoint_file)
        return best_checkpoint_file

    def visualize_one_img_prediction(self, img_untransformed, lp, lt_combined, pp, softmax_scores, true_labels, idx):
        # Segmentations
        segmentation_viz = visualization_utils.visualize_segmentation(
            lbl_pred=lp, lbl_true=lt_combined, pred_permutations=pp, img=img_untransformed,
            n_class=self.instance_problem.n_classes, overlay=False)
        # Scores
        sp = softmax_scores[idx, :, :, :]
        # TODO(allie): Fix this -- bug(?!)
        lp = np.argmax(sp, axis=0)
        if self.export_config.which_heatmaps_to_visualize == 'same semantic':
            inst_sem_classes_present = torch.np.unique(true_labels)
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
                                                           pred_permutations=pp,
                                                           n_class=self.instance_problem.n_classes,
                                                           score_vis_normalizer=sp.max(),
                                                           channel_labels=channel_labels,
                                                           channels_to_visualize=channels_to_visualize,
                                                           input_image=img_untransformed)
        return segmentation_viz, score_viz
